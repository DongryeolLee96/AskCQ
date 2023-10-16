import os
import torch

from tqdm import tqdm
from transformers import BartTokenizer, AlbertTokenizer, BertTokenizer
from transformers import BartConfig, AlbertConfig, BertConfig
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup

from CQDataset import CQGDataset, CQADataset, parse_clarification_question

from models.span_predictor import SpanPredictor
from models.seq2seq_with_prefix import MyBartWithPrefix
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM, BartForConditionalGeneration
from transformers.trainer_utils import set_seed

import pickle
import json
from torch.utils.data import DataLoader
from CQ_evaluate import CQA_evaluate, CQG_evaluate
import wandb
import random

def run(args, logger):
    set_seed(1004)
    args.is_seq2seq = 'bart' in args.bert_name
    if 'bart' in args.bert_name:
        tokenizer = BartTokenizer.from_pretrained(args.bert_name)
        tokenizer.add_tokens(["<SEP>"])
        Model = MyBartWithPrefix if args.do_predict and args.nq_answer_as_prefix else BartForConditionalGeneration
        Config = BartConfig
        args.append_another_bos = True
    elif 'bert' in args.bert_name:
        tokenizer = BertTokenizer.from_pretrained(args.bert_name)
        Model = SpanPredictor
        Config = BertConfig
    else:
        raise NotImplementedError()

    with open(args.dev_passage_dir, 'rb') as f:
        dev_passages=pickle.load(f)
    
    with open(args.dev_file) as f:
        dev_data = json.load(f)
    
    def _load_from_checkpoint(checkpoint):
        def convert_to_single_gpu(state_dict):
            if "model_dict" in state_dict:
                state_dict = state_dict["model_dict"]
            def _convert(key):
                if key.startswith('module.'):
                    return key[7:]
                return key
            return {_convert(key):value for key, value in state_dict.items()}
        state_dict = convert_to_single_gpu(torch.load(checkpoint))
        model = Model(Config.from_pretrained(args.bert_name))
        if "bart" in args.bert_name:
            model.resize_token_embeddings(len(tokenizer))
        logger.info("Loading from {}".format(checkpoint))
        return model.from_pretrained(None, config=model.config, state_dict=state_dict)
    
    if args.do_train:        
        with open(args.train_passage_dir, 'rb') as f:
            train_passages=pickle.load(f)
    
        with open(args.train_file) as f:
            train_data = json.load(f)
        if args.task=="cqg":
            train_dataset = CQGDataset(train_data, train_passages, tokenizer=tokenizer, max_token_nums=args.max_token_nums, process_type='train', MA_type="with_groundtruth_answers")
            train_input=[]
            for d in train_dataset:
                train_input.append(tokenizer.decode(d['input_ids'], skip_special_tokens=True))
            with open('{}/train_input.json'.format(args.output_dir), 'w') as f:         
                json.dump(train_input, f)                
        elif args.task=="cqa":
            train_dataset = CQADataset(train_data, train_passages, tokenizer=tokenizer, max_token_nums=args.max_token_nums, process_type='train', dq_type=args.dq_type, max_target_length=args.max_answer_length, pred_cq=None)
            train_input=[]
            for d in train_dataset:
                train_input.append(tokenizer.decode(d['input_ids'], skip_special_tokens=True))
            with open('{}/{}_{}_train_input.json'.format(args.output_dir, args.task, args.dq_type), 'w') as f:         
                json.dump(train_input, f)   
        else:
            raise KeyError
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn, shuffle=False)
        print("train dataset size={}".format(len(train_dataset)))
    
        
    if args.do_train and args.skip_inference:
        dev_data = None
    else:
        if args.task=="cqg":
            if args.MA_type == "with_predicted_answers":
                with open(args.pred_answers_file) as f:
                    pred_answers = json.load(f)
            else:
                pred_answers=None
            dev_dataset = CQGDataset(dev_data, dev_passages, tokenizer=tokenizer, max_token_nums=args.max_token_nums, process_type='inference', MA_type=args.MA_type, pred_answers=pred_answers)
            dev_input=[]
            for d in dev_dataset:
                dev_input.append(tokenizer.decode(d['input_ids'], skip_special_tokens=True))
            with open('{}/{}_dev_input.json'.format(args.output_dir, args.MA_type), 'w') as f:         
                json.dump(dev_input, f)                
        elif args.task=="cqa":
            if args.dq_type =="pred_cq":
                with open(args.pred_cq_file) as f:
                    pred_cq = json.load(f)
            else:
                pred_cq=None
            dev_dataset = CQADataset(dev_data, dev_passages, tokenizer=tokenizer, max_token_nums=args.max_token_nums, process_type='inference', dq_type=args.dq_type, max_target_length=args.max_answer_length,pred_cq=pred_cq)
            dev_input=[]
            for d in dev_dataset:
                dev_input.append(tokenizer.decode(d['input_ids'], skip_special_tokens=True))
            if args.dq_type =="pred_cq":
                with open('{}/{}_{}_{}_dev_input.json'.format(args.output_dir, args.task, args.dq_type, args.MA_type), 'w') as f:         
                    json.dump(dev_input, f)  
            elif args.dq_type =="gold_cq":
                with open('{}/{}_{}_dev_input.json'.format(args.output_dir, args.task, args.dq_type), 'w') as f:         
                    json.dump(dev_input, f)
        elif args.task=="MA_prediction":
            dev_dataset = CQGDataset(dev_data, dev_passages, tokenizer=tokenizer, max_token_nums=args.max_token_nums, process_type='inference', MA_type="without_answers")
        else:
            raise KeyError
        print("dev dataset size={}".format(len(dev_dataset)))
        dev_len=len(dev_dataset)

    if args.do_train:
        if args.checkpoint is not None:
            model = _load_from_checkpoint(args.checkpoint)
        else:
            model = Model.from_pretrained(args.bert_name)
        if "bart" in args.bert_name:
            model.resize_token_embeddings(len(tokenizer))
        if args.n_gpu>1:
            model = torch.nn.DataParallel(model)
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))  

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler =  get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=args.warmup_steps,
                                        num_training_steps=100000)
        train(args, logger, model, train_dataloader, dev_dataset, optimizer, scheduler, tokenizer, dev_len)

    if args.do_predict:
        checkpoint = os.path.join(args.output_dir, 'best-model.pt') if args.checkpoint is None else args.checkpoint
        model = _load_from_checkpoint(checkpoint)
        logger.info("Loading checkpoint from {}".format(checkpoint))
        if args.n_gpu>1 and 'bert' in args.bert_name:
            model = torch.nn.DataParallel(model)
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        model.eval()
        evaluation_result = inference(args, model, dev_dataset, tokenizer, dev_len, save_predictions=True)
        if args.task =="cqa":
            for k, v in evaluation_result.items():
                logger.info("%s on test data = %.2f" % (k, v*100))
        elif args.task =="cqg":
            for k, v in evaluation_result.items():
                logger.info("%s on test data = %.2f" % (k, v*100))

def train(args, logger, model, train_dataloader, dev_dataset, optimizer, scheduler, tokenizer, dev_len):
    model.train()
    global_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training=False

    for _ in range(args.resume_global_step):
        optimizer.step()
        scheduler.step()

    logger.info("Start training!")
    for epoch in range(int(args.num_train_epochs)):
        for idx ,batch in enumerate(train_dataloader):
            global_step += 1
            batch = {k:v.to(model.device) for k, v in batch.items()}
            output=model(**batch)
            loss=output.loss
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()
            
            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()    # We have accumulated enought gradients
                scheduler.step()
                model.zero_grad()

            if global_step % args.eval_period == 0:
                if args.skip_inference:
                    logger.info("Step %d (epoch %d) Train loss %.2f" % (
                            global_step,
                            epoch,
                            np.mean(train_losses)))
                    train_losses = []
                    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    torch.save(model_state_dict, os.path.join(args.output_dir,
                                                              "best-model-{}.pt".format(str(global_step).zfill(6))))
                else:
                    model.eval()
                    evaluation_result = inference(args, model, dev_dataset, tokenizer, dev_len)
                    logger.info("Step %d Train loss %.2f %s on epoch=%d" % (
                            global_step,
                            np.mean(train_losses),
                            " ".join([f"{name} {val:.2f}" for name, val in evaluation_result.items()]),
                            epoch))
                    wandb.log({'train/loss': np.mean(train_losses),'epoch': epoch, **evaluation_result}, step=global_step)
                    train_losses = []
                    # For CQG task -> BLEU score, for CQA task --> EM score
                    if args.task=='cqg':
                        best_metric ='CQ_BLEU'
                    elif args.task =='cqa':
                        best_metric = 'EM'
                    if best_accuracy < evaluation_result[best_metric]:
                        model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                        torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                        logger.info("Saving model with best %s: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" % \
                                (best_metric, best_accuracy*100.0, evaluation_result[best_metric]*100.0, epoch, global_step))
                        best_accuracy = evaluation_result[best_metric]
                        wait_step = 0
                        stop_training = False
                    else:
                        wait_step += 1
                        if wait_step >= args.wait_step:
                            stop_training = True
                            break
                    model.train()
        if stop_training:
            break

def inference(args, model, dev_dataset, tokenizer, dev_len, save_predictions=False):
    if "bart" in args.bert_name:
        print('Inference started')
        return inference_seq2seq(args, model, dev_dataset, tokenizer, dev_len, save_predictions)
    return inference_span_predictor(model, dev_dataset, save_predictions)

def inference_seq2seq(args, model, dev_dataset, tokenizer, dev_len, save_predictions=False):
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, collate_fn=dev_dataset.collate_fn, shuffle=False)
    predictions = []
    gold_set= []
    question_idx_list = []
    if args.task=="MA_prediction":
        max_answer_length = args.max_answer_length
        assert max_answer_length>=25 or not args.ambigqa
    else:
        max_answer_length = args.max_question_length
    if args.verbose:
        dev_dataloader = tqdm(dev_dataloader)
    
    for i, batch in enumerate(dev_dataloader):
        with torch.no_grad():
            outputs = model.generate(input_ids=batch['input_ids'].to(model.device),
                                     attention_mask=batch['attention_mask'].to(model.device),
                                     num_beams=4,
                                     max_length=max_answer_length,
                                     early_stopping=True,
                                    )
            
            
            predictions.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
            if args.task =="cqa":
                question_idx_list.extend(batch['question_idx'])
            elif args.task =="cqg":
                gold_set.extend(tokenizer.batch_decode(batch['labels'], skip_special_tokens=True))
                
                
    if args.task=="cqa":
        if args.dq_type=='pred_cq':
            evaluation_result, p_score_list, r_score_list = CQA_evaluate(dev_dataset.gold_answer_sets, predictions, dev_dataset.gold_option_sets, dev_dataset.pred_option_sets, args.dq_type, question_idx_list, dev_len)
        elif args.dq_type=='gold_cq' or args.dq_type=='gold_dq':
            evaluation_result, p_score_list, r_score_list = CQA_evaluate(dev_dataset.gold_answer_sets, predictions, dev_dataset.gold_option_sets, dev_dataset.gold_option_sets, args.dq_type, question_idx_list, dev_len)
        else:
            raise KeyError
    elif args.task=="cqg":
        evaluation_result = CQG_evaluate(gold_set, predictions, dev_len)
    else:
        evaluation_result =None
    if save_predictions:
        # For CQ gen model
        if args.task =="cqg":
            if not os.path.exists('{}/{}'.format(args.checkpoint_folder, args.jobid)):
                os.makedirs('{}/{}'.format(args.checkpoint_folder, args.jobid))
            with open('{}/{}/pred_{}_{}.json'.format(args.checkpoint_folder, args.jobid, args.task, args.MA_type), 'w') as f:
                json.dump(predictions, f)
            with open('{}/{}/gold_{}_{}.json'.format(args.checkpoint_folder, args.jobid, args.task, args.MA_type), 'w') as f:
                json.dump(gold_set, f)
            inputs=[]
            for batch in dev_dataloader:
                inputs.extend(tokenizer.batch_decode(batch['input_ids']))
            with open('{}/{}/input_{}_{}.json'.format(args.checkpoint_folder, args.jobid, args.task, args.MA_type), 'w') as f:
                json.dump(inputs, f)
        
        
        # For QA reader model 
        elif args.task =='cqa':
            if not os.path.exists('{}/{}'.format(args.checkpoint_folder, args.jobid)):
                os.makedirs('{}/{}'.format(args.checkpoint_folder, args.jobid))
            with open('{}/{}/pred_{}_{}_{}.json'.format(args.checkpoint_folder, args.jobid, args.task, args.dq_type, args.MA_type), 'w') as f:
                json.dump(predictions, f)
            with open('{}/{}/question_idx_list_{}_{}_{}.json'.format(args.checkpoint_folder, args.jobid, args.task, args.dq_type, args.MA_type), 'w') as f:
                json.dump(question_idx_list, f)
            with open('{}/{}/p_scores_{}_{}_{}.json'.format(args.checkpoint_folder, args.jobid, args.task, args.dq_type, args.MA_type), 'w') as f:
                json.dump(p_score_list, f)
            with open('{}/{}/r_scores_{}_{}_{}.json'.format(args.checkpoint_folder, args.jobid, args.task, args.dq_type, args.MA_type), 'w') as f:
                json.dump(r_score_list, f)
            inputs=[]
            for batch in dev_dataloader:
                inputs.extend(tokenizer.batch_decode(batch['input_ids']))
            with open('{}/{}/input_{}_{}_{}.json'.format(args.checkpoint_folder,args.jobid, args.task, args.dq_type, args.MA_type), 'w') as f:
                json.dump(inputs, f)
                
        # For Multiple Answers Prediction
        elif args.task =='MA_prediction':
            with open('{}/pred_{}.json'.format(args.checkpoint_folder, args.task), 'w') as f:
                json.dump(predictions, f)
            for batch in dev_dataloader:
                input= tokenizer.batch_decode(batch['input_ids'])
            with open('{}/input_{}.json'.format(args.checkpoint_folder, args.task), 'w') as f:
                json.dump(input, f)
                
    return evaluation_result

