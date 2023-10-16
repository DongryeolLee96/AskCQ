import torch
from torch.nn.utils.rnn import pad_sequence
import pickle
import json
from torch.utils.data import Dataset
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BartTokenizer
model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
reranker_tokenizer=AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
model.to('cuda')

bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
bart_tokenizer.add_tokens(["<SEP>"])
def parse_clarification_question(cq):
    temp = cq.split(":", 1)
    if len(temp) != 2:
        return "invalid form", ["invalid form"]
    category, option_string = temp
    # def _extract_ap(ap):
    #     if “Could you clarify ‘” in ap:
    #         temp = “could you clarify ‘”
    #         ap = ap[len(temp):-1]
    #     elif “Could you clarify’” in ap:
    #         temp = “could you clarify’”
    #         ap = ap[len(temp):-1]
    #     if “be more specific” in ap or len(ap) == 0:
    #         ap = “SPECIFY”
    #     return ap
    def _extract_options(option_string):
        options = []
        flag = False
        def _is_valid(ch, future):
            escapes = [", ", " ,", ",",
                       ", or ", ",or ", ", or", ",or",
                       ", ", ",", " ,",
                       " or ", "or ", " or",
                       "?", " ?"]
            if any([future.startswith(es) for es in escapes]):
                return False
            return True
        for idx, ch in enumerate(option_string):
            if "'" in ch and not flag:
                flag = True
                options.append("")
            elif flag and _is_valid(ch, option_string[idx:]):
                options[-1] += ch
                continue
            elif flag:
                flag=False
        return options
    # ap = _extract_ap(ap)
    options = []
    for token in option_string.split(", or"):
        for option in token.split(","):
            options.append(option.strip(" ?"))
    # options = _extract_options(option_string)
    return category, options

class CQGDataset(Dataset):
    def __init__(self, data, passage_data, tokenizer, max_token_nums, process_type, MA_type, pred_answers=None):
        if MA_type == "with_predicted_answers":
            assert pred_answers is not None
        self.tokenizer=tokenizer
        self.max_token_nums=max_token_nums
        self.process_type=process_type
        self.MA_type=MA_type
        self.data=self.process_data(data, passage_data, pred_answers)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        label_list, text_list, attention_list =[], [], []
        for b in batch:
            text_list.append(torch.tensor(b['input_ids']))
            label_list.append(torch.tensor(b['labels']))
            attention_list.append(torch.tensor(b['attention_mask']))
       
        if self.process_type=='train':
            return {'input_ids': pad_sequence(text_list, batch_first=True, padding_value=self.tokenizer.pad_token_id),
                    'attention_mask': pad_sequence(attention_list, batch_first=True, padding_value=self.tokenizer.pad_token_id),
                    'labels':pad_sequence(label_list, batch_first=True, padding_value=-100)}
        elif self.process_type=='inference':
            return {'input_ids': pad_sequence(text_list, batch_first=True, padding_value=self.tokenizer.pad_token_id),
                    'attention_mask': pad_sequence(attention_list, batch_first=True, padding_value=self.tokenizer.pad_token_id),
                    'labels':pad_sequence(label_list, batch_first=True, padding_value=0)}
        
    def process_data(self, data, passage_data, pred_answers=None):
        processed_data=[]
        detokenized_passages=[[bart_tokenizer.decode(p, skip_special_tokens=True) for p in rps] for rps in passage_data['input_ids']]
        if pred_answers is not None:
            assert len(data) == len(passage_data['input_ids']) == len(pred_answers)
        for i, (d, relevant_psg_input_ids) in enumerate(zip(data, passage_data['input_ids'])):
            
            cq_input=""
            cq_input+=d['question']+self.tokenizer.sep_token
            
            if not all([ann['type']=="multipleQAs" for ann in d['annotations']]):
                target='which one?'
            else:
                target=d['clarification_question'].strip()
            
                # For with_groundtruth_answers, our input to CQ generation model = AQ + Multiple Answers + Relevant Passages
                if self.MA_type =="with_groundtruth_answers":
                    for answer_reps in d['clarification_answers']:
                        cq_input+=answer_reps[0]+self.tokenizer.sep_token
                elif self.MA_type == "with_predicted_answers":
                    for answer in pred_answers[i].split("<SEP>"):
                        cq_input+=answer.strip()+self.tokenizer.sep_token
                # For without_answers, our input to CQ generation model = AQ + Relevant Passages
                elif self.MA_type =='without_answers':
                    pass
                else:
                    raise KeyError
            inputs=self.tokenizer(text=cq_input, text_target=target)

            rp_concat=[]
            for psg in detokenized_passages[i]:
                rp_concat.extend(self.tokenizer.encode(psg))
            # for i in relevant_psg_input_ids:
            #     rp_concat.extend(i)
            #     rp_concat.append(self.tokenizer.sep_token_id)

            inputs['input_ids']+=rp_concat
            inputs['input_ids']=inputs['input_ids'][:self.max_token_nums]
            inputs['attention_mask']=[1 for i in range(len(inputs['input_ids']))]
            processed_data.append(inputs)
            
        return processed_data



class CQADataset(Dataset):
    def __init__(self, data, passage_data, tokenizer, max_token_nums, process_type, dq_type, max_target_length, pred_cq=None):
        if dq_type == "pred_cq":
            assert pred_cq is not None
        self.tokenizer=tokenizer
        self.max_token_nums=max_token_nums
        self.process_type=process_type
        self.pred_cq=pred_cq
        self.max_target_length=max_target_length
        self.gold_answer_sets=[d['clarification_answers'] for d in data]
        self.gold_option_sets=[parse_clarification_question(d['clarification_question'])[1] for d in data]
        if pred_cq is not None:
            self.pred_option_sets=[parse_clarification_question(d)[1] for d in pred_cq]
        if process_type=="train":
            self.data=self.process_data_for_train(data, passage_data, dq_type)
        elif process_type=="inference":
            self.data=self.process_data_for_inference(data, passage_data, dq_type, pred_cq)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if self.process_type=="train":
            d=self.data[idx]
            d['labels']=random.choice(d['answer_set_ids'])
            return d
        else:
            return self.data[idx]

    def collate_fn(self, batch):
        if self.process_type=="train":
            label_list, text_list, attention_list= [], [], []
            for b in batch:
                text_list.append(torch.tensor(b['input_ids']))
                label_list.append(torch.tensor(b['labels']))
                attention_list.append(torch.tensor(b['attention_mask']))
            batched = {'input_ids': pad_sequence(text_list, batch_first=True, padding_value=self.tokenizer.pad_token_id),
                'attention_mask': pad_sequence(attention_list, batch_first=True, padding_value=self.tokenizer.pad_token_id),
                'labels':pad_sequence(label_list, batch_first=True, padding_value=-100)}
                
        else:
            text_list, attention_list, question_idx_list =[], [], []
            for b in batch:
                text_list.append(torch.tensor(b['input_ids']))
                attention_list.append(torch.tensor(b['attention_mask']))
                question_idx_list.append(b['question_idx'])
                # For train_set no gold_answer_set
            batched = {'input_ids': pad_sequence(text_list, batch_first=True, padding_value=self.tokenizer.pad_token_id),
                    'attention_mask': pad_sequence(attention_list, batch_first=True, padding_value=self.tokenizer.pad_token_id),
                    'question_idx': question_idx_list}
        return batched
    
    
    # input: DQ + Relevant Passages -> Target: Clarification Answer for corresponding DQ
    def process_data_for_train(self, data, passage_data, dq_type):
        processed_data=[]
        detokenized_passages=[[self.tokenizer.decode(p, skip_special_tokens=True) for p in rps] for rps in passage_data['input_ids']]
        for question_idx, (d, relevant_psg_input_ids) in enumerate(zip(data, passage_data['input_ids'])):
            assert len(d['dqs'])==len(d['clarification_answers'])
            if dq_type=="gold_cq":
                category, options = parse_clarification_question(d['clarification_question'])
                assert len(options)==len(d['clarification_answers'])
                category=category.strip()
                
            for idx, dq in enumerate(d['dqs']):
                # Create DQ + relevant_passages -> DQ_answer for each dq
                input=""
                if dq_type=="gold_dq":
                    input+=d['dqs'][idx]+self.tokenizer.sep_token
                    query = [input for i in range(100)]
                elif dq_type=="gold_cq":
                    input+=d['question']+" "+category+" "+options[idx]+"?"
                    query = [input for i in range(100)]
                else:
                    raise KeyError
                features= reranker_tokenizer(query, detokenized_passages[question_idx], padding=True, truncation=True, return_tensors='pt')
                features.to('cuda')
                model.eval()
                with torch.no_grad():
                    scores=model(**features).logits
                    
                reranked=[psg for score, psg in sorted(zip(scores, detokenized_passages[question_idx]), reverse=True)]
                target=d['clarification_answers'][idx]
                tokenized_target=self.tokenizer(target, max_length=self.max_target_length, truncation=True)['input_ids']

                inputs=self.tokenizer(text=input)
                rp_concat=[]
                for psg in reranked:
                    rp_concat.extend(self.tokenizer.encode(psg))
                    rp_concat.append(self.tokenizer.sep_token_id)
                # rp_concat=[]
                # for i in relevant_psg_input_ids:
                #     rp_concat.extend(i)
                #     rp_concat.append(self.tokenizer.sep_token_id)
            
                    
                inputs['input_ids']+=rp_concat
                inputs['input_ids']=inputs['input_ids'][:self.max_token_nums]
                inputs['attention_mask']=[1 for i in range(len(inputs['input_ids']))]
                inputs['answer_set_ids']=tokenized_target
                processed_data.append(inputs)
        return processed_data
    
    # Input: AQ revised by CQ (AQ+CQ+Option or AQ + CQ(with unique option))
    def process_data_for_inference(self, data, passage_data, dq_type, pred_cq):
        processed_data=[]
        detokenized_passages=[[self.tokenizer.decode(p, skip_special_tokens=True) for p in rps] for rps in passage_data['input_ids']]
        if pred_cq is not None:
            assert len(data) == len(passage_data['input_ids']) == len(pred_cq)
        
        for question_idx ,(d, relevant_psg_input_ids) in enumerate(zip(data, passage_data['input_ids'])):
            if pred_cq is not None:
                category, options = parse_clarification_question(pred_cq[question_idx])
            else:
                category, options = parse_clarification_question(d['clarification_question'])
            category=category.strip()
            # share the same RP-> detokenized_passages[question_idx]
            for idx, op in enumerate(options):              
                if dq_type=='pred_cq' or dq_type=="gold_cq":
                    input=d['question']+" "+category+" "+op+"?"
                    query = [input for i in range(100)]
                elif dq_type=="gold_dq":
                    input=d['dqs'][idx]
                    query =[input for i in range(100)]
                else:
                    raise KeyError
                features= reranker_tokenizer(query, detokenized_passages[question_idx], padding=True, truncation=True, return_tensors='pt')
                features.to('cuda')
                model.eval()
                with torch.no_grad():
                    scores=model(**features).logits
                
                reranked=[psg for score, psg in sorted(zip(scores, detokenized_passages[question_idx]), reverse=True)]
                inputs = self.tokenizer(text=input)
                rp_concat=[]
                for psg in reranked:
                    rp_concat.extend(self.tokenizer.encode(psg))
                    rp_concat.append(self.tokenizer.sep_token_id)
                
                # rp_concat=[]
                # for i in relevant_psg_input_ids:
                #     rp_concat.extend(i)
                #     rp_concat.append(self.tokenizer.sep_token_id)
                    
                inputs['input_ids']+=rp_concat
                inputs['input_ids']=inputs['input_ids'][:self.max_token_nums]
                inputs['attention_mask']=[1 for i in range(len(inputs['input_ids']))]
                inputs['question_idx']=question_idx
                processed_data.append(inputs)
        return processed_data


