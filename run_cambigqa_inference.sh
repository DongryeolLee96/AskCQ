#!/bin/bash
set -e
set -f 
conda activate cq_test
out="out"
data_dir="" # Data directory for cq train/dev file, and related passage file

train_file="${data_dir}/data/cq_v2/cq_train_after_annotation.json"
dev_file="${data_dir}/data/cq_v2/cq_dev_after_annotation.json"
train_passage_dir="${data_dir}/data/wikipedia_split/rel_psg_input_ids_bart_train.pkl"
dev_passage_dir="${data_dir}/data/wikipedia_split/rel_psg_input_ids_bart_dev.pkl"


TASK="cqa"
output_dir="${out}/${TASK}"

MAX_QUESTION_LENGTH=128
gradient_accumulation_steps=2  # Orignally, 1
train_batch_size=10  # Orignally, 20
eval_period=500  # Originally, 500. FYI steps<2000 take for one epoch.
num_train_epochs=100  # We set 100 epochs. (~ 108,000 steps)
max_token_nums=1024  # Originally, 1024 maximum model input tokens' nums
checkpoint_folder="" # Checkpoint folder for trained CQA model
#checkpoint_folder="out/cqa/nq-bart-large-24-0"
checkpoint="${checkpoint_folder}/best-model.pt" # NQ pretrained BART-large from AMBIGQA
#checkpoint="${checkpoint_folder}/best-model.pt" # DQ-Answer finetuned NQ pretrained BART-large
bert_name="facebook/bart-large"

MA_type=with_groundtruth_answers # with_groundtruth_answers, without_answers, with_predicted_answers
dq_type=gold_cq # gold_dq or pred_cq or gold_cq
set -x

if [[ $dq_type = "gold_cq" || $dq_type = "gold_dq" ]]; then 
    MA_type=with_groundtruth_answers

elif [[ $dq_type = "pred_cq" && $MA_type = "with_groundtruth_answers" ]]; then 
    pred_cq_file="" # Directory for pred_cqg_with_groundtruth_answers.json
    additional_args=(
        --pred_cq_file "$pred_cq_file"
    )

elif [[ $dq_type = "pred_cq" && $MA_type = "with_predicted_answers" ]]; then 
    pred_cq_file="" # Directory for pred_cqg_with_predicted_answers.json
    additional_args=(
        --pred_cq_file "$pred_cq_file"
    )

# above two case use model 19904

#model 20368 case - input: AQ + RP

elif [[ $dq_type = "pred_cq" && $MA_type = "without_answers" ]]; then 
    pred_cq_file="" # Directory for pred_cqg_without_answers.json
    additional_args=(
        --pred_cq_file "$pred_cq_file"
    )


fi

python cli.py --do_predict --task ${TASK} \
    --train_file ${train_file} \
    --dev_file ${dev_file} \
    --output_dir ${output_dir} \
    --bert_name $bert_name \
    --discard_not_found_answers \
    --train_batch_size ${train_batch_size} \
    --num_train_epochs ${num_train_epochs} \
    --max_token_nums ${max_token_nums} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --dev_batch_size 8 \
    --eval_period ${eval_period} \
    --wait_step 10 --wiki_2020 --max_question_length ${MAX_QUESTION_LENGTH} \
    --train_passage_dir $train_passage_dir\
    --dev_passage_dir $dev_passage_dir \
    --checkpoint $checkpoint\
    --checkpoint_folder $checkpoint_folder \
    --MA_type $MA_type\
    --dq_type $dq_type \
    --verbose \
    --jobid ${SLURM_JOB_ID}\
    "${additional_args[@]}"
