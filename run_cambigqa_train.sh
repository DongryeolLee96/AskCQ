#!/bin/bash
set -e
set -f 
conda activate cq
out="out"
# data_dir="/media/disk1/${USER}/openQA/AmbigQA"
data_dir="" # Data directory for cq train/dev file, and related passage file

train_file="${data_dir}/data/cq_v2/cq_train_after_annotation.json"
dev_file="${data_dir}/data/cq_v2/cq_dev_after_annotation.json"
train_passage_dir="${data_dir}/data/wikipedia_split/rel_psg_input_ids_bart_train.pkl"
dev_passage_dir="${data_dir}/data/wikipedia_split/rel_psg_input_ids_bart_dev.pkl"

output_dir="${out}/cqa-${SLURM_JOB_ID}"
TASK="cqa"
MAX_QUESTION_LENGTH=128
gradient_accumulation_steps=2  # Orignally, 1
train_batch_size=10  # Orignally, 20
eval_period=500  # Originally, 500. FYI steps<2000 take for one epoch.
num_train_epochs=100  # We set 100 epochs. (~ 108,000 steps)
max_token_nums=1024  # Originally, 1024 maximum model input tokens' nums
checkpoint="" # NQ pretrained BART-large from AMBIGQA
bert_name="facebook/bart-large"
dq_type=gold_cq
set -x

python cli.py --do_train --task ${TASK} \
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
    --verbose \
    --dq_type $dq_type \
    --checkpoint ${checkpoint}
