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

TASK="cqg" #cqg, cqa, MA_prediction
output_dir="${out}/${TASK}"

MAX_QUESTION_LENGTH=128
gradient_accumulation_steps=2  # Orignally, 1
train_batch_size=10  # Orignally, 20
eval_period=500  # Originally, 500. FYI steps<2000 take for one epoch.
num_train_epochs=100  # We set 100 epochs. (~ 108,000 steps)
max_token_nums=1024  # Originally, 1024 maximum model input tokens' nums
MA_type=without_answers # with_groundtruth_answers", "without_answers", "with_predicted_answers
dq_type=pred_cq # gold_dq or pred_cq or gold_cq
additional_args=()
#checkpoint_folder="out/cqg/cqg-bart_large-19904"



bert_name="facebook/bart-large"
set -x

if [[ $MA_type = "without_answers" ]]; then 
    checkpoint_folder="" # checkpoint folder for CQG model trained without_answers

elif [[ $MA_type = "with_groundtruth_answers" ]]; then 
    checkpoint_folder="" # checkpoint folder for CQG model trained with_groundtruth_answers

elif [[ $MA_type = "with_predicted_answers" ]]; then 
    checkpoint_folder="" # checkpoint folder for CQG model trained with_predicted_answers
    pred_answers_file="" # directory to pred_MA_prediction.json file
    additional_args=(
        --pred_answers_file "$pred_answers_file"
    )
fi
checkpoint="${checkpoint_folder}/best-model.pt"
echo "EVALUATING" $1
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
