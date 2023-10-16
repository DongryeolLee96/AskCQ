#!/bin/bash
set -e
set -f
conda activate cq
data_dir="" # Data directory for cq train/dev file, and related passage file
dev_file="${data_dir}/data/cq_v2/cq_dev_after_annotation.json"
dev_passage_dir="${data_dir}/data/wikipedia_split/rel_psg_input_ids_bart_dev.pkl"
output_dir="" # Output dir
checkpoint_folder="out/ambignq-span-seq-gen"
checkpoint="${checkpoint_folder}/best-model.pt"

checkpoint="" # Checkpoint for SpanSeqGen checkpoint
bert_name="facebook/bart-large"
task="MA_prediction"

python3 cli.py --do_predict --task $task --checkpoint ${checkpoint} \
    --checkpoint_folder $checkpoint_folder\
    --dev_file ${dev_file} \
    --output_dir ${output_dir} \
    --dev_passage_dir $dev_passage_dir \
    --bert_name $bert_name \
    --discard_not_found_answers \
    --dev_batch_size 8 \
    --eval_period 500 --wait_step 10 --wiki_2020 --max_answer_length 25 --verbose
