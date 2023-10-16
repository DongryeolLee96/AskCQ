#!/bin/bash
conda activate cq

# export TASK_NAME=mrpc
# MODEL_NAME_OR_PATH="bert-base-cased"
# MODEL_NAME_OR_PATH="microsoft/deberta-v3-base"
MODEL_NAME_OR_PATH="/home/minwoolee/workspace/AmbigQA/out/bert-base-cased_20056/checkpoint-30000"
# MODEL_NAME_OR_PATH="microsoft/deberta-v3-large"

args=(
  --model_name_or_path "$MODEL_NAME_OR_PATH"
#   --task_name "$TASK_NAME"
  --train_file "/home/minwoolee/workspace/AmbigQA/data/ambiguity_detection/train.jsonl"
  --validation_file "/home/minwoolee/workspace/AmbigQA/data/ambiguity_detection/dev.jsonl"
  --test_file "/home/minwoolee/workspace/AmbigQA/data/ambiguity_detection/dev.jsonl"
  # --do_train
  # --do_eval
  --do_predict
  --fp16

  --max_seq_length 1024
  # --max_seq_length 512
  --per_device_train_batch_size 8
  # --gradient_accumulation_steps 1
  # --learning_rate 2e-5
  --learning_rate 1e-5
  --num_train_epochs 10
  --overwrite_cache

  --output_dir "./out/ad/pred_${SLURM_JOB_ID}"
  --evaluation_strategy "steps"
  --eval_steps 500
  --save_steps 500
  --save_total_limit 3
  --load_best_model_at_end
  --metric_for_best_model "loss"
)

set -x
python run_ambiguity_detection.py "${args[@]}"
