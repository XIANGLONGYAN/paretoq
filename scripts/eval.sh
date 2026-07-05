# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CUDA_VISIBLE_DEVICES=2 torchrun --nnodes=1 --nproc_per_node=1 --master_port=29519 train.py \
--contain_weight_clip_val True \
--use_lsq True \
--use_stableqat False \
--local_dir "/home/jiaqichen/data2/paretoq" \
--input_model_filename "/home/jiaqichen/data2/paretoq/models/1B-finetuned-4bit" \
--output_model_filename "1B-finetuned-4bit" \
--train_data_local_path "/home/jiaqichen/data2/dataset/wikitext_wikitext-2-raw-v1_train.jsonl" \
--eval_data_local_path "/home/jiaqichen/data2/dataset/wikitext_wikitext-2-raw-v1_test.jsonl" \
--do_train False \
--do_eval True \
--use_muon True \
--learning_rate 2e-5 \
--muon_learning_rate 8e-5 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir /home/jiaqichen/data2/paretoq/logging/4bit_finetune \
--output_dir /home/jiaqichen/data2/paretoq/output/4bit_finetune \
--num_train_epochs 1 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \
--save_total_limit 1 \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 False \
--gradient_checkpointing True \
--qat True \
--full_determinism True \
--dataloader_num_workers 0 \
--seed 42 \
--w_bits 4 \
--eval_ppl \
--eval_lm_eval \
--tasks "piqa,hellaswag,winogrande,arc_easy,arc_challenge" \
--eval_batch_size 64