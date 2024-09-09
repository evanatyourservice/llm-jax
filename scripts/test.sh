#!/bin/bash

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

export XLA_FLAGS="--xla_force_host_platform_device_count=2"
export LLM_CONFIG=config/gpt2.yaml  # base config


python3 train.py \
    --out_dir=/Users/evanwalters/llm_testing/run_checkpointing_test \
    --attempt_to_load_checkpoint \
    --compute_dtype=bfloat16 \
    --params_dtype=bfloat16 \
    --min_size_to_shard_mb=0 \
    --train_steps=500 \
    --hellaswag_eval_interval=100 \
    --checkpoint_interval=100 \
    --keep_checkpoints=2 \
    --batch_size=2 \
    --optimizer.type=affine \
    --optimizer.learning_rate=0.001 \
    --optimizer.warmup_steps=20 \
    --optimizer.preconditioner_dtype=bfloat16 \
    --model.block_size=64 \
    --model.num_layers=4 \
    --model.num_heads=2 \
    --model.num_embeds=8