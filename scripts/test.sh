#!/bin/bash

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

export XLA_FLAGS="--xla_force_host_platform_device_count=2"
export LLM_CONFIG=config/gemma2.yaml  # base config


python3 train.py \
    --out_dir=/Users/evanwalters/llm_testing/run_checkpointing_test \
    --attempt_to_load_checkpoint \
    --compute_dtype=float32 \
    --params_dtype=float32 \
    --min_size_to_shard_mb=0 \
    --train_steps=200 \
    --hellaswag_eval_interval=100 \
    --checkpoint_interval=100 \
    --keep_checkpoints=2 \
    --batch_size=2 \
    --optimizer.type=affine \
    --optimizer.learning_rate=0.001 \
    --optimizer.warmup_steps=20 \
    --optimizer.preconditioner_dtype=float32 \
    --model.block_size=64 \
    --model.model_type=gemma2_test