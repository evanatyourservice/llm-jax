#!/bin/bash

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

export XLA_FLAGS="--xla_force_host_platform_device_count=2"
export LLM_CONFIG=config/mistral.yaml  # base config


python3 main.py \
    --out_dir=/Users/evanwalters/llm_testing/run_checkpointing_test \
    --attempt_to_load_checkpoint \
    --compute_dtype=float32 \
    --params_dtype=float32 \
    --min_size_to_shard_mb=0 \
    --train_steps=500 \
    --hellaswag_eval_interval=100 \
    --checkpoint_interval=100 \
    --keep_checkpoints=2 \
    --batch_size=2 \
    --optimizer.gradient_accumulation_steps=2 \
    --remat \
    --optimizer.type=affine \
    --optimizer.learning_rate=0.001 \
    --optimizer.warmup_steps=20 \
    --optimizer.preconditioner_dtype=float32 \
    --optimizer.preconditioner_update_probability=0.05 \
    --optimizer.precond_init_scale=0.01 \
    --optimizer.precond_lr=0.1 \
    --model.block_size=1024 \
    --model.sliding_window_size=64 \
    --model.num_layers=2 \
    --model.num_heads=4 \
    --model.num_embeds=8 \
    --model.head_dim=4 \
    --model.hidden_dim=8 \
    --model.num_kv_heads=2