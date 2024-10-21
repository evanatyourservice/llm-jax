#!/bin/bash

export XLA_FLAGS="--xla_force_host_platform_device_count=2"

python3 main.py \
    --out_dir=/Users/evanwalters/llm_testing \
    --no_attempt_to_load_checkpoint \
    --compute_dtype=float32 \
    --params_dtype=float32 \
    --model.min_size_to_shard_mb=0 \
    --train_steps=100 \
    --hellaswag_eval_interval=20 \
    --checkpoint_interval=20 \
    --batch_size=2 \
    --gradient_accumulation_steps=2 \
    --profile \
    --wandb.mode=offline \
    --optimizer.type=kron \
    --optimizer.schedule_free \
    --optimizer.learning_rate=0.001 \
    --optimizer.flat_lr \
    --optimizer.warmup_steps=20 \
    --optimizer.preconditioner_dtype=float32 \
    --optimizer.preconditioner_update_probability=0.03 \
    --optimizer.lax_map_scanned_layers \
    --optimizer.lax_map_batch_size=1 \
    --model.block_size=1024 \
    --model.sliding_window_size=512 \
    --model.num_layers=2 \
    --model.num_heads=4 \
    --model.num_embeds=8 \
    --model.head_dim=4 \
    --model.hidden_dim=8 \
    --model.num_kv_heads=2 \
    --model.scan_layers