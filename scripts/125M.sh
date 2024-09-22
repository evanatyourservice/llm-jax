#!/bin/bash

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

python3 main.py \
    --experiment_name=$EXPERIMENT \
    --out_dir=gs://uscentral2stuff/llm-jax \
    --attempt_to_load_checkpoint \
    --hellaswag_eval_interval=500 \
    --checkpoint_interval=1000 \
    --train_steps=100000 \
    --batch_size=128 \
    --gradient_accumulation_steps=1 \
    --compute_dtype=bfloat16 \
    --params_dtype=float32 \
    --optimizer.type=psgd \
    --optimizer.learning_rate=0.003 \
    --optimizer.warmup_steps=1000 \
    --optimizer.weight_decay=0.01 \
    --optimizer.grad_clip=1.0 \
    --optimizer.max_size_triangular=8192 \
    --optimizer.max_skew_triangular=10 \
    --optimizer.precond_lr=0.1 \
    --optimizer.precond_init_scale=0.1 \
    --optimizer.preconditioner_dtype=float32 \
    --optimizer.preconditioner_update_probability=0.04