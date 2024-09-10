#!/bin/bash

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

export LLM_CONFIG=config/gpt2.yaml  # base config

python3 train.py \
    --out_dir=gs://uscentral2stuff/llm-jax/$EXPERIMENT \
    --attempt_to_load_checkpoint \
    --hellaswag_eval_interval=500 \
    --checkpoint_interval=1000 \
    --train_steps=100000 \
    --batch_size=256 \
    --optimizer.gradient_accumulation_steps=1 \
    --compute_dtype=bfloat16 \
    --params_dtype=float32 \
    --optimizer.type=affine \
    --optimizer.learning_rate=0.003 \
    --optimizer.warmup_steps=0 \
    --optimizer.weight_decay=0.01 \
    --optimizer.grad_clip=1.0 \
    --optimizer.max_size_triangular=16384 \
    --optimizer.max_skew_triangular=16 \
    --optimizer.precond_lr=0.1 \
    --optimizer.precond_init_scale=1.0 \
    --optimizer.preconditioner_dtype=float32