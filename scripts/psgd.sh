#!/bin/bash

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

export LLM_CONFIG=config/llama3.yaml  # base config

python3 train.py \
    --out_dir=gs://uscentral2stuff/llm-jax/$EXPERIMENT \
    --batch_size=1024 \
    --optimizer.gradient_accumulation_steps=1 \
    --compute_dtype=bfloat16 \
    --params_dtype=bfloat16 \
    --optimizer.type=affine \
    --optimizer.learning_rate=0.001 \
    --optimizer.weight_decay=0.01 \
    --optimizer.grad_clip=0.0 \
    --optimizer.preconditioner_update_probability=0.5 \
    --optimizer.max_size_triangular=1000000000 \
    --optimizer.max_skew_triangular=0 \
    --optimizer.precond_lr=0.1 \
    --optimizer.precond_init_scale=1.0