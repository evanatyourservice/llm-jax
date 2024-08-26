#!/bin/bash

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

export GPT_CONFIG=config/gpt2.yaml  # base config

python3 train.py \
    --out_dir=gs://uscentral1stuff/gpt_models/gpt_small/$EXPERIMENT \
    --train_pattern=gs://uscentral1stuff/openwebtext/train_??.tfrecord \
    --val_pattern=gs://uscentral1stuff/openwebtext/val_??.tfrecord \
    --batch_size=64 \
    --optimizer.gradient_accumulation_steps=2 \
    --optimizer.type=affine \
    --optimizer.psgd_use_hessian \
    --optimizer.learning_rate=0.01 \
    --optimizer.weight_decay=0.01 \
    --optimizer.grad_clip=0.0 \
    --optimizer.preconditioner_update_probability=0.5 \
    --optimizer.max_size_triangular=1000000000 \
    --optimizer.max_skew_triangular=0 \
    --optimizer.precond_lr=0.1 \
    --optimizer.precond_init_scale=1.0 \
    --optimizer.update_global_norm_clip=10000.0