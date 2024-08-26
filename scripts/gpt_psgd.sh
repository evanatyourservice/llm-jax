#!/bin/bash

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

export GPT_CONFIG=config/gpt2.yaml  # base config

python3 train.py \
    --out_dir=gs://uscentral1stuff/gpt_models/gpt_small/$EXPERIMENT \
    --train_pattern=gs://uscentral1stuff/openwebtext/train_??.tfrecord \
    --val_pattern=gs://uscentral1stuff/openwebtext/val_??.tfrecord \
    --batch_size=192 \
    --bfloat16_compute \
    --optimizer.type=affine \
    --optimizer.learning_rate=0.0014 \
    --optimizer.weight_decay=0.01 \
    --optimizer.preconditioner_update_probability=0.5 \
    --optimizer.max_size_triangular=0 \
    --optimizer.max_skew_triangular=0 \
    --optimizer.precond_lr=0.1 \
    --optimizer.precond_init_scale=1.0 \
    --optimizer.update_global_norm_clip=10000.0 \
