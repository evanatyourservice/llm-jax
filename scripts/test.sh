#!/bin/bash

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

export XLA_FLAGS="--xla_force_host_platform_device_count=2"
export GPT_CONFIG=config/gpt2.yaml  # base config

python3 train.py \
    --out_dir=/Users/evanwalters/gpt_testing/$EXPERIMENT \
    --train_pattern=/Users/evanwalters/owt_10k_data/train_??.tfrecord \
    --val_pattern=/Users/evanwalters/owt_10k_data/val_??.tfrecord \
    --min_size_to_shard_mb=0 \
    --train_steps=100 \
    --eval_interval=5 \
    --eval_steps=2 \
    --hs_eval_steps=2 \
    --batch_size=4 \
    --optimizer.type=affine \
    --optimizer.learning_rate=0.00001 \
    --optimizer.warmup_steps=20 \
    --model.n_embd=8 \
    --model.n_head=2 \
    --model.n_layer=1 \
    --model.n_inner=8 \
    --wandb.mode=disabled