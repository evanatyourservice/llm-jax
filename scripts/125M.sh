#!/bin/bash

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

python3 main.py \
    --experiment_name=$EXPERIMENT \
    --out_dir=gs://uscentral2stuff/llm-jax \
    --attempt_to_load_checkpoint \
    --hellaswag_eval_interval=1000 \
    --checkpoint_interval=1000 \
    --train_steps=50000 \
    --batch_size=256 \
    --gradient_accumulation_steps=2 \
    --compute_dtype=bfloat16 \
    --params_dtype=float32 \
    --profile \
    --model.block_size=2048 \
    --model.sliding_window_size=1024 \
    --model.scan_layers \
    --model.remat \
    --model.remat_everything \
    --optimizer.type=kron \
    --optimizer.learning_rate=0.001 \
    --optimizer.warmup_steps=1000 \
    --optimizer.weight_decay=0.1 \
    --optimizer.grad_clip=1.0