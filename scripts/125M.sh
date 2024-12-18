#!/bin/bash

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

python3 main.py \
    --experiment_name=$EXPERIMENT \
    --out_dir=gs://optimizertesting/llm-jax \
    --attempt_to_load_checkpoint \
    --hellaswag_eval_interval=1000 \
    --checkpoint_interval=1000 \
    --train_steps=10000 \
    --batch_size=512 \
    --gradient_accumulation_steps=1 \
    --compute_dtype=bfloat16 \
    --params_dtype=float32 \
    --profile \
    --model.scan_layers \
    --model.remat \
    --model.no_remat_everything \
    --optimizer.type=kron \
    --optimizer.learning_rate=0.001 \
    --optimizer.flat_lr \
    --optimizer.warmup_steps=1000 \
    --optimizer.b1=0.95 \
    --optimizer.weight_decay=0.1