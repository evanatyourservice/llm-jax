#!/bin/bash
# single host TPU run script for 125M model

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

python3 main.py \
    --experiment_name=$EXPERIMENT \
    --out_dir=gs://uscentral2stuff/llm-jax \
    --attempt_to_load_checkpoint \
    --hellaswag_eval_interval=1000 \
    --checkpoint_interval=1000 \
    --train_steps=150000 \
    --batch_size=256 \
    --gradient_accumulation_steps=2 \
    --compute_dtype=bfloat16 \
    --params_dtype=float32 \
    --profile \
    --model.block_size=2048 \
    --model.sliding_window_size=1024 \
    --model.scan_layers \
    --model.scan_unroll=3 \
    --model.remat \
    --model.no_remat_everything \
    --optimizer.type=kron \
    --optimizer.learning_rate=0.003 \
    --optimizer.warmup_steps=1000 \
    --optimizer.weight_decay=0.01 \
    --optimizer.grad_clip=1.0 \
    --optimizer.preconditioner_update_probability=0.03 \
    --optimizer.max_size_triangular=8192 \
    --optimizer.max_skew_triangular=10 \
    --optimizer.precond_lr=0.3 \
    --optimizer.preconditioner_dtype=float32