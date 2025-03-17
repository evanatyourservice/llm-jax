#!/bin/bash

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

python3 main.py \
    --experiment_name=$EXPERIMENT \
    --out_dir=gs://optimizertesting/llm-jax \
    --attempt_to_load_checkpoint \
    --hellaswag_eval_interval=1000 \
    --checkpoint_interval=1000000000 \
    --train_steps=10000 \
    --batch_size=256 \
    --gradient_accumulation_steps=1 \
    --compute_dtype=bfloat16 \
    --params_dtype=float32 \
    --profile \
    --model.block_size=1024 \
    --model.num_layers=24 \
    --model.num_heads=8 \
    --model.num_kv_heads=4 \
    --model.head_dim=64 \
    --model.num_embeds=512 \
    --model.hidden_dim=1536 \
    --model.no_remat_everything \
    --model.no_use_ssm \
    --model.ssm_state_size=512 \
    --optimizer.type=kron \
    --optimizer.learning_rate=0.001 \
    --optimizer.warmup_steps=1000 \
    --optimizer.weight_decay=0.5