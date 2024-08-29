#!/bin/bash

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

export LLM_CONFIG=config/llama3.yaml  # base config

python3 train.py \
    --out_dir=gs://uscentral2stuff/llm-jax/$EXPERIMENT \
    --batch_size=1024 \
    --optimizer.gradient_accumulation_steps=1 \
    --compute_dtype=bfloat16 \
    --optimizer.type=shampoo \
    --optimizer.learning_rate=0.001 \
    --optimizer.weight_decay=0.01