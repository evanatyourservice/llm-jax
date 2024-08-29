#!/bin/bash

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

export XLA_FLAGS="--xla_force_host_platform_device_count=2"
export LLM_CONFIG=config/llama3.yaml  # base config


python3 train.py \
    --out_dir=/Users/evanwalters/gpt_testing/$EXPERIMENT \
    --compute_dtype=float32 \
    --params_dtype=float32 \
    --shuffle_buffer_size=10 \
    --min_size_to_shard_mb=0 \
    --train_steps=200 \
    --hellaswag_eval_interval=100 \
    --batch_size=2 \
    --optimizer.type=affine \
    --optimizer.learning_rate=0.001 \
    --optimizer.warmup_steps=20 \
    --optimizer.preconditioner_dtype=float32 \
    --model.llama_huggingface_model_name="trl-internal-testing/tiny-random-LlamaForCausalLM" \
    --model.no_use_scan_mlp \
    --model.block_size=128 \
    --wandb.mode=disabled
