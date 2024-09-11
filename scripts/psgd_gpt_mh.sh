#!/bin/bash
# usage: bash scripts/multihost.sh <wandb_api_key> <huggingface_token>

WANDB_API_KEY=$1
HF_TOKEN=$2

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

gcloud compute tpus tpu-vm ssh --zone "us-central2-b" "LLaMA" --project "my-phd-research-o" \
--worker=all --command="
export WANDB_API_KEY=$WANDB_API_KEY && \
export HF_TOKEN=$HF_TOKEN && \

export LLM_CONFIG=config/gpt2.yaml && \

cd llm-jax && \

python3 main_multihost.py \
    --experiment_name=run_2024-09-10_22-50-29 \
    --out_dir=gs://uscentral2stuff/llm-jax \
    --attempt_to_load_checkpoint \
    --hellaswag_eval_interval=500 \
    --checkpoint_interval=1000 \
    --train_steps=100000 \
    --batch_size=512 \
    --optimizer.gradient_accumulation_steps=1 \
    --compute_dtype=bfloat16 \
    --params_dtype=float32 \
    --optimizer.type=affine \
    --optimizer.learning_rate=0.004 \
    --optimizer.warmup_steps=500 \
    --optimizer.weight_decay=0.01 \
    --optimizer.grad_clip=1.0 \
    --optimizer.max_size_triangular=8192 \
    --optimizer.max_skew_triangular=16 \
    --optimizer.precond_lr=0.1 \
    --optimizer.precond_init_scale=1.0 \
    --optimizer.preconditioner_dtype=float32 \
    --model.block_size=1024 \
    --model.num_layers=24 \
    --model.num_heads=16 \
    --model.num_embeds=1024
"