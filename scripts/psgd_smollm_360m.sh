#!/bin/bash
# usage: bash scripts/multihost.sh <wandb_api_key> <huggingface_token>

WANDB_API_KEY=$1
HF_TOKEN=$2

gcloud compute tpus tpu-vm ssh --zone "us-central2-b" "LLaMA" --project "my-phd-research-o" \
--worker=all --command="
EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

export WANDB_API_KEY=$WANDB_API_KEY
export HF_TOKEN=$HF_TOKEN

export LLM_CONFIG=config/gemma2.yaml

cd llm-jax

python3 train.py \
    --out_dir=gs://uscentral2stuff/llm-jax/$EXPERIMENT \
    --attempt_to_load_checkpoint \
    --hellaswag_eval_interval=500 \
    --checkpoint_interval=1000 \
    --train_steps=1000000 \
    --batch_size=512 \
    --optimizer.gradient_accumulation_steps=1 \
    --compute_dtype=bfloat16 \
    --params_dtype=bfloat16 \
    --optimizer.type=psgd_affine \
    --optimizer.learning_rate=0.003 \
    --optimizer.warmup_steps=0 \
    --optimizer.nesterov \
    --optimizer.weight_decay=0.01 \
    --optimizer.grad_clip=1.0 \
    --optimizer.max_size_triangular=16384 \
    --optimizer.max_skew_triangular=16 \
    --optimizer.precond_lr=0.1 \
    --optimizer.precond_init_scale=0.01 \
    --optimizer.preconditioner_dtype=bfloat16 \
    --model.model_type=smollm_360m \
    --model.sliding_window_size=512 \
    --model.block_size=1024
"