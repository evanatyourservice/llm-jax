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

export LLM_CONFIG=config/gpt2.yaml

cd llm-jax

python3 main_multihost.py \
    --out_dir=gs://uscentral2stuff/llm-jax/test_mh \
    --attempt_to_load_checkpoint \
    --hellaswag_eval_interval=500 \
    --checkpoint_interval=20 \
    --train_steps=100 \
    --batch_size=64 \
    --optimizer.gradient_accumulation_steps=1 \
    --compute_dtype=bfloat16 \
    --params_dtype=float32 \
    --optimizer.type=psgd_affine \
    --optimizer.learning_rate=0.001 \
    --optimizer.warmup_steps=0 \
    --optimizer.nesterov \
    --optimizer.weight_decay=0.01 \
    --optimizer.grad_clip=1.0 \
    --optimizer.max_size_triangular=16384 \
    --optimizer.max_skew_triangular=0 \
    --optimizer.precond_lr=0.1 \
    --optimizer.precond_init_scale=1.0 \
    --optimizer.preconditioner_dtype=float32 \
    --model.block_size=64 \
    --model.num_layers=2 \
    --model.num_heads=4 \
    --model.num_embeds=512
"