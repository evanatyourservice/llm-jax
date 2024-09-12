#!/bin/bash
# usage: bash scripts/multihost.sh <wandb_api_key> <huggingface_token>

WANDB_API_KEY=$1
HF_TOKEN=$2

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

gcloud compute tpus tpu-vm ssh --zone "us-central2-b" "LLaMA" --project "my-phd-research-o" \
--worker=all --command "bash -c \"
export WANDB_API_KEY=$WANDB_API_KEY
export HF_TOKEN=$HF_TOKEN
export LLM_CONFIG=config/gpt2.yaml
cd llm-jax
nohup python3 main_multihost.py \
    --experiment_name=$EXPERIMENT \
    --out_dir=gs://uscentral2stuff/llm-jax \
    --attempt_to_load_checkpoint \
    --hellaswag_eval_interval=1000 \
    --checkpoint_interval=1000 \
    --train_steps=191000 \
    --batch_size=512 \
    --optimizer.gradient_accumulation_steps=1 \
    --compute_dtype=bfloat16 \
    --params_dtype=float32 \
    --model.num_layers=24 \
    --model.num_heads=16 \
    --model.num_embeds=1536 \
    --optimizer.type=affine \
    --optimizer.learning_rate=0.003 \
    --optimizer.warmup_steps=500 \
    --optimizer.weight_decay=0.01 \
    --optimizer.grad_clip=1.0 \
    --optimizer.max_size_triangular=10000 \
    --optimizer.max_skew_triangular=10 \
    --optimizer.precond_lr=0.1 \
    --optimizer.precond_init_scale=1.0 \
    --optimizer.preconditioner_dtype=float32 > nohup.out 2>&1 &
echo 'Background process started'
exit
\""