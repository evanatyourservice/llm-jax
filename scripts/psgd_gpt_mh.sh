#!/bin/bash
# usage: bash scripts/multihost.sh <wandb_api_key> <huggingface_token>

WANDB_API_KEY=$1
HF_TOKEN=$2

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

gcloud compute tpus tpu-vm ssh --zone "us-central2-b" "node-1" --project "distributedmuzerojax" \
--worker=all --command "bash -c \"
export WANDB_API_KEY=$WANDB_API_KEY 
export HF_TOKEN=$HF_TOKEN
export LLM_CONFIG=config/mistral.yaml
cd llm-jax
nohup python3 main_multihost.py \
    --experiment_name=$EXPERIMENT \
    --out_dir=gs://optimizertesting/llm-jax \
    --attempt_to_load_checkpoint \
    --hellaswag_eval_interval=1000 \
    --checkpoint_interval=1000 \
    --train_steps=175000 \
    --batch_size=128 \
    --optimizer.gradient_accumulation_steps=4 \
    --compute_dtype=floa \
    --params_dtype=float32 \
    --model.block_size=2048 \
    --model.sliding_window_size=1024 \
    --optimizer.type=psgd_affine \
    --optimizer.learning_rate=0.005 \
    --optimizer.warmup_steps=0 \
    --optimizer.weight_decay=0.01 \
    --optimizer.grad_clip=1.0 \
    --optimizer.preconditioner_update_probability=0.05 \
    --optimizer.max_size_triangular=10000 \
    --optimizer.max_skew_triangular=10 \
    --optimizer.precond_lr=0.5 \
    --optimizer.precond_init_scale=0.000000000001 \
    --optimizer.preconditioner_dtype=float32 > nohup.out 2>&1 & 
PID=\\\$!
echo 'Background process started with PID '\\\$PID
disown \\\$PID
exit
\""