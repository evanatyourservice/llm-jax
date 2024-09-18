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
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
cd llm-jax
nohup python3 main_multihost.py \
    --experiment_name=$EXPERIMENT \
    --out_dir=gs://optimizertesting/llm-jax \
    --attempt_to_load_checkpoint \
    --hellaswag_eval_interval=1000 \
    --checkpoint_interval=1000 \
    --train_steps=150000 \
    --batch_size=128 \
    --optimizer.gradient_accumulation_steps=4 \
    --compute_dtype=bfloat16 \
    --params_dtype=bfloat16 \
    --remat \
    --model.block_size=2048 \
    --model.sliding_window_size=1024 \
    --model.num_layers=32 \
    --model.num_heads=32 \
    --model.num_kv_heads=8 \
    --model.head_dim=64 \
    --model.num_embeds=2048 \
    --model.hidden_dim=8192 \
    --optimizer.type=psgd_affine \
    --optimizer.learning_rate=0.001 \
    --optimizer.warmup_steps=1000 \
    --optimizer.weight_decay=0.1 \
    --optimizer.grad_clip=1.0 \
    --optimizer.preconditioner_update_probability=0.04 \
    --optimizer.max_size_triangular=10000 \
    --optimizer.max_skew_triangular=7 \
    --optimizer.precond_lr=0.3 \
    --optimizer.precond_init_scale=0.01 \
    --optimizer.preconditioner_dtype=bfloat16 \
    --optimizer.best_effort_vmap \
    > nohup.out 2>&1 & 
PID=\\\$!
echo 'Background process started with PID '\\\$PID
disown \\\$PID
exit
\""