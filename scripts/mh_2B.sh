#!/bin/bash
# usage: bash scripts/multihost.sh <wandb_api_key> <huggingface_token>

WANDB_API_KEY=$1
HF_TOKEN=$2

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

# gcloud compute tpus tpu-vm ssh --zone "us-central2-b" "node-1" --project "distributedmuzerojax"
# gcloud compute tpus tpu-vm ssh --zone "us-central2-b" "LLaMA" --project "my-phd-research-o"
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
    --train_steps=100000 \
    --batch_size=128 \
    --gradient_accumulation_steps=2 \
    --compute_dtype=bfloat16 \
    --params_dtype=bfloat16 \
    --profile \
    --model.block_size=4096 \
    --model.sliding_window_size=2048 \
    --model.num_layers=32 \
    --model.num_heads=32 \
    --model.num_kv_heads=8 \
    --model.head_dim=96 \
    --model.num_embeds=2048 \
    --model.hidden_dim=7168 \
    --model.scan_layers \
    --model.scan_unroll=1 \
    --optimizer.type=psgd \
    --optimizer.learning_rate=0.001 \
    --optimizer.warmup_steps=1000 \
    --optimizer.weight_decay=0.01 \
    --optimizer.grad_clip=1.0 \
    --optimizer.preconditioner_update_probability=0.03 \
    --optimizer.max_size_triangular=8192 \
    --optimizer.max_skew_triangular=10 \
    --optimizer.precond_lr=0.3 \
    --optimizer.precond_init_scale=0.1 \
    --optimizer.preconditioner_dtype=bfloat16 \
    > nohup.out 2>&1 & 
PID=\\\$!
echo 'Background process started with PID '\\\$PID
disown \\\$PID
exit
\""