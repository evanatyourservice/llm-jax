#!/bin/bash
# script for multihost tpu (set for v4-16, increase settings for larger vms)
# usage: bash scripts/125M_mh_tpu.sh <wandb_api_key> <huggingface_token>

WANDB_API_KEY=$1
HF_TOKEN=$2

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

gcloud compute tpus tpu-vm ssh --zone "us-central2-b" "tpu_vm_name" --project "project_name" --worker=all --command "bash -c \"
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
    --train_steps=250000 \
    --batch_size=512 \
    --gradient_accumulation_steps=1 \
    --compute_dtype=bfloat16 \
    --profile \
    --model.num_layers=32 \
    --model.num_heads=15 \
    --model.num_kv_heads=5 \
    --model.head_dim=64 \
    --model.num_embeds=960 \
    --model.hidden_dim=2560 \
    --model.scan_layers \
    --model.remat \
    --model.no_remat_everything \
    --optimizer.type=kron \
    --optimizer.learning_rate=0.001 \
    --optimizer.warmup_steps=1000 \
    --optimizer.weight_decay=0.1 \
    --optimizer.grad_clip=1.0 \
    --optimizer.preconditioner_update_probability=0.05 \
    > nohup.out 2>&1 & 
PID=\\\$!
echo 'Background process started with PID '\\\$PID
disown \\\$PID
exit
\""