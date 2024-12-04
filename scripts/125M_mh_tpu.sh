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
    --train_steps=100000 \
    --batch_size=512 \
    --gradient_accumulation_steps=1 \
    --compute_dtype=bfloat16 \
    --params_dtype=float32 \
    --profile \
    --model.scan_layers \
    --model.remat \
    --model.no_remat_everything \
    --optimizer.type=kron \
    --optimizer.learning_rate=0.001 \
    --optimizer.warmup_steps=1000 \
    --optimizer.b1=0.95 \
    --optimizer.weight_decay=0.1 \
    --optimizer.preconditioner_update_probability=0.03 \
    --optimizer.preconditioner_dtype=float32 \
    > nohup.out 2>&1 & 
PID=\\\$!
echo 'Background process started with PID '\\\$PID
disown \\\$PID
exit
\""