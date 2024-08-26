#!/bin/bash
# usage: bash scripts/gpt_mh.sh <wandb_api_key>

WANDB_API_KEY=$1

gcloud compute tpus tpu-vm ssh --zone "us-central2-b" "LLaMA" --project "my-phd-research-o" \
--worker=all --command="
EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

export WANDB_API_KEY=$WANDB_API_KEY
export GPT_CONFIG=config/gpt2.yaml

cd gpt-jax

python3 train.py \
    --out_dir=gs://uscentral1stuff/gpt_models/gpt_small/$EXPERIMENT \
    --train_pattern=gs://uscentral1stuff/openwebtext/train_??.tfrecord \
    --val_pattern=gs://uscentral1stuff/openwebtext/val_??.tfrecord \
    --optimizer.type=affine \
    --optimizer.weight_decay=0.1 \
    --optimizer.grad_clip=1.0 \
    --optimizer.preconditioner_update_probability=0.5 \
    --optimizer.update_global_norm_clip=10000.0 \
    --optimizer.update_elementwise_clip \
    --optimizer.max_size_triangular=1000000000 \
    --optimizer.max_skew_triangular=1 \
    --optimizer.precond_lr=0.1 \
    --optimizer.precond_init_scale=1.0
"