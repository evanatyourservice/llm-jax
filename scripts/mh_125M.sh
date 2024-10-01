#!/bin/bash
# multihost TPU run script for 125M model
# usage: bash scripts/mh_125M.sh \
# --wandb_api_key <wandb_api_key> \
# --huggingface_token <huggingface_token> \
# --tpu_zone <tpu_zone> \
# --tpu_name <tpu_name> \
# --gcp_project <gcp_project> \
# --out_dir <out_dir> \
# [--experiment <experiment_name>]

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --wandb_api_key)
      WANDB_API_KEY="$2"
      shift 2
      ;;
    --huggingface_token)
      HF_TOKEN="$2"
      shift 2
      ;;
    --tpu_zone)
      TPU_ZONE="$2"
      shift 2
      ;;
    --tpu_name)
      TPU_NAME="$2"
      shift 2
      ;;
    --gcp_project)
      GCP_PROJECT="$2"
      shift 2
      ;;
    --out_dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --experiment)
      EXPERIMENT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if required arguments are provided
if [[ -z $WANDB_API_KEY || -z $HF_TOKEN || -z $TPU_ZONE || -z $TPU_NAME || -z $GCP_PROJECT ]]; then
  echo "Error: Missing required arguments"
  exit 1
fi

# Set default experiment name if not provided
if [[ -z $EXPERIMENT ]]; then
  EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
fi

echo $EXPERIMENT

gcloud compute tpus tpu-vm ssh --zone $TPU_ZONE $TPU_NAME --project $GCP_PROJECT \
--worker=all --command "bash -c \"
export WANDB_API_KEY=$WANDB_API_KEY 
export HF_TOKEN=$HF_TOKEN
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
cd llm-jax
nohup python3 main_multihost.py \
    --experiment_name=$EXPERIMENT \
    --out_dir=$OUT_DIR \
    --attempt_to_load_checkpoint \
    --hellaswag_eval_interval=1000 \
    --checkpoint_interval=1000 \
    --train_steps=150000 \
    --batch_size=512 \
    --gradient_accumulation_steps=1 \
    --compute_dtype=bfloat16 \
    --params_dtype=float32 \
    --profile \
    --model.block_size=2048 \
    --model.sliding_window_size=1024 \
    --model.scan_layers \
    --model.scan_unroll=3 \
    --optimizer.type=psgd \
    --optimizer.learning_rate=0.003 \
    --optimizer.warmup_steps=1000 \
    --optimizer.weight_decay=0.01 \
    --optimizer.grad_clip=1.0 \
    --optimizer.preconditioner_update_probability=0.04 \
    --optimizer.max_size_triangular=8192 \
    --optimizer.max_skew_triangular=10 \
    --optimizer.preconditioner_dtype=float32 \
    > nohup.out 2>&1 & 
PID=\\\$!
echo 'Background process started with PID '\\\$PID
disown \\\$PID
exit
\""