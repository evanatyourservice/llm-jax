# llm-jax

Pretrain a mistral-style model with fineweb-edu.

Started with [this repo, credit to @jenkspt](https://github.com/jenkspt/gpt-jax). Also pulled some tools 
from [big_vision](https://github.com/google-research/big_vision) to add simple FSDP rules.

Has some different optimizers, adamw, schedule-free, PSGD, shampoo, and CASPR. Shampoo and CASPR probably 
not good for large nets, compile time problems.

Only set up for pretraining for now, working on inference and conversion to pytorch and huggingface hub.

## Install

Clone llm-jax
```shell
git clone https://github.com/evanatyourservice/llm-jax.git
```

Install python dependencies TPU
```shell
cd llm-jax && pip install -U pip && pip install -r requirements.txt && pip install --force-reinstall --upgrade --no-cache-dir 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && pip install 'numpy<2'
```

Install python dependencies GPU
```shell
cd llm-jax && pip install -U pip && pip install -r requirements.txt && pip install --force-reinstall --upgrade --no-cache-dir 'jax[cuda12]' && pip install 'numpy<2'
```


## Run

See examples in /scripts like `scripts/mh_125M.sh`.

create TPU using queued-resources
```shell
gcloud compute tpus queued-resources create node-1 --node-id node-1 --project distributedmuzerojax --zone us-central2-b --accelerator-type v4-64 --runtime-version tpu-ubuntu2204-base --scopes https://www.googleapis.com/auth/cloud-platform
```