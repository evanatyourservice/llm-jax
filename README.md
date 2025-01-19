# llm-jax

Pretrain a SmolLM-style language model on the fineweb-edu dataset. A 350M param model can reach 51% hellaswag in only 250B tokens by using psgd kron optimizer and architecture improvements.

Has various optimizers: PSGD Kron, adamw, shampoo, CASPR, and schedule-free. Any optimizer can be wrapped in 
schedule-free, see configs.py for more details.

Only set up for pretraining right now, working on inference, conversion to pytorch, and uploading to huggingface hub.

Saves checkpoints to out_dir, set same experiment name to resume.

Set --profile to profile training to tensorboard, tensorboard dir is <out_dir>/profile.

See configs.py for other settings and all hyperparameters.

This repo is made possible by [Google's TRC program](https://sites.research.google/trc/about/).

Started with [this repo, credit to @jenkspt](https://github.com/jenkspt/gpt-jax). Also pulled some tools 
from [big_vision](https://github.com/google-research/big_vision) to add FSDP sharding.

Shoutout to @Grad62304977 for sharing model tips to improve training stability.


## Install

Clone llm-jax
```shell
git clone https://github.com/evanatyourservice/llm-jax.git
```

Install python dependencies TPU
```shell
cd llm-jax && pip install -U pip && pip install -U -r requirements.txt && pip install --force-reinstall --upgrade --no-cache-dir 'jax[tpu]' 'jaxlib' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && pip install 'numpy<2'
```

Install python dependencies GPU
```shell
cd llm-jax && pip install -U pip && pip install -r requirements.txt && pip install --force-reinstall --upgrade --no-cache-dir 'jax[cuda12]' && pip install 'numpy<2'
```


## Run

See examples in /scripts like `scripts/125M_mh_tpu.sh`.

create TPU using queued-resources
```shell
gcloud compute tpus queued-resources create node-4 --node-id node-4 --project distributedmuzerojax --zone us-central2-b --accelerator-type v4-16 --runtime-version tpu-ubuntu2204-base --scopes https://www.googleapis.com/auth/cloud-platform
```
