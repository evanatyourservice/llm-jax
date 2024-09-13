# llm-jax

Started with [this repo, credit to @jenkspt](https://github.com/jenkspt/gpt-jax). 
Also pulled some tools from [big_vision](https://github.com/google-research/big_vision) to add simple FSDP.

TODO:
- scan model layers for psgd affine
- checkpointing huggingface dataset
- add sharding and more models from EasyDeL


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

The base settings are in `config/mistral.yaml`. This is loaded in scripts using `export LLM_CONFIG=config/mistral.yaml`. 
You can override with your own settings by either loading your own config in a script, or using flags 
like in `scripts/psgd_gpt_mh.sh`.


create TPU with queued resources
```shell
gcloud compute tpus queued-resources create node-3 --node-id node-3 --project distributedmuzerojax --zone us-central2-b --accelerator-type v4-16 --runtime-version tpu-ubuntu2204-base --scopes https://www.googleapis.com/auth/cloud-platform
```