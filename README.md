# llm-jax

Started with [this repo, credit to @jenkspt](https://github.com/jenkspt/gpt-jax). 
Also pulled some tools from [big_vision](https://github.com/google-research/big_vision) to add simple FSDP.
Model is from [EasyDeL](https://github.com/erfanzar/EasyDeL).

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

The base settings are in `config/llama3.yaml`. This is loaded in scripts using `export LLM_CONFIG=config/llama3.yaml`. 
You can override with your own settings by either loading your own config in a script, or using flags 
like in `scripts/psgd.sh`.