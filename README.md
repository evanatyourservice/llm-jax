# llm-jax

Started with [this repo](https://github.com/jenkspt/gpt-jax). Also pulled some tools from [big_vision](https://github.com/google-research/big_vision).

TODO:
- any huggingface model
- scan huggingface model layers
- any huggingface dataset (stream)


## Install

Clone gpt-jax
```shell
git clone https://github.com/evanatyourservice/gpt-jax.git
```

Install python dependencies TPU
```shell
cd gpt-jax && pip install -U pip && pip install -r requirements.txt && pip install --force-reinstall --upgrade --no-cache-dir 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && pip install 'numpy<2'
```

Install python dependencies GPU
```shell
cd gpt-jax && pip install -U pip && pip install -r requirements.txt && pip install --force-reinstall --upgrade --no-cache-dir 'jax[cuda12]' && pip install 'numpy<2'
```


## Data

Prepare data tokenizes and saves openwebtext to tfrecords.
```shell
python data/openwebtext/prepare.py
```

This will generate the following files:  
`train_0.tfrecord`, `train_1.tfrecord` ... `train_{num_shards}`  
`val_0.tfrecord`, `val_1.tfrecord` ... `val_{num_shards}`

If you're training on a TPU, you should copy these files to a GCS bucket.


## Run

The base settings are in `config/gpt2.yaml`. This is loaded in scripts using `export GPT_CONFIG=config/gpt2.yaml`. 
You can override with your own settings by either loading your own config in a script, or using flags 
like in `scripts/gpt_psgd.sh`.

To run on multi-host TPU, install requirements on all hosts
```shell
gcloud compute tpus tpu-vm ssh gpt-jax --zone=us-central2-a --worker=all --command="cd gpt-jax && pip install -r requirements.txt"
```

Then run a script on all hosts

```shell
gcloud compute tpus tpu-vm ssh gpt-jax --zone=us-central2-a --worker=all --command="cd gpt-jax && bash scripts/gpt_psgd.sh"
```
