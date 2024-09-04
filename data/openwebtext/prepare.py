# Copied from https://github.com/karpathy/nanoGPT/blob/177d5f7dc5f44d6f373cd7767c2a9259d740436e/data/openwebtext/prepare.py
# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from multiprocessing import cpu_count
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, DatasetDict
import tensorflow as tf


save_dir = "/dev/shm/openwebtext"
os.makedirs(save_dir, exist_ok=True)


# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = cpu_count() // 2
# number of files for each split
num_shards = {"train": 32, "val": 8}

# takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
dataset = load_dataset("openwebtext", cache_dir="/dev/shm/hf_cache")

# owt by default only contains the 'train' split, so create a test split
split_dataset = dataset["train"].train_test_split(
    test_size=0.0005, seed=2357, shuffle=True
)
split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val

shard_dataset = DatasetDict()
for split, dset in split_dataset.items():
    for i in range(num_shards[split]):
        shard_dataset[f"{split}_{i:02}"] = dset.shard(num_shards[split], i)

# this results in:
# >>> shard_dataset
# DatasetDict({
#     train_0: Dataset({
#         features: ['text'],
#         num_rows: 8009762 / num_shards
#     })
#     train_1: Dataset({
#         features: ['text'],
#         num_rows: 8009762 / num_shards
#     })
#     ...
#     val_0: Dataset({
#         features: ['text'],
#         num_rows: 4007 / num_shards
#     })
#     ...
# })

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
enc = tiktoken.get_encoding("gpt2")


def process(example):
    ids = enc.encode_ordinary(
        example["text"]
    )  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {"ids": ids}
    return out


# tokenize the dataset
tokenized = shard_dataset.map(
    process,
    remove_columns=["text"],
    desc="tokenizing the splits",
    num_proc=num_proc,
)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    filename = f"{split}.tfrecord"
    full_path = os.path.join(save_dir, filename)

    print(f"writing {full_path}...")
    with tf.io.TFRecordWriter(full_path) as writer:
        for example in tqdm(dset):
            feature = np.asarray(example["ids"], dtype=np.uint16).tobytes()
            example_proto = tf.train.Example(
                features=tf.train.Features(feature={"ids": _bytes_feature(feature)})
            )
            writer.write(example_proto.SerializeToString())


# train.bin is ~17GB, val.bin ~8.5MB
# train has ~9B tokens (9,035,582,198)
# val has ~4M tokens (4,434,897)

# to read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')
