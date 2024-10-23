import os
import json
from typing import Optional
import numpy as np
from tqdm import tqdm

import jax
import tensorflow as tf
import tensorflow_io as tfio
import datasets
from datasets import load_dataset, Dataset
import datasets.config
from transformers import AutoTokenizer

from utils import (
    make_fsarray_from_local_slice,
    prefetch_iterator,
    threadstart_iterator,
    write_note,
)


os.environ["TOKENIZERS_PARALLELISM"] = "true"
datasets.config.STREAMING_READ_MAX_RETRIES = 17280  # 17280 * 5 = 1 day
datasets.config.STREAMING_READ_RETRY_INTERVAL = 5

OPTIONS = tf.data.Options()
OPTIONS.deterministic = False
OPTIONS.threading.private_threadpool_size = 48
OPTIONS.threading.max_intra_op_parallelism = 1
# Stop a whole bunch of magic stuff that eats up all RAM:
OPTIONS.experimental_optimization.inject_prefetch = False


TOKENIZER = "mistralai/Mistral-7B-v0.3"


def prepare_hellaswag(
    batch_size: int,
    block_size: int,
    flat_devices,
    tf_prefetch: int = 2,
    device_prefetch: int = 0,
):
    """Read file and tokenize the hellaswag dataset."""
    write_note("preparing hellaswag")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER, trust_remote_code=True, use_fast=True
    )

    all_data = []
    all_beginning_lengths = []
    all_seq_lengths = []
    all_labels = []
    with open("data/hellaswag_val.jsonl", "r") as f:
        # iterate over lines and tokenize
        for line in tqdm(f, total=10042):
            item = json.loads(line)

            context = item["ctx"]
            endings = item["endings"]
            correct_end = item["label"]

            beginning_length = len(tokenizer(context)["input_ids"])

            data_to_concat = []
            beginning_lengths_to_concat = []
            seq_lengths_to_concat = []
            for ending in endings:
                output = tokenizer(context + " " + ending)["input_ids"]
                output_len = len(output)

                # pad to block_size
                if output_len < block_size:
                    output = output + [tokenizer.eos_token_id] * (
                        block_size - output_len
                    )
                # max length is block_size
                output = output[:block_size]

                data_to_concat.append(output)
                beginning_lengths_to_concat.append(beginning_length)
                seq_lengths_to_concat.append(output_len)

            all_data.append(np.array(data_to_concat, dtype=np.uint16))
            all_beginning_lengths.append(
                np.array(beginning_lengths_to_concat, dtype=np.int32)
            )
            all_seq_lengths.append(np.array(seq_lengths_to_concat, dtype=np.int32))
            all_labels.append(int(correct_end))

    all_data = np.array(all_data, dtype=np.uint16)
    all_beginning_lengths = np.array(all_beginning_lengths, dtype=np.int32)
    all_seq_lengths = np.array(all_seq_lengths, dtype=np.int32)
    all_labels = np.array(all_labels, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices(
        (all_data, all_beginning_lengths, all_seq_lengths, all_labels)
    )
    ds = ds.shard(jax.process_count(), jax.process_index())
    ds = ds.repeat()

    ds = ds.batch(
        batch_size // jax.process_count(),
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.with_options(OPTIONS)
    ds = ds.prefetch(tf_prefetch)
    ds = ds.as_numpy_iterator()
    ds = iter(ds)
    # ds = threadstart_iterator(ds)
    ds = (
        jax.tree.map(lambda x: make_fsarray_from_local_slice(x, flat_devices), elem)
        for elem in ds
    )
    if device_prefetch > 0:
        ds = prefetch_iterator(ds, device_prefetch)
    return ds


def fineweb_edu_dataset(
    data_dir: str,
    batch_size: int,
    block_size: int,
    flat_devices,
    fineweb_edu_name: Optional[str] = None,
    tf_prefetch: int = 5,
    device_prefetch: int = 0,
):
    """Load the fineweb-edu dataset."""
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER, trust_remote_code=True, use_fast=True
    )

    files = tf.io.gfile.glob(f"{data_dir}/fineweb-edu-dedup/*.parquet")
    np.random.shuffle(files)

    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.shuffle(128)

    ds = ds.interleave(
        map_func=lambda f: tfio.IOTensor.from_parquet(f, columns=["text", "id", "metadata"]),
        cycle_length=16,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    def tokenize(example):
        # mistral tokenizer adds bos token to beginning
        tokenized = tokenizer(example["text"])["input_ids"]
        # cap tokenized lengths to 10 * block_size to prevent too much
        # similarity between blocks in a batch or group of batches
        tokenized = [t[: 10 * block_size] for t in tokenized]
        return tokenized

    ds = ds.map(tokenize)

    ds = ds.shuffle(128)  # shuffle dataset examples
    ds = ds.unbatch()
    ds = ds.batch(block_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(20 * 1024)  # shuffle blocks
    ds = ds.batch(
        batch_size // jax.process_count(),
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.with_options(OPTIONS)
    ds = ds.prefetch(tf_prefetch)
    ds = ds.as_numpy_iterator()
    ds = iter(ds)
    # ds = threadstart_iterator(ds)
    ds = (
        jax.tree.map(lambda x: make_fsarray_from_local_slice(x, flat_devices), elem)
        for elem in ds
    )
    if device_prefetch > 0:
        ds = prefetch_iterator(ds, device_prefetch)
    return ds
