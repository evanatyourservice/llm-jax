import json
from typing import Optional
import numpy as np
from tqdm import tqdm

import jax
import tensorflow as tf
import tiktoken

from utils import make_fsarray_from_local_slice, prefetch_iterator, threadstart_iterator


OPTIONS = tf.data.Options()
OPTIONS.deterministic = False
OPTIONS.autotune.enabled = True


def get_dataset(
    pattern: str,
    flat_devices,
    batch_size: int,
    block_size: int = 1024,
    interleave_cycle_length: int = 1,
    shuffle_buffer_size: Optional[int] = None,
    tf_prefetch: int = 5,
    device_prefetch: int = 0,
) -> tf.data.Dataset.as_numpy_iterator:
    file_ds = tf.data.Dataset.list_files(pattern, shuffle=True)
    file_ds = file_ds.shard(jax.process_count(), jax.process_index())
    file_ds = file_ds.repeat()

    if interleave_cycle_length > 1:
        ds = file_ds.interleave(
            lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=tf.data.AUTOTUNE),
            cycle_length=interleave_cycle_length,
            block_length=1,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = tf.data.TFRecordDataset(file_ds, num_parallel_reads=tf.data.AUTOTUNE)

    # each element of the dataset is a tokenized string
    feature_description = {
        "ids": tf.io.FixedLenFeature([], tf.string, default_value="")
    }

    def parse_example(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        return tf.io.decode_raw(example["ids"], tf.uint16)

    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    # here we shuffle each group of tokens and then unbatch into a single
    # contiguous sequence of ids, we then chunk the sequence into blocks
    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size)

    ds = ds.unbatch().batch(block_size + 1, drop_remainder=True)

    # blocks are then shuffled and batched
    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size)

    ds = ds.batch(batch_size // jax.process_count(), drop_remainder=True)
    ds = ds.with_options(OPTIONS)
    ds = ds.prefetch(tf_prefetch)
    ds = ds.as_numpy_iterator()
    ds = iter(ds)
    ds = threadstart_iterator(ds)
    ds = (
        jax.tree.map(lambda x: make_fsarray_from_local_slice(x, flat_devices), elem)
        for elem in ds
    )
    if device_prefetch > 0:
        ds = prefetch_iterator(ds, device_prefetch)
    return ds


def prepare_hellaswag(
    batch_size: int,
    block_size: int,
    flat_devices,
    shuffle_buffer_size: Optional[int] = 1250,
    tf_prefetch: int = 2,
    device_prefetch: int = 0,
):
    """Read file and tokenize the hellaswag dataset."""
    print("preparing hellaswag")

    enc = tiktoken.get_encoding("gpt2")

    all_data = []
    all_labels = []
    all_lengths = []
    with open("data/hellaswag_val.jsonl", "r") as f:
        # iterate over lines and tokenize
        for line in tqdm(f, total=10042):
            item = json.loads(line)
            context = item["ctx"]
            endings = item["endings"]
            correct_end = item["label"]
            to_concat = []
            lens = []
            for ending in endings:
                input_text = context + " " + ending
                input_ids = enc.encode_ordinary(input_text)
                # +1 for [1:] input [:-1] target shift
                if len(input_ids) > block_size + 1:
                    continue
                lens.append(len(input_ids))
                input_ids = np.pad(input_ids, (0, block_size + 1 - len(input_ids)))
                to_concat.append(input_ids)
            all_data.append(np.array(to_concat))  # (4, block_size)
            all_labels.append(correct_end)  # scalar
            all_lengths.append(np.array(lens))  # (4,)

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    all_lengths = np.array(all_lengths)

    ds = tf.data.Dataset.from_tensor_slices((all_data, all_labels, all_lengths))
    ds = ds.shard(jax.process_count(), jax.process_index())
    ds = ds.repeat()
    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.batch(batch_size // jax.process_count(), drop_remainder=True)
    ds = ds.with_options(OPTIONS)
    ds = ds.prefetch(tf_prefetch)
    ds = ds.as_numpy_iterator()
    ds = iter(ds)
    ds = threadstart_iterator(ds)
    ds = (
        jax.tree.map(lambda x: make_fsarray_from_local_slice(x, flat_devices), elem)
        for elem in ds
    )
    if device_prefetch > 0:
        ds = prefetch_iterator(ds, device_prefetch)
    return ds
