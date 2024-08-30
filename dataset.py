import json
import numpy as np
from tqdm import tqdm

import jax
import tensorflow as tf
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer

from utils import (
    make_fsarray_from_local_slice,
    prefetch_iterator,
    threadstart_iterator,
    write_note,
)

OPTIONS = tf.data.Options()
OPTIONS.deterministic = False
OPTIONS.threading.private_threadpool_size = 48
OPTIONS.threading.max_intra_op_parallelism = 1
# Stop a whole bunch of magic stuff that eats up all RAM:
OPTIONS.experimental_optimization.inject_prefetch = False


def get_dataset(
    pattern: str,
    flat_devices,
    batch_size: int,
    block_size: int = 1024,
    interleave_cycle_length: int = 1,
    tf_prefetch: int = 5,
    device_prefetch: int = 0,
) -> tf.data.Dataset.as_numpy_iterator:
    seq_len = block_size + 1

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

    ds = ds.unbatch().batch(
        seq_len, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE
    )

    # blocks are then shuffled and batched
    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size)

    ds = ds.batch(
        batch_size // jax.process_count(),
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
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
    tokenizer_name: str,
    batch_size: int,
    block_size: int,
    flat_devices,
    tf_prefetch: int = 2,
    device_prefetch: int = 0,
):
    """Read file and tokenize the hellaswag dataset."""
    write_note("preparing hellaswag")

    seq_len = block_size + 1

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
                input_ids = tokenizer(
                    input_text,
                    add_special_tokens=False,
                    max_length=seq_len,
                    padding="max_length",
                    truncation=True,
                )["input_ids"]
                lens.append(len(input_ids))
                to_concat.append(input_ids)
            all_data.append(np.array(to_concat))  # (4, seq_len)
            all_labels.append(int(correct_end))  # Convert to integer
            all_lengths.append(np.array(lens))  # (4,)

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    all_lengths = np.array(all_lengths)

    ds = tf.data.Dataset.from_tensor_slices((all_data, all_labels, all_lengths))
    ds = ds.shard(jax.process_count(), jax.process_index())
    ds = ds.batch(
        batch_size // jax.process_count(),
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds_length = len(ds)

    ds = ds.repeat()
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
    return ds, ds_length


def fineweb_edu_dataset(
    tokenizer_name: str,
    batch_size: int,
    block_size: int,
    flat_devices,
    tf_prefetch: int = 5,
    device_prefetch: int = 0,
    streaming: bool = False,
    start_index: int = 0,
    seed: int = 42,
    num_shards_per_process: int = 1,
):
    seq_len = block_size + 1

    # awful way to shard but whatever
    idx = jax.process_index()
    # download n shards for each process
    names = _fw_shard_names[
        idx * num_shards_per_process : idx * num_shards_per_process
        + num_shards_per_process
    ]
    # use /dev/shm if on a TPU vm for more space
    platform = jax.devices()[0].platform
    if platform == "tpu":
        cache_dir = "/dev/shm/huggingface_cache"
    else:
        cache_dir = None

    if streaming:
        write_note("streaming fineweb-edu")

        def gen():
            hf_ds: Dataset = concatenate_datasets(
                [
                    load_dataset(
                        "HuggingFaceFW/fineweb-edu",
                        name=name,
                        split="train",
                        cache_dir=cache_dir,
                        streaming=True,
                    )
                    for name in names
                ]
            )
            hf_ds = (
                hf_ds.shuffle(seed=seed).shuffle(seed=seed + 1).shuffle(seed=seed + 2)
            )

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True, use_fast=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            def tokenize(data_chunk):
                return tokenizer(
                    data_chunk["text"],
                    add_special_tokens=False,
                    max_length=seq_len,
                    padding="max_length",
                    truncation=True,
                )

            hf_ds = hf_ds.map(tokenize)
            hf_ds = hf_ds.with_format("tensorflow")

            for example in hf_ds:
                yield example["input_ids"]

        ds = tf.data.Dataset.from_generator(
            gen, output_signature=tf.TensorSpec(shape=(seq_len,), dtype=tf.int32)
        )
        ds = ds.batch(
            batch_size // jax.process_count(),
            drop_remainder=True,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
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

    else:
        write_note("loading fineweb-edu")

        hf_ds: Dataset = concatenate_datasets(
            [
                load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    name=name,
                    split="train",
                    cache_dir=cache_dir,
                )
                for name in names
            ]
        )

        hf_ds = hf_ds.shuffle(seed=seed).shuffle(seed=seed + 1).shuffle(seed=seed + 2)

        ds_len = len(hf_ds)
        hf_ds = hf_ds.skip(start_index % ds_len)

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True, use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def tokenize(data_chunk):
            return tokenizer(
                data_chunk["text"],
                add_special_tokens=False,
                max_length=seq_len,
                padding="max_length",
                truncation=True,
            )

        hf_ds = hf_ds.map(tokenize, num_proc=16)
        hf_ds = hf_ds.shuffle(seed=seed).shuffle(seed=seed + 1).shuffle(seed=seed + 2)

        ds = hf_ds.to_tf_dataset(prefetch=False, label_cols=["input_ids"])
        ds = ds.batch(
            batch_size // jax.process_count(),
            drop_remainder=True,
            num_parallel_calls=16,
        )
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


# fineweb-edu has 96 shards
_fw_shard_names = [
    "CC-MAIN-2024-10",
    "CC-MAIN-2023-50",
    "CC-MAIN-2023-40",
    "CC-MAIN-2023-23",
    "CC-MAIN-2023-14",
    "CC-MAIN-2023-06",
    "CC-MAIN-2022-49",
    "CC-MAIN-2022-40",
    "CC-MAIN-2022-33",
    "CC-MAIN-2022-27",
    "CC-MAIN-2022-21",
    "CC-MAIN-2022-05",
    "CC-MAIN-2021-49",
    "CC-MAIN-2021-43",
    "CC-MAIN-2021-39",
    "CC-MAIN-2021-31",
    "CC-MAIN-2021-25",
    "CC-MAIN-2021-21",
    "CC-MAIN-2021-17",
    "CC-MAIN-2021-13",
    "CC-MAIN-2021-09",
    "CC-MAIN-2021-04",
    "CC-MAIN-2020-50",
    "CC-MAIN-2020-45",
    "CC-MAIN-2020-40",
    "CC-MAIN-2020-34",
    "CC-MAIN-2020-29",
    "CC-MAIN-2020-24",
    "CC-MAIN-2020-16",
    "CC-MAIN-2020-10",
    "CC-MAIN-2019-51",
    "CC-MAIN-2019-47",
    "CC-MAIN-2019-43",
    "CC-MAIN-2019-39",
    "CC-MAIN-2019-35",
    "CC-MAIN-2019-30",
    "CC-MAIN-2019-26",
    "CC-MAIN-2019-22",
    "CC-MAIN-2019-18",
    "CC-MAIN-2019-13",
    "CC-MAIN-2019-09",
    "CC-MAIN-2019-04",
    "CC-MAIN-2018-51",
    "CC-MAIN-2018-47",
    "CC-MAIN-2018-43",
    "CC-MAIN-2018-39",
    "CC-MAIN-2018-34",
    "CC-MAIN-2018-30",
    "CC-MAIN-2018-26",
    "CC-MAIN-2018-22",
    "CC-MAIN-2018-17",
    "CC-MAIN-2018-13",
    "CC-MAIN-2018-09",
    "CC-MAIN-2018-05",
    "CC-MAIN-2017-51",
    "CC-MAIN-2017-47",
    "CC-MAIN-2017-43",
    "CC-MAIN-2017-39",
    "CC-MAIN-2017-34",
    "CC-MAIN-2017-30",
    "CC-MAIN-2017-26",
    "CC-MAIN-2017-22",
    "CC-MAIN-2017-17",
    "CC-MAIN-2017-13",
    "CC-MAIN-2017-09",
    "CC-MAIN-2017-04",
    "CC-MAIN-2016-50",
    "CC-MAIN-2016-44",
    "CC-MAIN-2016-40",
    "CC-MAIN-2016-36",
    "CC-MAIN-2016-30",
    "CC-MAIN-2016-26",
    "CC-MAIN-2016-22",
    "CC-MAIN-2016-18",
    "CC-MAIN-2016-10",
    "CC-MAIN-2016-07",
    "CC-MAIN-2015-48",
    "CC-MAIN-2015-40",
    "CC-MAIN-2015-35",
    "CC-MAIN-2015-32",
    "CC-MAIN-2015-27",
    "CC-MAIN-2015-22",
    "CC-MAIN-2015-18",
    "CC-MAIN-2015-14",
    "CC-MAIN-2015-11",
    "CC-MAIN-2015-06",
    "CC-MAIN-2014-52",
    "CC-MAIN-2014-49",
    "CC-MAIN-2014-42",
    "CC-MAIN-2014-41",
    "CC-MAIN-2014-35",
    "CC-MAIN-2014-23",
    "CC-MAIN-2014-15",
    "CC-MAIN-2014-10",
    "CC-MAIN-2013-48",
    "CC-MAIN-2013-20",
]
