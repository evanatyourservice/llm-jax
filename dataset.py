from itertools import chain
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
    shard_idx: int = 0,
    start_index: int = 0,
):
    """Load the fineweb-edu dataset.

    For now we load in a weird way to save memory and from having to
    use a bucket... not ideal.
    """
    seq_len = block_size + 1

    platform = jax.devices()[0].platform
    # use /dev/shm if on a TPU vm for more space
    if platform == "tpu":
        cache_dir = "/dev/shm/huggingface_cache"
    else:
        cache_dir = None

    if streaming:
        write_note("streaming fineweb-edu")

        def gen():
            hf_ds = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                split="train",
                cache_dir=cache_dir,
                streaming=True,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True, use_fast=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            def tokenize(text):
                return tokenizer(
                    text, max_length=seq_len, padding="max_length", truncation=True
                )

            hf_ds = hf_ds.map(tokenize, input_columns="text")

            hf_ds = hf_ds.with_format("numpy")

            for example in hf_ds:
                yield example["input_ids"]

        ds = tf.data.Dataset.from_generator(
            gen, output_signature=tf.TensorSpec(shape=(seq_len,), dtype=tf.int32)
        )
        ds = ds.shuffle(10000)
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

        fineweb_files_list = _fw_shard_names
        rng = np.random.RandomState(100)
        rng.shuffle(fineweb_files_list)

        n_procs = jax.process_count()
        curr_proc = jax.process_index()

        # split into n_procs shards
        fineweb_shards = [
            fineweb_files_list[curr_proc::n_procs] for _ in range(n_procs)
        ]

        # grab current shard and subshard using shard_idx
        proc_shard = fineweb_shards[curr_proc]
        proc_subshard = proc_shard[shard_idx % len(proc_shard)]

        print(f"process {curr_proc} shard {proc_shard}")
        print(f"process {curr_proc} subshard {proc_subshard}")

        hf_ds: Dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=proc_subshard,
            split="train",
            cache_dir=cache_dir,
        )

        hf_ds = hf_ds.skip(start_index)

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True, use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def tokenize(data_chunk):
            text = data_chunk["text"]
            return tokenizer(
                text, max_length=seq_len, padding="max_length", truncation=True
            )

        hf_ds.set_transform(tokenize)

        hf_ds.cleanup_cache_files()

        ds = hf_ds.to_tf_dataset(columns="input_ids", prefetch=False)
        ds = ds.shuffle(10000)
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
