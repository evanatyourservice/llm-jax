import os
import json
import numpy as np
from tqdm import tqdm
import random

import jax
import tensorflow as tf
import datasets
from datasets import load_dataset, IterableDataset
import datasets.config
import tiktoken

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


def prepare_hellaswag(
    batch_size: int,
    block_size: int,
    flat_devices,
    tf_prefetch: int = 2,
    device_prefetch: int = 0,
):
    """Read file and tokenize the hellaswag dataset."""
    write_note("preparing hellaswag")

    tokenizer = tiktoken.get_encoding("gpt2")

    all_data = []
    all_seq_lens = []
    all_labels = []
    with open("data/hellaswag_val.jsonl", "r") as f:
        # iterate over lines and tokenize
        for line in tqdm(f, total=10042):
            item = json.loads(line)
            context = item["ctx"]
            endings = item["endings"]
            correct_end = item["label"]
            data_to_concat = []
            seq_lens_to_concat = []
            for ending in endings:
                input_text = context + " " + ending
                # encode with gpt2 tokenizer
                output = tokenizer.encode(input_text)
                # output len, at least block_size
                output_len = min(len(output), block_size)
                # pad with eot_token if less than block_size
                if output_len < block_size:
                    output = output + [tokenizer.eot_token] * (block_size - output_len)
                # max length is block_size
                output = output[:block_size]
                data_to_concat.append(output)
                seq_lens_to_concat.append(output_len)
            all_data.append(np.array(data_to_concat))  # (4, seq_len)
            all_seq_lens.append(np.array(seq_lens_to_concat))  # (4,)
            all_labels.append(int(correct_end))  # []

    all_data = np.array(all_data, dtype=np.uint16)  # (10042, 4, seq_len)
    all_seq_lens = np.array(all_seq_lens, dtype=np.int32)  # (10042, 4)
    all_labels = np.array(all_labels, dtype=np.int32)  # (10042,)

    ds = tf.data.Dataset.from_tensor_slices((all_data, all_seq_lens, all_labels))
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
    batch_size: int,
    block_size: int,
    flat_devices,
    tf_prefetch: int = 5,
    device_prefetch: int = 0,
    shard_idx: int = 0,
):
    """Load the fineweb-edu dataset."""
    platform = jax.devices()[0].platform
    # use /dev/shm if on a TPU vm for more space
    if platform == "tpu":
        cache_dir = "/dev/shm/huggingface_cache"
    else:
        cache_dir = None

    tokenizer = tiktoken.get_encoding("gpt2")

    if jax.process_count() == 1:
        # just stream fineweb-edu regularly
        proc_subshard = None
    else:
        # use different shards pre process
        # grab current shard and shuffle
        proc_shard = _fw_shard_names[jax.process_index() :: jax.process_count()]
        random.shuffle(proc_shard)

        proc_subshard = proc_shard[shard_idx % len(proc_shard)]

    def gen():
        hf_ds: IterableDataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            name=proc_subshard,
            cache_dir=cache_dir,
            streaming=True,
        )

        def tokenize(example):
            tokenized = tokenizer.encode_batch(
                example,
                num_threads=64,
            )
            tokenized = [t + [tokenizer.eot_token] for t in tokenized]
            return {"tokens": tokenized}

        hf_ds = hf_ds.map(
            tokenize, input_columns=["text"], batched=True, batch_size=128
        )

        hf_ds = hf_ds.with_format("numpy")

        for example in hf_ds:
            yield example["tokens"].astype(np.uint16)

    ds = tf.data.Dataset.from_generator(
        gen, output_signature=tf.TensorSpec(shape=(None,), dtype=tf.uint16)
    )
    ds = ds.shuffle(128)  # shuffle dataset examples
    ds = ds.unbatch()
    ds = ds.batch(block_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(10240)  # shuffle blocks
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
    "CC-MAIN-2021-10",
    "CC-MAIN-2021-04",
    "CC-MAIN-2020-50",
    "CC-MAIN-2020-45",
    "CC-MAIN-2020-40",
    "CC-MAIN-2020-34",
    "CC-MAIN-2020-29",
    "CC-MAIN-2020-24",
    "CC-MAIN-2020-16",
    "CC-MAIN-2020-10",
    "CC-MAIN-2020-05",
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
