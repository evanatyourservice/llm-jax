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


def smollm_corpus_dataset(
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
    """
    Load the smollm corpus dataset.

    For now we load in a weird way to save memory and from having to
    use a bucket... not ideal.

    fineweb edu deduplicated file structure:
    `fineweb-edu-dedup/train-00000-of-00234.parquet`

    cosmopedia v2 file structure:
    `cosmopedia-v2/train-00000-of-00104.parquet`
    """
    seq_len = block_size + 1

    platform = jax.devices()[0].platform
    # use /dev/shm if on a TPU vm for more space
    if platform == "tpu":
        cache_dir = "/dev/shm/huggingface_cache"
    else:
        cache_dir = None

    if streaming:
        write_note("streaming smollm-corpus fineweb-edu-dedup")

        def gen():
            hf_ds = load_dataset(
                "HuggingFaceTB/smollm-corpus",
                "fineweb-edu-dedup",
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
        write_note("loading smollm-corpus")

        fineweb_files_list = [
            f"fineweb-edu-dedup/train-{i:05d}-of-00234.parquet" for i in range(234)
        ]
        cosmo_files_list = [
            f"cosmopedia-v2/train-{i:05d}-of-00104.parquet" for i in range(104)
        ]
        rng = np.random.RandomState(100)
        rng.shuffle(fineweb_files_list)
        rng.shuffle(cosmo_files_list)

        n_procs = jax.process_count()
        curr_proc = jax.process_index()

        n_cosmo = n_procs // 4
        n_fineweb = n_procs - n_cosmo

        fineweb_shards = [fineweb_files_list[i::n_fineweb] for i in range(n_fineweb)]
        cosmo_shards = [cosmo_files_list[i::n_cosmo] for i in range(n_cosmo)]

        def chunks(L, n):
            for i in range(0, len(L), n):
                yield L[i : i + n]

        # join the lists with every fourth shard being cosmopedia
        if n_procs > 3:
            zipper = zip(chunks(fineweb_shards, 3), cosmo_shards)
            shards = list(chain.from_iterable((*x, y) for x, y in zipper))
        else:
            shards = fineweb_shards
        assert len(shards) == n_procs

        proc_shard = shards[curr_proc]
        is_cosmo_shard = curr_proc + 1 % 4 == 0

        if is_cosmo_shard:
            max_files_per_subshard = 20
        else:  # fineweb
            max_files_per_subshard = 10

        # grab current subshard using shard_idx
        n_shard_files = len(proc_shard)
        n_subshards = n_shard_files // max_files_per_subshard
        temp_shard_idx = shard_idx % n_subshards
        start_idx = temp_shard_idx * max_files_per_subshard
        end_idx = (
            start_idx + max_files_per_subshard
            if temp_shard_idx < n_subshards - 1
            else n_shard_files
        )
        proc_subshard = proc_shard[start_idx:end_idx]

        print(
            f"process {curr_proc} data files ({len(proc_subshard)} files): "
            f"{proc_subshard}"
        )

        hf_ds: Dataset = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            "cosmopedia-v2" if is_cosmo_shard else "fineweb-edu-dedup",
            split="train",
            data_files=proc_subshard,
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
