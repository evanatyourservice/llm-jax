"""Download the smollm corpus.

This repo uses fineweb edu and cosmopedia v2 from the smollm corpus.
"""

from gcsfs import GCSFileSystem
from datasets import load_dataset


storage_options = {"project": "distributedmuzerojax"}
fs = GCSFileSystem(**storage_options)

output_dir = "gs://optimizertesting/smollm-corpus"

# fineweb-edu-dedup
print("Downloading fineweb-edu-dedup")
dataset = load_dataset(
    "HuggingFaceTB/smollm-corpus",
    "fineweb-edu-dedup",
    split="train",
    cache_dir="/hf",
    num_proc=4,
)
dataset.save_to_disk(
    output_dir + "/fineweb-edu-dedup/train", storage_options=storage_options
)

# cosmopedia-v2
print("Downloading cosmopedia-v2")
dataset = load_dataset(
    "HuggingFaceTB/smollm-corpus",
    "cosmopedia-v2",
    split="train",
    cache_dir="/hf",
    num_proc=4,
)
dataset.save_to_disk(
    output_dir + "/cosmopedia-v2/train", storage_options=storage_options
)

print("Done downloading")
