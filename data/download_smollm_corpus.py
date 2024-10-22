"""Download the smollm corpus.

This repo uses fineweb edu and cosmopedia v2 from the smollm corpus.
"""

from gcsfs import GCSFileSystem
from datasets import load_dataset_builder


storage_options = {"project": "distributedmuzerojax"}
fs = GCSFileSystem(**storage_options)

output_dir = "gs://optimizertesting/smollm-corpus"

# fineweb-edu-dedup
print("Downloading fineweb-edu-dedup")
builder = load_dataset_builder(
    "HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", cache_dir="/hf"
)
builder.download_and_prepare(
    output_dir=output_dir + "/fineweb-edu-dedup/train",
    storage_options=storage_options,
    file_format="parquet",
    num_proc=4,
    max_shard_size="300MB",
)

# cosmopedia-v2
print("Downloading cosmopedia-v2")
builder = load_dataset_builder(
    "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", cache_dir="/hf"
)
builder.download_and_prepare(
    output_dir=output_dir + "/cosmopedia-v2/train",
    storage_options=storage_options,
    file_format="parquet",
    num_proc=4,
    max_shard_size="300MB",
)

print("Done downloading")
