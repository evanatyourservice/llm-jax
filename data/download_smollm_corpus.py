import gcsfs
from datasets import load_dataset_builder


storage_options = {"project": "distributedmuzerojax"}
fs = gcsfs.GCSFileSystem(**storage_options)


# fineweb-edu-dedup
print("Downloading fineweb-edu-dedup")
output_dir = "gs://optimizertesting/smollm-corpus/fineweb-edu-dedup"
builder = load_dataset_builder("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", cache_dir="/hf")
builder.download_and_prepare(
    output_dir=output_dir, storage_options=storage_options, file_format="parquet"
)

# cosmopedia-v2
print("Downloading cosmopedia-v2")
output_dir = "gs://optimizertesting/smollm-corpus/cosmopedia-v2"
builder = load_dataset_builder("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", cache_dir="/hf")
builder.download_and_prepare(
    output_dir=output_dir, storage_options=storage_options, file_format="parquet"
)

print("Done downloading")
