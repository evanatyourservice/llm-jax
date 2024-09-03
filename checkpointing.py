import os
import pickle
from pathlib import Path
from typing import Any
import pickle
from google.cloud import storage
from google.auth import default

import jax

from utils import write_note


class Checkpointer(object):
    def __init__(
        self,
        checkpoint_dir: str,
        enable: bool = True,
        save_every_n: int = 1000,
        save_milestone_every_n: int = 10000,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.enable = enable and save_every_n > 0
        self.save_every_n = save_every_n
        self.save_milestone_every_n = save_milestone_every_n
        self.checkpoint_name = "train_state"

        if self.enable:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _save_checkpoint(self, filename, data, data_gather_fn=None):
        if not self.enable:
            return

        path = os.path.join(self.checkpoint_dir, filename)

        if data_gather_fn is not None:
            data = jax.device_get(data_gather_fn(data))

        save_pickle(data, path)

    def save_all(self, step, data, data_gather_fn=None, final: bool = False):
        if step > 0 and step % self.save_every_n == 0:
            self._save_checkpoint(
                f"{self.checkpoint_name}.pickle", data, data_gather_fn
            )
            write_note(f"Saved checkpoint at step {step} to {self.checkpoint_dir}")

        if step > 0 and step % self.save_milestone_every_n == 0 or final:
            step_str = "final" if final else step
            self._save_checkpoint(
                f"{self.checkpoint_name}_{step_str}.pickle", data, data_gather_fn
            )
            write_note(
                f"Saved milestone checkpoint at step {step} to {self.checkpoint_dir}"
            )

    def load_trainstate_checkpoint(self, load_shard_fn=None):
        path = os.path.join(self.checkpoint_dir, f"{self.checkpoint_name}.pickle")
        try:
            train_state = load_pickle(path)
        except FileNotFoundError:
            return None

        if load_shard_fn is not None:
            train_state = load_shard_fn(train_state)

        return train_state


def save_pickle(data: Any, path: str) -> None:
    use_gcs = path.startswith("gs://")
    if use_gcs:
        credentials, _ = default()
        client = storage.Client(credentials=credentials)
        bucket_name, blob_name = path.replace("gs://", "").split("/", 1)
        try:
            bucket = client.get_bucket(bucket_name)
        except Exception as e:
            raise ValueError(f"Error accessing GCS bucket '{bucket_name}': {str(e)}")
        blob = bucket.blob(blob_name)
        with blob.open("wb") as f:
            pickle.dump(jax.device_get(data), f)
    else:
        path = Path(path)
        suffix = ".pickle"
        if path.suffix != suffix:
            path = path.with_suffix(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        with open(path, "wb") as file:
            pickle.dump(data, file)


def load_pickle(path: str) -> Any:
    use_gcs = path.startswith("gs://")
    if use_gcs:
        credentials, _ = default()
        client = storage.Client(credentials=credentials)
        bucket_name, blob_name = path.replace("gs://", "").split("/", 1)
        try:
            bucket = client.get_bucket(bucket_name)
        except Exception as e:
            raise ValueError(f"Error accessing GCS bucket '{bucket_name}': {str(e)}")
        blob = bucket.blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with blob.open("rb") as f:
            return pickle.load(f)
    else:
        suffix = ".pickle"
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix != suffix:
            raise ValueError(f"Not a {suffix} file: {path}")
        with open(path, "rb") as file:
            return pickle.load(file)
