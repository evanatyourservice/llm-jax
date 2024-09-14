from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple, Optional
import tyro
import os

from utils import write_note


@dataclass(frozen=True)
class ModelConfig:
    block_size: int = 2048
    sliding_window_size: int = 1024
    vocab_size: int = 32768
    num_layers: int = 30
    num_heads: int = 9
    num_kv_heads: int = 3
    head_dim: int = 576 // 9
    num_embeds: int = 576
    hidden_dim: int = 1536
    scan_layers: bool = False
    scan_unroll: int = 4


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer configuration.

    Attributes:
        type: Optimizer type, one of ["adamw", "psgd_affine", "shampoo",
            "caspr", "schedule_free"]
        learning_rate: Learning rate.
        warmup_steps: Warmup steps.
        weight_decay: Weight decay.
        grad_clip: Gradient clip.
        gradient_accumulation_steps: Gradient accumulation steps.
        b1: Beta 1.
        b2: Beta 2.
        nesterov: Whether to use nesterov momentum.
        preconditioner_update_probability: Probability of updating the
            preconditioner.
        max_size_triangular: Max size for affine preconditioner to be
            triangular.
        max_skew_triangular: Max skew for affine preconditioner to be
            triangular.
        precond_lr: Learning rate for the preconditioner.
        precond_init_scale: Initial scale for the preconditioner.
        preconditioner_dtype: Dtype of the preconditioner.
    """

    type: str = "adamw"
    learning_rate: float = 0.001
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    b1: float = 0.9
    b2: float = 0.95
    nesterov: bool = False
    preconditioner_update_probability: float = 0.02
    max_size_triangular: int = 4096
    max_skew_triangular: int = 10
    precond_lr: float = 0.1
    precond_init_scale: Optional[float] = 1.0
    preconditioner_dtype: str = "float32"


@dataclass(frozen=True)
class WandbConfig:
    """Wandb logging configuration."""

    entity: str = ""
    project: str = "llm-jax"
    mode: str = "online"


date_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration.

    Attributes:
        experiment_name: Name of the experiment.
        out_dir: Output directory for checkpoints (can be gcs path).
        attempt_to_load_checkpoint: Whether to attempt to load a checkpoint.
        only_print_model: Whether to only print the model then quit.
        min_size_to_shard_mb: Minimum size of shards to create.
        hellaswag_eval_interval: Interval to evaluate hellaswag.
        checkpoint_interval: Interval to save checkpoints.
        checkpoint_milestone: Milestone to save checkpoints.
        keep_checkpoints: Number of historical checkpoints to keep.
        batch_size: Batch size.
        train_steps: Total number of training iterations.
        compute_dtype: Compute dtype.
        params_dtype: Params dtype.
        optimizer: Optimizer config.
        wandb: Wandb logging config.
        model: Model config.
    """

    seed: int = 10
    experiment_name: str = f"run_{date_and_time}"
    out_dir: str = "gs://uscentral2stuff/llm-jax"
    attempt_to_load_checkpoint: bool = True
    only_print_model: bool = False
    min_size_to_shard_mb: int = 0.1
    hellaswag_eval_interval: int = 1000
    checkpoint_interval: int = 1000
    keep_checkpoints: int = 2
    checkpoint_milestone: int = 25000
    batch_size: int = 128
    train_steps: int = 100000
    compute_dtype: str = "float32"
    params_dtype: str = "float32"
    remat: bool = False
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    assert (
        hellaswag_eval_interval % 100 == 0
    ), "Hellaswag_eval_interval must be a multiple of 100"


def get_default_config() -> TrainConfig:
    # Use this file to set default values
    path = os.environ.get("LLM_CONFIG", os.path.join("config", "mistral.yaml"))
    if not os.path.exists(path):
        write_note("Using default config")
        return TrainConfig()
    write_note(f"Using config file at {path}")
    with open(path, "r") as f:
        return tyro.from_yaml(TrainConfig, f)
