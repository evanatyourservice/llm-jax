from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class ModelConfig:
    """Default model config for 125M.

    Mistral-like model with default arch similar to MobileLLM and SmolLM.

    Attributes:
        block_size: Block size.
        sliding_window_size: Sliding window size.
        vocab_size: Vocabulary size.
        num_layers: Number of layers.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key-value heads.
        head_dim: Head dimension.
        num_embeds: Number of embeddings.
        hidden_dim: Hidden dimension.
        rope_theta: Rotary embedding theta.
        scan_layers: Whether to scan layers.
        remat: Whether to use remat. Should be used if scanning layers.
        remat_everything: Whether to remat everything, otherwise only use
            `checkpoint_dots_with_no_batch_dims`.
    """

    block_size: int = 2048
    sliding_window_size: int = 1024
    vocab_size: int = 32768
    num_layers: int = 30
    num_heads: int = 9
    num_kv_heads: int = 3
    head_dim: int = 576 // 9
    num_embeds: int = 576
    hidden_dim: int = 1536
    rope_theta: float = 1000000.0
    scan_layers: bool = False
    remat: bool = False
    remat_everything: bool = False


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer configuration.

    Attributes:
        type: Optimizer type, one of ["adamw", "psgd", "shampoo", "caspr",
            "schedule_free"]
        learning_rate: Learning rate.
        warmup_steps: Warmup steps.
        flat_lr: Whether to use a flat learning rate or decay linearly to 0.05x.
        weight_decay: Weight decay.
        grad_clip: Gradient clip.
        b1: Beta 1.
        b2: Beta 2.
        eps: Epsilon.
        nesterov: Whether to use nesterov momentum.
        preconditioner_update_probability: Probability of updating the
            preconditioner in PSGD. Default for PSGD kron is 0.03.
        max_size_triangular: Max dim size for preconditioner to be triangular
            in PSGD.
        max_skew_triangular: Max skew for preconditioner to be triangular
            in PSGD.
        preconditioner_dtype: Dtype of the preconditioner in PSGD.
        lax_map_scanned_layers: Whether to use lax.map for scanned layers instead
            of vmap. Useful for large models (>1B) to save memory.
        lax_map_batch_size: Batch size for lax.map, see jax docs for more info.
    """

    type: str = "kron"
    learning_rate: float = 0.001
    warmup_steps: int = 1000
    flat_lr: bool = False
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    nesterov: bool = False
    preconditioner_update_probability: float = 0.03
    max_size_triangular: int = 8192
    max_skew_triangular: float = 1000000000.0
    preconditioner_dtype: str = "float32"
    lax_map_scanned_layers: bool = False
    lax_map_batch_size: int = 8


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
        gradient_accumulation_steps: Number of gradient accumulation steps.
        compute_dtype: Compute dtype.
        params_dtype: Params dtype.
        profile: Whether to profile the training to tensorboard.
        n_profile_steps: Number of steps to profile.
        optimizer: Optimizer config.
        wandb: Wandb logging config.
        model: Model config.
    """

    seed: int = 10
    experiment_name: str = f"run_{date_and_time}"
    out_dir: str = "gs://optimizertesting/llm-jax"
    attempt_to_load_checkpoint: bool = True
    only_print_model: bool = False
    min_size_to_shard_mb: int = 0.1
    hellaswag_eval_interval: int = 1000
    checkpoint_interval: int = 1000
    keep_checkpoints: int = 1
    checkpoint_milestone: int = 25000
    batch_size: int = 256
    train_steps: int = 50000
    gradient_accumulation_steps: int = 1
    compute_dtype: str = "float32"
    params_dtype: str = "float32"
    profile: bool = False
    n_profile_steps: int = 5
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    assert (
        hellaswag_eval_interval % 100 == 0
    ), "Hellaswag_eval_interval must be a multiple of 100"


def get_default_config() -> TrainConfig:
    return TrainConfig()
