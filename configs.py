from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple, Optional


@dataclass(frozen=True)
class ModelConfig:
    """model configuration.

    Attributes:
        model_type: str, 'gemma2_test', 'gemma_2b', 'gemma_7b', 'smollm_135m',
            'smollm_360m', 'smollm_1_7b', 'gemma2_2b', 'gemma2_9b', 'gemma2_27b'
        sliding_window_size: int, default 512, sliding window size
        block_size: int, default 1024, total sequence length
    """

    model_type: str = "smollm_135m"
    sliding_window_size: int = 512
    block_size: int = 1024

    assert (
        sliding_window_size <= block_size
    ), "sliding_window_size must be less than or equal to block_size"


@dataclass(frozen=True)
class OptimizerConfig:
    """optimizer configuration.

    Attributes:
        type: optimizer type.
        learning_rate: learning rate.
        warmup_steps: warmup steps.
        weight_decay: weight decay.
        grad_clip: gradient clip.
        gradient_accumulation_steps: gradient accumulation steps.
        betas: betas.
        nesterov: whether to use nesterov momentum.
        preconditioner_update_probability: probability of updating the
            preconditioner.
        max_size_triangular: max size for affine preconditioner to be
            triangular.
        max_skew_triangular: max skew for affine preconditioner to be
            triangular.
        precond_lr: learning rate for the preconditioner.
        precond_init_scale: initial scale for the preconditioner.
        preconditioner_dtype: dtype of the preconditioner.
    """

    type: str = "adamw"
    learning_rate: float = 0.001
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    betas: Tuple[float, float] = (0.9, 0.95)
    nesterov: bool = False
    preconditioner_update_probability: float = 1.0
    max_size_triangular: int = 8192
    max_skew_triangular: int = 16
    precond_lr: float = 0.1
    precond_init_scale: Optional[float] = 1.0
    preconditioner_dtype: str = "float32"


@dataclass(frozen=True)
class WandbConfig:
    """wandb logging configuration.

    Attributes:
        entity: Username or team name where you're sending runs.
        project: Project name.
        name: Experiment name.
        mode: Can be 'offline', 'online', or 'disabled'.
        notes: Notes for this run.
    """

    entity: str = "evanatyourservice"
    project: str = "llm-jax"
    name: str = ""
    mode: str = "online"
    notes: str = ""


date_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@dataclass(frozen=True)
class TrainConfig:
    """training configuration.

    Attributes:
        seed: random seed.
        out_dir: output directory for checkpoints (can be gcs path).
        attempt_to_load_checkpoint: whether to attempt to load a checkpoint.
        only_print_model: whether to only print the model then quit.
        min_size_to_shard_mb: minimum size of shards to create.
        hellaswag_eval_interval: interval to evaluate hellaswag.
        checkpoint_interval: interval to save checkpoints.
        checkpoint_milestone: milestone to save checkpoints.
        keep_checkpoints: number of historical checkpoints to keep.
        batch_size: batch size.
        train_steps: total number of training iterations.
        compute_dtype: compute dtype.
        params_dtype: params dtype.
        optimizer: optimizer config.
        wandb: wandb logging config.
        model: model config.
    """

    seed: int = 8
    out_dir: str = f"gs://uscentral2stuff/llm-jax/run_{date_and_time}"
    attempt_to_load_checkpoint: bool = False
    only_print_model: bool = False
    min_size_to_shard_mb: int = 0.2
    hellaswag_eval_interval: int = 500
    checkpoint_interval: int = 1000
    keep_checkpoints: int = 2
    checkpoint_milestone: int = 10000
    batch_size: int = 128
    train_steps: int = 100000
    compute_dtype: str = "float32"
    params_dtype: str = "float32"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    assert (
        checkpoint_milestone % checkpoint_interval == 0
    ), "checkpoint_milestone must be a multiple of checkpoint_interval"
