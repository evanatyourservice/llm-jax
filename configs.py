from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple, Optional


@dataclass(frozen=True)
class LlamaModelConfig:
    llama_huggingface_model_name: str = (
        "trl-internal-testing/tiny-random-LlamaForCausalLM"
    )
    tokenizer_name: Optional[str] = None
    use_scan_mlp: bool = False
    block_size: int = 1024


@dataclass(frozen=True)
class OptimizerConfig:
    type: str = "adamw"
    learning_rate: float = 0.001
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    betas: Tuple[float, float] = (0.9, 0.95)
    preconditioner_update_probability: float = 1.0
    psgd_use_hessian: bool = False
    max_size_triangular: int = 0
    max_skew_triangular: int = 0
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

    seed: int = 0
    out_dir: str = f"gs://uscentral2stuff/llm-jax/run_{date_and_time}"
    attempt_to_load_checkpoint: bool = False
    min_size_to_shard_mb: int = 4
    hellaswag_eval_interval: int = 1000
    checkpoint_interval: int = 1000
    keep_checkpoints: int = 2  # number of historical checkpoints to keep
    checkpoint_milestone: int = 10000
    batch_size: int = 128
    train_steps: int = 100000  # total number of training iterations
    compute_dtype: str = "float32"  # "float32" or "bfloat16"
    params_dtype: str = "float32"  # "float32" or "bfloat16"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)  # wandb logging
    model: LlamaModelConfig = field(default_factory=LlamaModelConfig)

    assert (
        checkpoint_milestone % checkpoint_interval == 0
    ), "checkpoint_milestone must be a multiple of checkpoint_interval"
