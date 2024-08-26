from functools import partial
from pprint import pprint
from typing import Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import os
import numpy as np
import wandb
import tyro

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import transformers
import flax
from flax.training import checkpoints
from flax.training.train_state import TrainState
import optax
import tensorflow as tf

from dataset import get_dataset, prepare_hellaswag
from optimizers.psgd_affine import affine, _shape_as_matrix
from optimizers.psgd_xmat import xmat
from optimizers.tearfree import optimizer as tearfree_opt
from optimizers.tearfree import shampoo, second_order
from optimizers.utils import hessian_helper
from sharding import infer_sharding, fsdp_sharding
from utils import reshard, write_note


wandb.require("core")
tf.config.experimental.set_visible_devices([], "GPU")
tf.config.experimental.set_visible_devices([], "TPU")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Transfer guard will fail the program whenever that data between a host and
# a device is transferred implicitly. This often catches subtle bugs that
# cause slowdowns and memory fragmentation. Explicit transfers are done
# with jax.device_put and jax.device_get.
jax.config.update("jax_transfer_guard", "disallow")
# Fixes design flaw in jax.random that may cause unnecessary d2d comms.
jax.config.update("jax_threefry_partitionable", True)


@dataclass(frozen=True)
class GPT2Config:
    vocab_size: int = 50304  # divisible by 64
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int = None
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    layer_norm_epsilon: float = 0.00001
    initializer_range: float = 0.01
    summary_type: str = "cls_index"
    summary_use_proj: bool = True
    summary_activation: str = None
    summary_proj_to_labels: bool = True
    summary_first_dropout: float = 0.0
    scale_attn_weights: bool = True
    use_cache: bool = False
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    scale_attn_by_inverse_layer_idx: bool = True
    reorder_and_upcast_attn: bool = True


@dataclass(frozen=True)
class WandbConfig:
    """
    wandb logging configuration
    """

    entity: str = "evanatyourservice"
    """username or team name where you're sending runs"""
    project: str = "owt"
    """project name"""
    name: str = ""
    """experiment name"""
    mode: str = "online"
    """'offline', 'online', or 'disabled'"""
    notes: str = ""


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
    schedule_precond_lr: bool = False
    precond_init_scale: float = 1.0
    update_global_norm_clip: Optional[float] = 10000


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    out_dir: str = os.path.expanduser(
        "~/gpt_out_dir"
    )  # output directory for checkpoints (can be gcs path)
    train_pattern: str = (
        "owt_data/train_??.tfrecord"  # training files glob pattern (can be gcs path)
    )
    val_pattern: str = (
        "owt_data/val_??.tfrecord"  # validation files glob pattern (can be gcs path)
    )
    min_size_to_shard_mb: int = 4
    shuffle_buffer_size: int = 128
    eval_interval: int = 250
    eval_steps: int = 16  # evaluate for this number of steps (per-device)
    hs_eval_steps: int = 16  # evaluate for this number of steps (per-device)
    keep_checkpoints: int = 0  # number of historical checkpoints to keep
    batch_size: int = 128
    train_steps: int = 100000  # total number of training iterations
    bfloat16_compute: bool = False  # use bfloat16 for compute
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)  # wandb logging
    model: GPT2Config = field(default_factory=GPT2Config)  # gpt model config
    remat: bool = False  # set to True to rematerialize gradients during backward pass


def train_step(
    state: TrainState,
    tokens: jnp.ndarray,
    rng_key,
    bfloat16_compute: bool,
    compute_hessian: bool = False,
    params_sharding: Any = None,
    remat: bool = False,
) -> Tuple[jnp.ndarray, TrainState]:

    rng_key = jax.random.fold_in(rng_key, state.step)

    def loss_fn(params) -> jnp.ndarray:
        X, Y = tokens[:, :-1], tokens[:, 1:]
        if bfloat16_compute:
            X = X.astype(jnp.bfloat16)
            params = optax.tree_utils.tree_cast(params, jnp.bfloat16)
        logits = state.apply_fn(X, params=params, dropout_rng=rng_key, train=True)[0]
        logits = logits.astype(jnp.float32)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y).mean()

        # palm style z loss
        loss += 1e-4 * jax.scipy.special.logsumexp(logits, axis=-1).mean() ** 2

        return loss

    if remat:
        loss_fn = jax.checkpoint(loss_fn)

    if compute_hessian:
        update_prob_schedule = lambda n: jnp.maximum(0.5 * jnp.exp(-0.002 * n), 0.01)
        loss, grads, hvp, vector, update_precond = hessian_helper(
            jax.random.split(rng_key, 1)[0],
            state.step,
            loss_fn=loss_fn,
            params=state.params,
            preconditioner_update_probability=update_prob_schedule(state.step),
            params_sharding=params_sharding,
        )
        updates, new_opt_state = state.tx.update(
            grads,
            state.opt_state,
            state.params,
            Hvp=hvp,
            vector=vector,
            update_preconditioner=update_precond,
        )
        new_params = optax.apply_updates(state.params, updates)
        new_state = state.replace(
            step=state.step + 1, params=new_params, opt_state=new_opt_state
        )
    else:
        loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(state.params)
        new_state = state.apply_gradients(grads=grads)

    return loss, new_state


def eval_step(state: TrainState, tokens: jnp.ndarray) -> jnp.ndarray:
    X, Y = tokens[:, :-1], tokens[:, 1:]
    logits = state.apply_fn(X, params=state.params, train=False)[0]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y).mean()
    return loss


def eval_step_unreduced(state: TrainState, tokens: jnp.ndarray) -> jnp.ndarray:
    X, Y = tokens[:, :-1], tokens[:, 1:]
    logits = state.apply_fn(X, params=state.params, train=False)[0]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y)
    return loss


def eval_hellaswag(state: TrainState, data, labels, lengths):
    """Evaluate the hellaswag dataset."""
    # data comes in shape (b, 4, block_size)
    # labels comes in shape (b,)
    # lengths comes in shape (b, 4)
    batch = jnp.reshape(data, (-1, data.shape[-1]))
    losses = eval_step_unreduced(state, batch)
    losses = jax.vmap(jnp.cumsum)(losses)
    lengths = jnp.reshape(lengths, (-1,))
    losses = jax.vmap(
        lambda x, l: jax.lax.dynamic_index_in_dim(x, l - 2, axis=0, keepdims=False)
    )(losses, lengths)
    choices = jnp.argmin(jnp.reshape(losses, (data.shape[0], data.shape[1])), axis=1)
    correct = jnp.sum(choices == labels)
    accuracy = correct / data.shape[0]
    return accuracy


def count_params(params) -> int:
    p = jax.tree_util.tree_map(
        lambda a: a.size if isinstance(a, jnp.ndarray) else 0, params
    )
    return jax.tree_util.tree_reduce(lambda a, b: a + b, p)


def param_decay_mask(params):
    """pytree mask for non-bias parameters"""
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_param_mask = {
        k: k[-1] not in ("bias", "embedding", "scale") for k in flat_params.keys()
    }
    return flax.traverse_util.unflatten_dict(flat_param_mask)


def get_default_config() -> TrainConfig:
    # use this file to set default values
    path = os.environ.get("GPT_CONFIG", os.path.join("config", "gpt2.yaml"))
    if not os.path.exists(path):
        write_note("using default config")
        return TrainConfig()
    write_note(f"using config file at {path}")
    with open(path, "r") as f:
        return tyro.from_yaml(TrainConfig, f)


def main(config: TrainConfig):
    write_note(f"Number of JAX devices: {jax.device_count()}")
    write_note(f"Number of JAX processes: {jax.process_count()}")

    # set seeds
    np.random.seed(config.seed)
    # keeping tensorflow wild for now

    # wandb
    if config.wandb is not None and jax.process_index() == 0:
        wandb.init(**asdict(config.wandb))
        wandb_config = asdict(config)
        wandb_config["jax_n_devices"] = jax.device_count()
        wandb_config["jax_n_processes"] = jax.process_count()
        wandb.config.update(wandb_config)

    block_size = config.model.n_positions
    using_gpu = jax.devices()[0].platform == "gpu"

    # ===== create device mesh =====
    write_note("creating 1D FSDP mesh")
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    devices_flat = device_mesh.flatten()
    mesh = Mesh(devices=device_mesh, axis_names="fsdp")

    # ===== datasets =====
    write_note("creating datasets")
    train_ds = get_dataset(
        config.train_pattern,
        devices_flat,
        config.batch_size,
        block_size,
        interleave_cycle_length=max(1, 32 // jax.process_count()),
        shuffle_buffer_size=config.shuffle_buffer_size,
        tf_prefetch=10,
        device_prefetch=2 if using_gpu else 1,
    )
    val_ds = get_dataset(
        config.val_pattern,
        devices_flat,
        config.batch_size,
        block_size,
        interleave_cycle_length=max(1, 8 // jax.process_count()),
    )
    # hellaswag has 4 seqs per multiple choice problem
    hellaswag_ds = prepare_hellaswag(
        max(config.batch_size // 4, jax.device_count()),
        block_size,
        devices_flat,
        shuffle_buffer_size=10042 // jax.process_count(),
    )

    # ===== optimizer =====
    write_note("creating optimizer")
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                0.0, config.optimizer.learning_rate, config.optimizer.warmup_steps
            ),
            optax.linear_schedule(
                config.optimizer.learning_rate,
                0.0,
                config.train_steps - config.optimizer.warmup_steps,
            ),
        ],
        boundaries=[config.optimizer.warmup_steps],
    )

    def make_opt(precond_sharding=None):
        write_note(f"using {config.optimizer.type} optimizer")

        optimizer = []
        if config.optimizer.grad_clip > 0.0:
            optimizer.append(optax.clip_by_global_norm(config.optimizer.grad_clip))

        if config.optimizer.type in ["adam", "adamw"]:
            optimizer.append(
                optax.adamw(
                    lr_schedule,
                    *config.optimizer.betas,
                    weight_decay=config.optimizer.weight_decay,
                    mask=param_decay_mask,
                    mu_dtype=jnp.bfloat16,
                )
            )
        elif config.optimizer.type in ["psgd_affine", "affine"]:
            update_prob_schedule = lambda n: jnp.maximum(jnp.exp(-0.002 * n), 0.01)
            optimizer.append(
                affine(
                    lr_schedule,
                    update_prob_schedule,
                    b1=config.optimizer.betas[0],
                    weight_decay=config.optimizer.weight_decay,
                    mask=param_decay_mask,
                    max_size_triangular=config.optimizer.max_size_triangular,
                    max_skew_triangular=config.optimizer.max_skew_triangular,
                    precond_lr=config.optimizer.precond_lr,
                    precond_init_scale=config.optimizer.precond_init_scale,
                    update_global_norm_clip=config.optimizer.update_global_norm_clip,
                    momentum_before_precond_update=True,  # experimental
                    mu_dtype=jnp.bfloat16,
                    precision="tensorfloat32",
                    precond_sharding=precond_sharding,
                )
            )
        elif config.optimizer.type in ["psgd_xmat", "xmat"]:
            update_prob_schedule = lambda n: jnp.maximum(jnp.exp(-0.002 * n), 0.02)
            optimizer.append(
                xmat(
                    lr_schedule,
                    update_prob_schedule,
                    b1=config.optimizer.betas[0],
                    weight_decay=config.optimizer.weight_decay,
                    mask=param_decay_mask,
                    precond_lr=config.optimizer.precond_lr,
                    precond_init_scale=config.optimizer.precond_init_scale,
                    update_global_norm_clip=config.optimizer.update_global_norm_clip,
                    momentum_before_precond_update=True,  # experimental
                    mu_dtype=jnp.bfloat16,
                    precond_dtype=jnp.float32,
                    precision="tensorfloat32",
                    precond_sharding=precond_sharding,
                )
            )
        elif config.optimizer.type == "shampoo":
            optimizer.append(
                tearfree_opt.tearfree(
                    lr_schedule,
                    tearfree_opt.TearfreeOptions(
                        momentum_options=tearfree_opt.momentum.Options(
                            weight_decay=config.optimizer.weight_decay,
                            momentum_decay=config.optimizer.betas[0],
                            momentum_dtype="bfloat16",
                        ),
                        second_order_options=second_order.Options(
                            shampoo_options=shampoo.Options(use_CASPR_variant=False)
                        ),
                    ),
                )
            )
        else:
            raise ValueError("Unknown optimizer type")

        optimizer.append(
            optax.apply_every(config.optimizer.gradient_accumulation_steps)
        )
        return optax.chain(*optimizer)

    # ===== shard and transfer =====
    write_note("creating and sharding train state")
    repl_sharding = NamedSharding(mesh, P())
    data_sharding = NamedSharding(mesh, P("fsdp"))

    rng = jax.random.PRNGKey(jax.device_put(config.seed, jax.devices("cpu")[0]))

    def init_train_state(key):
        model_config = transformers.GPT2Config(**asdict(config.model))
        model = transformers.FlaxAutoModelForCausalLM.from_config(
            model_config, _do_init=False
        )
        params = model.init_weights(rng=key, input_shape=(1, config.model.n_positions))
        # delay optimizer creation to pass in preconditioner sharding
        train_state = TrainState(
            step=0, apply_fn=model.__call__, params=params, tx=None, opt_state=None
        )
        return train_state

    train_state_shapes = jax.eval_shape(init_train_state, rng)

    op = fsdp_sharding("fsdp", min_size_to_shard_mb=config.min_size_to_shard_mb)
    train_state_sharding = infer_sharding(params=train_state_shapes, mesh=mesh, op=op)

    rng, rng_init = jax.random.split(rng, 2)
    rng_init = reshard(rng_init, repl_sharding)

    train_state = jax.jit(init_train_state, out_shardings=train_state_sharding)(
        rng_init
    )

    # make optimizer and its shardings
    optimizer = make_opt()

    opt_state_shapes = jax.eval_shape(optimizer.init, train_state.params)
    opt_state_sharding = infer_sharding(params=opt_state_shapes, mesh=mesh, op=op)
    opt_state = jax.jit(optimizer.init, out_shardings=opt_state_sharding)(
        train_state.params
    )

    if config.optimizer.type in ["psgd_affine", "affine"]:

        def get_precond_sharding(params):
            """Follows PSGD affine matrix reshaping and applies sharding strategy."""
            # returns tuples if (reshape_fn, unreshape_fn, shape)
            affine_reshapers = [_shape_as_matrix(x) for x in jax.tree.leaves(params)]
            # grab preconditioner shapes and make jax.ShapeDtypeStructs
            precond_shapes = [
                jax.ShapeDtypeStruct(r[2], jnp.float32) for r in affine_reshapers
            ]
            # apply sharding strategy
            return infer_sharding(params=precond_shapes, mesh=mesh, op=op)

        precond_sharding = get_precond_sharding(train_state.params)
    else:
        # psgd xmat preconditioners are same shapes as params
        precond_sharding = train_state_sharding.params

    # remake optimizer with preconditioner sharding passed in
    optimizer = make_opt(precond_sharding=precond_sharding)

    # finish making train state (pass in optimizer and opt_state)
    train_state = train_state.replace(tx=optimizer, opt_state=opt_state)
    train_state_sharding = train_state_sharding.replace(
        tx=optimizer, opt_state=opt_state_sharding
    )

    rng = reshard(rng, repl_sharding)

    num_params = count_params(train_state.params)
    if jax.process_index() == 0:
        write_note("TRAIN STATE SHAPES:")
        pprint(
            jax.tree.map(lambda x: x.shape, train_state),
            indent=2,
            width=120,
            compact=True,
        )
        write_note("TRAIN STATE SHARDING:")
        pprint(train_state_sharding, indent=2, width=120, compact=True)
        write_note(f"PARAMETER COUNT: {num_params:,}")

    # ==== restore train state ====
    # restore unreplicated optimizer + model state from last checkpoint.
    # this is a no-op if no checkpoints exist
    if config.keep_checkpoints > 0:  # TODO implement checkpointing to wandb
        train_state = checkpoints.restore_checkpoint(
            f"{config.out_dir}/checkpoints/train_state", train_state
        )

    # ====== jit functions ========
    # we specify in_shardings for sake of clarity, but they are inferred
    write_note(f"PSGD use Hessian = {config.optimizer.psgd_use_hessian}")
    train_step_w_sharding = partial(
        train_step,
        bfloat16_compute=config.bfloat16_compute,
        compute_hessian=config.optimizer.psgd_use_hessian,
        params_sharding=train_state_sharding.params,
        remat=config.remat,
    )
    train_step_jit = jax.jit(
        train_step_w_sharding,
        donate_argnames=("state",),
        in_shardings=(train_state_sharding, data_sharding, rng.sharding),
        out_shardings=(repl_sharding, train_state_sharding),
    )
    eval_step_jit = jax.jit(
        eval_step,
        in_shardings=(train_state_sharding, data_sharding),
        out_shardings=repl_sharding,
    )
    eval_hellaswag_jit = jax.jit(
        eval_hellaswag,
        in_shardings=(
            train_state_sharding,
            data_sharding,
            data_sharding,
            data_sharding,
        ),
        out_shardings=repl_sharding,
    )
    get_lr = jax.jit(
        lr_schedule, in_shardings=repl_sharding, out_shardings=repl_sharding
    )

    # ======= train ========
    # grab step from last checkpoint
    step = 0

    best_val_loss = float("inf")

    train_losses = []
    write_note("starting training")
    for step in range(step, config.train_steps):
        tokens = next(train_ds)
        loss, train_state = train_step_jit(train_state, tokens, rng)
        train_losses.append(jax.device_get(loss).item())

        if (config.wandb is not None) and (jax.process_index() == 0) and step % 10 == 0:
            train_loss = np.mean(train_losses)
            wandb.log(
                {
                    "train_loss": train_loss,
                    "lr": (
                        jax.device_get(get_lr(jax.device_put(step, repl_sharding)))
                        if callable(lr_schedule)
                        else lr_schedule
                    ),
                    "tokens": step
                    * config.batch_size
                    * jax.device_count()
                    * block_size,
                },
                step=step,
            )

            train_losses = []

        if step % config.eval_interval == 0 and step > 0:
            val_losses = []
            for _ in range(config.eval_steps):
                tokens = next(val_ds)
                loss = eval_step_jit(train_state, tokens)
                val_losses.append(jax.device_get(loss).item())
            val_loss = np.mean(val_losses)

            # hellaswag
            hs_accs = []
            for _ in range(config.hs_eval_steps):
                hs_batch = next(hellaswag_ds)
                hs_acc = eval_hellaswag_jit(train_state, *hs_batch)
                hs_accs.append(jax.device_get(hs_acc).item())
            hellaswag_acc = np.mean(hs_accs)

            write_note(
                f"step: {step}, val_loss: {val_loss:.4f}, "
                f"hellaswag_acc: {hellaswag_acc:.4f}"
            )

            if val_loss < best_val_loss and config.keep_checkpoints > 0:
                best_val_loss = val_loss
                if jax.process_index() == 0:
                    # save train state in process 0
                    checkpoints.save_checkpoint(
                        f"{config.out_dir}/checkpoints/train_state",
                        jax.device_get(train_state),
                        step,
                        keep=config.keep_checkpoints,
                        overwrite=True,
                    )

            if (config.wandb is not None) and (jax.process_index() == 0):
                wandb.log(
                    {"val_loss": val_loss, "hellaswag_acc": hellaswag_acc}, step=step
                )


if __name__ == "__main__":
    config = tyro.cli(TrainConfig, default=get_default_config(), use_underscores=True)
    main(config)
