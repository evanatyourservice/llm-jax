from functools import partial
from pprint import pprint
import shutil
import time
from typing import Tuple
from dataclasses import asdict
import os
import random
import numpy as np
import wandb
import tyro

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import flax
from flax.training import checkpoints
from flax.training.train_state import TrainState as ts
import optax
import optax.tree_utils as otu
import tensorflow as tf
import easydel as ed

from dataset import prepare_hellaswag, fineweb_edu_dataset
from configs import TrainConfig
from optimizers.psgd_affine import affine, _shape_as_matrix
from optimizers.tearfree import optimizer as tearfree_opt
from optimizers.tearfree import shampoo, second_order
from sharding import infer_sharding, fsdp_sharding
from utils import check_dtypes, reshard, write_note


wandb.require("core")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
tf.config.experimental.set_visible_devices([], "GPU")
tf.config.experimental.set_visible_devices([], "TPU")
# Transfer guard will fail the program whenever that data between a host and
# a device is transferred implicitly. This often catches subtle bugs that
# cause slowdowns and memory fragmentation. Explicit transfers are done
# with jax.device_put and jax.device_get.
jax.config.update("jax_transfer_guard", "disallow")
# Fixes design flaw in jax.random that may cause unnecessary d2d comms.
jax.config.update("jax_threefry_partitionable", True)


class TrainState(ts):
    shard_idx: int = 0
    dataset_step: int = 0


def train_step(
    state: TrainState, tokens: jnp.ndarray, rng_key
) -> Tuple[jnp.ndarray, jnp.ndarray, TrainState, jnp.ndarray]:

    rng_key = jax.random.fold_in(rng_key, state.step)  # same key each grad accum step

    def loss_fn(params):
        X, Y = tokens[:, :-1], tokens[:, 1:]

        logits = state.apply_fn(
            X, params={"params": params}, dropout_rng=rng_key, train=True
        )[0]

        # compute loss in float32
        logits = logits.astype(jnp.float32)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y).mean()

        # palm style z-loss
        loss += 1e-4 * jax.scipy.special.logsumexp(logits, axis=-1).mean() ** 2

        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == Y)

        return loss, accuracy

    before_dtypes = jax.tree.map(lambda x: x.dtype, state)

    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    # new_state = new_state.replace(dataset_step=state.dataset_step + 1)

    check_dtypes(before_dtypes, jax.tree.map(lambda x: x.dtype, new_state))

    grad_norm = optax.global_norm(grads)

    return loss, acc, new_state, grad_norm


def eval_step_unreduced(state: TrainState, tokens: jnp.ndarray) -> jnp.ndarray:
    X, Y = tokens[:, :-1], tokens[:, 1:]
    logits = state.apply_fn(X, params={"params": state.params}, train=False)[0]
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


def get_default_config() -> TrainConfig:
    # use this file to set default values
    path = os.environ.get("LLM_CONFIG", os.path.join("config", "llama3.yaml"))
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
    # random.seed(config.seed)
    # np.random.seed(config.seed)
    tf.random.set_seed(config.seed)

    # wandb init
    if config.wandb is not None and jax.process_index() == 0:
        wandb.init(**asdict(config.wandb))
        wandb_config = asdict(config)
        wandb_config["jax_n_devices"] = jax.device_count()
        wandb_config["jax_n_processes"] = jax.process_count()
        wandb.config.update(wandb_config)

    block_size = config.model.block_size
    platform = jax.devices()[0].platform

    # ====== create device mesh ======
    write_note("creating 1D FSDP mesh")
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    devices_flat = device_mesh.flatten()
    mesh = Mesh(devices=device_mesh, axis_names="fsdp")

    # ====== optimizer ======
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

    def make_opt(reshaped_params_sharding=None):
        write_note(f"using {config.optimizer.type} optimizer")

        def param_decay_mask(params):
            """Only lets through kernel weights for weight decay."""
            non_kernels = flax.traverse_util.ModelParamTraversal(
                lambda p, _: "bias" in p
                or "norm" in p
                or "embedding" in p
                or "scale" in p
            )
            all_true = jax.tree.map(lambda _: True, params)
            out = non_kernels.update(lambda _: False, all_true)
            return out

        optimizer = []
        if config.optimizer.grad_clip > 0.0:
            optimizer.append(optax.clip_by_global_norm(config.optimizer.grad_clip))

        # decays to 0.01 at around 2000 steps
        update_prob_schedule = lambda n: jnp.maximum(jnp.exp(-0.002 * n), 0.01)

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
        elif config.optimizer.type in ["psgd", "psgd_affine", "affine"]:
            optimizer.append(
                affine(
                    lr_schedule,
                    update_prob_schedule,
                    b1=config.optimizer.betas[0],
                    nesterov=config.optimizer.nesterov,
                    weight_decay=config.optimizer.weight_decay,
                    mask=param_decay_mask,
                    max_size_triangular=config.optimizer.max_size_triangular,
                    max_skew_triangular=config.optimizer.max_skew_triangular,
                    precond_lr=config.optimizer.precond_lr,
                    precond_init_scale=config.optimizer.precond_init_scale,
                    mu_dtype=jnp.bfloat16,
                    precond_dtype=config.optimizer.preconditioner_dtype,
                    precision="bfloat16",
                    reshaped_params_sharding=reshaped_params_sharding,
                    best_effort_scan=True,
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

        optimizer = optax.chain(*optimizer)

        if config.optimizer.gradient_accumulation_steps > 1:
            optimizer = optax.MultiSteps(
                optimizer, config.optimizer.gradient_accumulation_steps
            )

        return optimizer

    # ====== train state ======
    write_note("creating and sharding train state")
    repl_sharding = NamedSharding(mesh, P())
    data_sharding = NamedSharding(mesh, P("fsdp"))

    rng = jax.random.PRNGKey(jax.device_put(config.seed, jax.devices("cpu")[0]))

    def init_train_state(key):
        """Initialize the train state."""
        # get easydel model config
        model_config = ed.AutoEasyDeLConfig.from_pretrained(
            config.model.huggingface_model_name
        )

        # override some settings
        model_config.use_cache = False
        model_config.use_scan_mlp = config.model.use_scan_mlp
        if hasattr(model_config, "scale_attn_by_inverse_layer_idx"):
            model_config.scale_attn_by_inverse_layer_idx = True
        if hasattr(model_config, "reorder_and_upcast_attn"):
            model_config.reorder_and_upcast_attn = True
        print(model_config)

        # get easydel flax module
        model_type: str = model_config.model_type
        _, module, _ = ed.get_modules_by_type(model_type)

        # create model and params
        model = module(
            config=model_config,
            dtype=config.compute_dtype,
            param_dtype=config.params_dtype,
            precision=jax.lax.Precision.DEFAULT,
            seed=config.seed,
            _do_init=False,
        )
        params = model.init_weights(key, input_shape=(1, block_size))
        params = otu.tree_cast(params, config.params_dtype)

        # delay optimizer creation to pass in preconditioner sharding
        apply_fn = partial(model.__call__, return_dict=False)
        train_state = TrainState(
            step=0, apply_fn=apply_fn, params=params, tx=None, opt_state=None
        )
        return train_state

    train_state_shapes = jax.eval_shape(init_train_state, rng)

    op = fsdp_sharding("fsdp", min_size_to_shard_mb=config.min_size_to_shard_mb)
    train_state_sharding = infer_sharding(params=train_state_shapes, mesh=mesh, op=op)

    if config.only_print_model:
        print("TRAIN STATE SHAPES:")
        pprint(
            jax.tree.map(lambda x: (x.shape, x.dtype), train_state_shapes),
            indent=2,
            width=120,
            compact=True,
        )
        print("TRAIN STATE SHARDING:")
        pprint(train_state_sharding, indent=2, width=120)
        raise KeyboardInterrupt("Only printing model")

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

    # PSGD reshapes params into matrices. Here we get sharding rules for them
    # similarly to how we shard params. We can pass this into PSGD for internal
    # sharding constraints, although it's not absolutely necessary.
    def get_reshaped_params_sharding(params):
        # returns tuples of (reshape_fn, unreshape_fn, shape)
        affine_reshapers = jax.tree.map(_shape_as_matrix, params)
        p_struct = jax.tree.structure(params)
        affine_reshapers = p_struct.flatten_up_to(affine_reshapers)
        matrix_shapes = [
            jax.ShapeDtypeStruct(r[2], jnp.float32) for r in affine_reshapers
        ]
        matrix_shapes = p_struct.unflatten(matrix_shapes)
        return infer_sharding(params=matrix_shapes, mesh=mesh, op=op)

    reshaped_params_sharding = get_reshaped_params_sharding(train_state.params)

    # remake optimizer with reshaped params sharding passed in
    # again, not strictly necessary, but could ensure things stay well-sharded
    optimizer = make_opt(reshaped_params_sharding=reshaped_params_sharding)

    # finish making train state (pass in optimizer and opt_state)
    train_state = train_state.replace(tx=optimizer, opt_state=opt_state)
    train_state_sharding = train_state_sharding.replace(
        tx=optimizer, opt_state=opt_state_sharding
    )

    rng = reshard(rng, repl_sharding)

    num_params = count_params(train_state.params)
    if jax.process_index() == 0:
        write_note("TRAIN STATE SHAPES AND DTYPES:")
        pprint(
            jax.tree.map(lambda x: (x.shape, x.dtype), train_state), indent=2, width=120
        )
        write_note("TRAIN STATE SHARDING:")
        pprint(train_state_sharding, indent=2, width=120)
        write_note(f"PARAMETER COUNT: {num_params:,}")

    # ==== restore train state ====
    if config.attempt_to_load_checkpoint:
        write_note(
            f"Attempting to load checkpoint from "
            f"{config.out_dir}/checkpoints/train_state if it exists."
        )
        write_note(
            "If loading checkpoint is unintended, set "
            "`attempt_to_load_checkpoint=False`."
        )
        train_state = checkpoints.restore_checkpoint(
            f"{config.out_dir}/checkpoints/train_state", train_state
        )
        # reshard for good measure
        train_state = jax.device_put(train_state, train_state_sharding)

    # grab start step from loaded train state
    step = jax.device_get(train_state.step).item()
    write_note(f"starting training at step {step}")

    # ====== jit functions ========
    # we specify in_shardings for sake of clarity, but they are inferred
    train_step_jit = jax.jit(
        train_step,
        # donate_argnums=(0,),
        in_shardings=(train_state_sharding, data_sharding, rng.sharding),
        out_shardings=(
            repl_sharding,
            repl_sharding,
            train_state_sharding,
            repl_sharding,
        ),
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

    @partial(
        jax.jit,
        # donate_argnums=(0,),
        in_shardings=(train_state_sharding,),
        out_shardings=train_state_sharding,
    )
    def step_minus_1(state):
        return state.replace(step=state.step - 1)

    @partial(
        jax.jit,
        in_shardings=(train_state_sharding,),
        out_shardings=train_state_sharding,
    )
    def advance_shard_idx_and_zero_dataset_step(state):
        """Used when current shard is exhausted and new one begins."""
        return state.replace(shard_idx=state.shard_idx + 1, dataset_step=0)

    # ===== datasets =====
    write_note("creating datasets")
    tokenizer_name = (
        config.model.tokenizer_name
        if config.model.tokenizer_name is not None
        else config.model.huggingface_model_name
    )
    if platform == "cpu":
        device_prefetch = 0
    elif platform == "gpu":
        device_prefetch = 2
    else:  # tpu
        device_prefetch = 1

    make_train_ds = partial(
        fineweb_edu_dataset,
        tokenizer_name=tokenizer_name,
        batch_size=config.batch_size,
        block_size=block_size,
        flat_devices=devices_flat,
        tf_prefetch=10,
        device_prefetch=device_prefetch,
    )

    shard_idx = 0
    train_ds = make_train_ds(shard_idx=shard_idx)

    # hellaswag has 4 seqs per example
    hellaswag_ds, hellaswag_len = prepare_hellaswag(
        tokenizer_name,
        max(config.batch_size // 4, jax.device_count()),
        block_size,
        devices_flat,
    )

    # ======= train ========
    orig_dtypes = jax.tree.map(lambda x: x.dtype, train_state)
    min_loss = float("inf")
    max_acc = 0.0
    max_hellaswag_acc = 0.0
    train_losses = []
    train_accs = []
    grad_norms = []
    start_time = None
    write_note("starting training")
    for step in range(step, config.train_steps):
        for accum_step in range(config.optimizer.gradient_accumulation_steps):
            try:
                tokens = next(train_ds)
            except StopIteration:
                print(
                    f"current dataset subshard exhausted on process "
                    f"{jax.process_index()}, loading next subshard"
                )
                del train_ds

                # delete huggingface datasets cache to save space
                if platform == "tpu":
                    hf_cache_dir = "/dev/shm/huggingface_cache"
                else:
                    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
                if os.path.exists(hf_cache_dir):
                    print(f"removing {hf_cache_dir} to save space")
                    try:
                        shutil.rmtree(hf_cache_dir)
                    except Exception as e:
                        print(f"Error removing {hf_cache_dir}: {e}")

                # start next subshard
                shard_idx += 1
                train_ds = make_train_ds(shard_idx=shard_idx)

                tokens = next(train_ds)

            loss, acc, train_state, g_norm = train_step_jit(train_state, tokens, rng)
            train_losses.append(jax.device_get(loss).item())
            train_accs.append(jax.device_get(acc).item())
            grad_norms.append(jax.device_get(g_norm).item())
            if accum_step < config.optimizer.gradient_accumulation_steps - 1:
                # only update step if we're on the last accumulation step
                train_state = step_minus_1(train_state)

        # log to wandb every 10 steps
        if config.wandb is not None and jax.process_index() == 0 and step % 10 == 0:
            train_state = jax.block_until_ready(train_state)
            end_time = time.time()

            train_loss = np.mean(train_losses)
            min_loss = min(min_loss, train_loss)
            train_acc = np.mean(train_accs)
            max_acc = max(max_acc, train_acc)
            grad_norm = np.mean(grad_norms)

            tokens_per_batch = (
                config.optimizer.gradient_accumulation_steps
                * config.batch_size
                * jax.device_count()
                * block_size
            )
            to_log = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "grad_norm": grad_norm,
                "lr": (
                    jax.device_get(get_lr(jax.device_put(step, repl_sharding)))
                    if callable(lr_schedule)
                    else lr_schedule
                ),
                "tokens": step * tokens_per_batch,
            }

            # performance metrics
            if start_time is not None:
                seconds_per_step = (end_time - start_time) / 10
                to_log["seconds_per_step"] = seconds_per_step
                to_log["tokens_per_second"] = tokens_per_batch / seconds_per_step

            wandb.log(to_log, step=step)
            wandb.summary["min_train_loss"] = min_loss
            wandb.summary["max_train_acc"] = max_acc

            train_losses = []
            train_accs = []
            grad_norms = []

            # print every 100 steps
            if step % 100 == 0:
                write_note(
                    f"step: {step}, loss: {train_loss:.4f}, accuracy: {train_acc:.4f}"
                )

                # double check dtypes are consistent
                check_dtypes(orig_dtypes, jax.tree.map(lambda x: x.dtype, train_state))

            start_time = time.time()

        # checkpoint
        if (
            config.checkpoint_interval > 0
            and step % config.checkpoint_interval == 0
            and config.keep_checkpoints > 0
            and step > 0
        ):
            if jax.process_index() == 0:
                checkpoints.save_checkpoint(
                    f"{config.out_dir}/checkpoints/train_state",
                    jax.device_get(train_state),
                    step,
                    keep=config.keep_checkpoints,
                    overwrite=True,
                    keep_every_n_steps=config.checkpoint_milestone,
                )
            if step % config.checkpoint_milestone == 0:
                write_note(f"saved milestone checkpoint at step {step}")
            else:
                write_note(f"saved checkpoint at step {step}")

            start_time = time.time()

        # eval hellaswag
        if step % config.hellaswag_eval_interval == 0 and step > 0:
            hs_accs = []
            for _ in range(10 if platform == "cpu" else hellaswag_len):
                hs_batch = next(hellaswag_ds)
                hs_acc = eval_hellaswag_jit(train_state, *hs_batch)
                hs_accs.append(jax.device_get(hs_acc).item())
            hellaswag_acc = np.mean(hs_accs)
            max_hellaswag_acc = max(max_hellaswag_acc, hellaswag_acc)

            if config.wandb is not None and jax.process_index() == 0:
                wandb.log({"hellaswag_acc": hellaswag_acc}, step=step)
                wandb.summary["max_hellaswag_acc"] = max_hellaswag_acc

            write_note(f"step: {step}, hellaswag_acc: {hellaswag_acc:.4f}")

            start_time = time.time()


if __name__ == "__main__":
    config = tyro.cli(TrainConfig, default=get_default_config(), use_underscores=True)
    main(config)
