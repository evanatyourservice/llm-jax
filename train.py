import builtins
from functools import partial
from pprint import pprint
import shutil
import time
from typing import Callable, List, Tuple
from dataclasses import asdict
import os
import random
import numpy as np
import wandb

import jax
import jax.numpy as jnp
from jaxlib import xla_client
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import flax
from flax import struct
from flax.training.train_state import TrainState as ts
from flax.training import orbax_utils
import flax.traverse_util
import optax
import optax.tree_utils as otu
import orbax.checkpoint as ocp
import tensorflow as tf

from dataset import prepare_hellaswag, fineweb_edu_dataset, _fw_shard_names
from configs import TrainConfig
from optimizers.kron import kron, precond_update_prob_schedule
from optimizers.tearfree import optimizer as tearfree_opt
from optimizers.tearfree import shampoo, second_order
from optimizers.adam import adamw
from optimizers.schedule_free import schedule_free, schedule_free_eval_params
from sharding import infer_sharding, fsdp_sharding
from utils import check_dtypes, reshard, write_note, count_params, get_step
from model.mistral import Mistral


# hack to allow pickling of bfloat16 arrays
builtins.bfloat16 = xla_client.bfloat16

wandb.require("core")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
jax.config.update("jax_transfer_guard", "disallow")
jax.config.update("jax_threefry_partitionable", True)


class TrainState(ts):
    lr_fn: Callable = struct.field(pytree_node=False)


def main(config: TrainConfig):
    write_note(f"Number of JAX devices: {jax.device_count()}")
    write_note(f"Number of JAX processes: {jax.process_count()}")

    # set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)

    # wandb init
    if jax.process_index() == 0 and config.wandb.mode == "online":
        wandb.init(
            name=config.experiment_name,
            id=config.experiment_name,
            resume="allow",
            **asdict(config.wandb),
        )
        wandb_config = asdict(config)
        wandb_config["jax_n_devices"] = jax.device_count()
        wandb_config["jax_n_processes"] = jax.process_count()
        wandb.config.update(wandb_config)

    platform = jax.devices()[0].platform

    # ====== checkpointer ======
    with jax.transfer_guard("allow"):
        options = ocp.CheckpointManagerOptions(
            save_interval_steps=config.checkpoint_interval,
            max_to_keep=1,
            keep_period=config.checkpoint_milestone,  # milestones
            create=True,
            cleanup_tmp_directories=True,
        )
        async_checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
        async_checkpoint_manager = ocp.CheckpointManager(
            config.out_dir + "/" + config.experiment_name, async_checkpointer, options
        )

    # ====== create device mesh ======
    write_note("Creating 1D FSDP mesh")
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    devices_flat = device_mesh.flatten()
    mesh = Mesh(devices=device_mesh, axis_names="fsdp")

    # ====== optimizer ======
    write_note("Creating optimizer")

    if config.optimizer.flat_lr or config.optimizer.schedule_free:
        lr_schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    0.0, config.optimizer.learning_rate, config.optimizer.warmup_steps
                ),
                optax.constant_schedule(config.optimizer.learning_rate),
            ],
            boundaries=[config.optimizer.warmup_steps],
        )
    else:
        lr_schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    0.0, config.optimizer.learning_rate, config.optimizer.warmup_steps
                ),
                optax.linear_schedule(
                    config.optimizer.learning_rate,
                    config.optimizer.learning_rate * 0.01,
                    config.train_steps - config.optimizer.warmup_steps,
                ),
            ],
            boundaries=[config.optimizer.warmup_steps],
        )

    def make_opt(scanned_layers=None):
        write_note(f"Using {config.optimizer.type} optimizer")

        def param_decay_mask(params):
            """Only lets through kernel weights for weight decay."""
            all_true = jax.tree.map(lambda _: True, params)
            non_kernels = flax.traverse_util.ModelParamTraversal(
                lambda p, _: "bias" in p or "scale" in p or "embedding" in p
            )
            out = non_kernels.update(lambda _: False, all_true)
            return out

        optimizer = []

        # only apply grad clipping if not using schedule-free (see schedule-free paper)
        if config.optimizer.grad_clip > 0.0 and not config.optimizer.schedule_free:
            optimizer.append(optax.clip_by_global_norm(config.optimizer.grad_clip))

        if config.optimizer.type in ["adam", "adamw"]:
            optimizer.append(
                adamw(
                    lr_schedule,
                    0.0 if config.optimizer.schedule_free else config.optimizer.b1,
                    config.optimizer.b2,
                    config.optimizer.eps,
                    mu_dtype=jnp.float32,
                    weight_decay=config.optimizer.weight_decay,
                    mask=param_decay_mask,
                    nesterov=config.optimizer.nesterov,
                )
            )
            optimizer = optax.chain(*optimizer)
        elif config.optimizer.type in ["psgd", "psgd_kron", "kron"]:
            optimizer.append(
                kron(
                    lr_schedule,
                    b1=0.0 if config.optimizer.schedule_free else config.optimizer.b1,
                    weight_decay=config.optimizer.weight_decay,
                    weight_decay_mask=param_decay_mask,
                    preconditioner_update_probability=precond_update_prob_schedule(
                        min_prob=config.optimizer.preconditioner_update_probability
                    ),
                    max_size_triangular=config.optimizer.max_size_triangular,
                    max_skew_triangular=config.optimizer.max_skew_triangular,
                    preconditioner_init_scale = 0.1,  # lower than default for LLM
                    mu_dtype=jnp.float32,
                    precond_dtype=config.optimizer.preconditioner_dtype,
                    scanned_layers=scanned_layers,
                    lax_map_scanned_layers=config.optimizer.lax_map_scanned_layers,
                    lax_map_batch_size=config.optimizer.lax_map_batch_size,
                )
            )
            optimizer = optax.chain(*optimizer)
        elif config.optimizer.type in ["shampoo", "caspr"]:
            optimizer.append(
                tearfree_opt.tearfree(
                    lr_schedule,
                    tearfree_opt.TearfreeOptions(
                        second_order_options=second_order.Options(
                            shampoo_options=shampoo.Options(
                                use_CASPR_variant=config.optimizer.type == "caspr",
                                update_preconditioners_freq=25,
                            )
                        ),
                        momentum_options=tearfree_opt.momentum.Options(
                            weight_decay=config.optimizer.weight_decay,
                            momentum_decay=(
                                0.0
                                if config.optimizer.schedule_free
                                else config.optimizer.b1
                            ),
                            momentum_dtype="float32",
                            nesterov=config.optimizer.nesterov,
                        ),
                    ),
                )
            )
            optimizer = optax.chain(*optimizer)
        else:
            raise ValueError("Unknown optimizer type")

        if config.optimizer.schedule_free:
            optimizer = schedule_free(optimizer, lr_schedule, b1=config.optimizer.b1)

        return optimizer

    # ====== train state and sharding ======
    write_note("Creating and sharding train state")
    repl_sharding = NamedSharding(mesh, P())
    data_sharding = NamedSharding(mesh, P("fsdp"))

    def init_train_state(key):
        """Initialize the train state."""
        # make model
        model = Mistral(config.model, mesh, config.gradient_accumulation_steps > 1)

        # init params
        dummy_tokens = jnp.zeros(
            (config.batch_size, config.model.block_size), dtype=jnp.uint16
        )
        params = model.init(key, dummy_tokens)
        params = otu.tree_cast(params, config.params_dtype)

        # which layers are scanned
        if config.model.scan_layers:
            all_false = jax.tree.map(lambda _: False, params)
            scanned_layers = flax.traverse_util.ModelParamTraversal(
                lambda p, _: "scan" in p or "Scan" in p
            ).update(lambda _: True, all_false)
        else:
            scanned_layers = None

        # make optimizer
        optimizer = make_opt(scanned_layers=scanned_layers)

        # make train state
        train_state = TrainState(
            step=0,
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
            opt_state=optimizer.init(params),
            lr_fn=lr_schedule,
        )
        return train_state

    rng = jax.random.PRNGKey(
        jax.device_put(config.seed, jax.local_devices(backend="cpu")[0])
    )

    # get train state shapes and shardings
    train_state_shapes = jax.eval_shape(init_train_state, rng)

    op = fsdp_sharding("fsdp", min_size_to_shard_mb=config.min_size_to_shard_mb)
    train_state_sharding, _ = infer_sharding(
        params=train_state_shapes, mesh=mesh, op=op
    )

    # make train state
    rng_init = reshard(rng, repl_sharding)
    train_state = jax.jit(init_train_state, out_shardings=train_state_sharding)(
        rng_init
    )

    # load checkpoint
    with jax.transfer_guard("allow"):
        if (
            config.attempt_to_load_checkpoint
            and async_checkpoint_manager.latest_step() is not None
        ):
            write_note(
                f"LOADING CHECKPOINT from {config.out_dir}/{config.experiment_name}"
            )
            restore_args = orbax_utils.restore_args_from_target(train_state)
            train_state = async_checkpoint_manager.restore(
                async_checkpoint_manager.latest_step(),
                items=train_state,
                restore_kwargs={"restore_args": restore_args},
            )

    num_params = count_params(train_state.params)
    if jax.process_index() == 0:
        print("TRAIN STATE SHAPES AND DTYPES:")
        pprint(
            jax.tree.map(lambda x: (x.shape, x.dtype), train_state),
            indent=2,
            width=150,
            compact=True,
        )
        print("TRAIN STATE SHARDING:")
        pprint(train_state_sharding, indent=2, width=150, compact=True)
        print(f"PARAMETER COUNT: {num_params:,}")
        if config.only_print_model:
            raise KeyboardInterrupt("Only printing model")

    # ====== datasets ======
    write_note("Creating datasets")

    curr_step = get_step(train_state)

    shard_idx = 0
    if jax.process_count() == 1:
        # stream fineweb-edu regularly
        ds_name = None
    else:
        # use separate shards per process
        # we just restart this with a new random shuffle if restarting
        process_shard = _fw_shard_names[jax.process_index() :: jax.process_count()]
        # shuffle using current step so first steps are deterministic,
        # after that it doesn't matter as much if restarting training
        rng = np.random.RandomState(curr_step + jax.process_index())
        rng.shuffle(process_shard)
        ds_name = process_shard[shard_idx % len(process_shard)]

    make_train_ds = partial(
        fineweb_edu_dataset,
        batch_size=config.batch_size,
        block_size=config.model.block_size + 1,
        flat_devices=devices_flat,
        tf_prefetch=10,
        device_prefetch=2 if platform == "gpu" else 0,
    )

    train_ds = make_train_ds(fineweb_edu_name=ds_name)

    # hellaswag has 4 seqs per example
    hs_batch_size = max(config.batch_size // 4, jax.device_count())
    hellaswag_ds = prepare_hellaswag(
        hs_batch_size, config.model.block_size + 1, devices_flat, tf_prefetch=4
    )

    # ====== train and eval steps ======
    def train_step(
        state: TrainState, batch: List[jnp.ndarray]
    ) -> Tuple[jnp.ndarray, TrainState, jnp.ndarray, jnp.ndarray]:

        def loss_fn(params, tokens):
            logits = state.apply_fn(
                otu.tree_cast(params, config.compute_dtype), tokens[:, :-1]
            )
            assert logits.dtype == config.compute_dtype

            logits = logits.astype(jnp.float32)

            targets = tokens[:, 1:]

            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)

            bos_token_id = 1  # mistral tokenizer
            mask = targets != bos_token_id

            loss = jnp.sum(loss * mask) / jnp.sum(mask)

            return loss

        before_dtypes = jax.tree.map(lambda x: x.dtype, state)

        if config.gradient_accumulation_steps > 1:

            def scan_body(carry, tokens):
                step, loss_sum, accum_grads = carry

                loss, grads = jax.value_and_grad(loss_fn)(state.params, tokens)
                grads = jax.lax.with_sharding_constraint(
                    grads, train_state_sharding.params
                )

                accum_grads = jax.lax.cond(
                    step == 0,
                    lambda: grads,
                    lambda: jax.tree.map(
                        lambda ag, g: ag + (g - ag) / (step + 1), accum_grads, grads
                    ),
                )
                accum_grads = jax.lax.with_sharding_constraint(
                    accum_grads, train_state_sharding.params
                )

                carry = (step + 1, loss_sum + loss, accum_grads)
                return carry, None

            # init state is (step, loss_sum, accum_grads)
            init_grads = jax.tree.map(jnp.zeros_like, state.params)
            init_grads = jax.lax.with_sharding_constraint(
                init_grads, train_state_sharding.params
            )
            init_state = (0, 0.0, init_grads)

            carry, _ = jax.lax.scan(scan_body, init_state, jnp.stack(batch))
            _, loss_sum, grads = carry
            loss = loss_sum / config.gradient_accumulation_steps
        else:
            loss, grads = jax.value_and_grad(loss_fn)(state.params, batch[0])
            grads = jax.lax.with_sharding_constraint(grads, train_state_sharding.params)

        new_state = state.apply_gradients(grads=grads)

        check_dtypes(before_dtypes, jax.tree.map(lambda x: x.dtype, new_state))

        grad_norm = optax.global_norm(grads)
        lr = state.lr_fn(state.step)

        return loss, new_state, grad_norm, lr

    def hs_eval_step_unreduced(
        state: TrainState,
        tokens: jnp.ndarray,
        begin_lens: jnp.ndarray,
        seq_lens: jnp.ndarray,
    ) -> jnp.ndarray:
        params_in = state.params
        if config.optimizer.schedule_free:
            params_in = schedule_free_eval_params(state.opt_state, state.params)

        logits = state.apply_fn(
            otu.tree_cast(params_in, config.compute_dtype), tokens[:, :-1]
        )
        assert logits.dtype == config.compute_dtype

        logits = logits.astype(jnp.float32)

        targets = tokens[:, 1:]

        losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)

        @jax.vmap
        def unreduced_losses(loss, begin_len, seq_len):
            seq_range = jnp.arange(len(loss))
            seq_mask = jnp.logical_and(
                seq_range < seq_len - 1, seq_range >= begin_len - 1
            ).astype(jnp.bool)
            loss = loss * seq_mask
            return jnp.sum(loss) / jnp.sum(seq_mask)

        losses = unreduced_losses(losses, begin_lens, seq_lens)
        return losses

    def eval_hellaswag(state: TrainState, data, begin_lens, seq_lens, labels):
        """Evaluate the hellaswag dataset."""
        # data comes in shape (b, 4, block_size + 1)
        # begin and seq lens come in shape (b, 4)
        # labels come in shape (b,)
        bs_in = data.shape[0]
        data = jnp.reshape(data, (-1, data.shape[-1]))
        begin_lens = jnp.reshape(begin_lens, (-1,))
        seq_lens = jnp.reshape(seq_lens, (-1,))
        losses = hs_eval_step_unreduced(state, data, begin_lens, seq_lens)
        choices = jnp.argmin(jnp.reshape(losses, (bs_in, 4)), axis=-1)
        correct = jnp.sum(choices == labels)
        accuracy = correct / bs_in
        return accuracy

    # ====== jit functions ========
    # we specify in_shardings for sake of clarity, but they are inferred
    train_step_jit = jax.jit(
        train_step,
        donate_argnums=(0,),
        in_shardings=(
            train_state_sharding,
            [data_sharding for _ in range(config.gradient_accumulation_steps)],
        ),
        out_shardings=(
            repl_sharding,
            train_state_sharding,
            repl_sharding,
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
            data_sharding,
        ),
        out_shardings=repl_sharding,
    )

    # ======= train ========
    write_note(f"Starting training at step {curr_step}")

    orig_dtypes = jax.tree.map(lambda x: x.dtype, train_state)
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    min_loss = float("inf")
    max_hellaswag_acc = 0.0
    train_losses = []
    grad_norms = []
    start_time = None  # skip first loop for compile
    while curr_step < config.train_steps:
        with jax.profiler.StepTraceAnnotation("train", step_num=curr_step):
            batch = []
            for _ in range(config.gradient_accumulation_steps):
                try:
                    batch.append(next(train_ds))
                except StopIteration:
                    print(
                        f"Current dataset subshard exhausted on process "
                        f"{jax.process_index()}, loading next subshard"
                    )
                    del train_ds

                    # delete huggingface datasets cache to save space
                    if platform == "tpu":
                        hf_cache_dir = "/dev/shm/huggingface_cache"
                    else:
                        hf_cache_dir = os.path.expanduser(
                            "~/.cache/huggingface/datasets"
                        )
                    if os.path.exists(hf_cache_dir):
                        print(f"Removing {hf_cache_dir} to save space")
                        try:
                            shutil.rmtree(hf_cache_dir)
                        except Exception as e:
                            print(f"Error removing {hf_cache_dir}: {e}")

                    # start next subshard or restart whole dataset
                    shard_idx += 1
                    if jax.process_count() == 1:
                        ds_name = None
                    else:
                        ds_name = process_shard[shard_idx % len(process_shard)]
                    train_ds = make_train_ds(fineweb_edu_name=ds_name)

                    batch.append(next(train_ds))

            if curr_step == 0:
                write_note(
                    f"Batch info: {jax.tree.map(lambda x: (x.shape, x.dtype), batch)}"
                )

            loss, train_state, g_norm, lr = train_step_jit(train_state, batch)
            train_losses.append(jax.device_get(loss).item())
            grad_norms.append(jax.device_get(g_norm).item())

        curr_step = get_step(train_state)

        # save checkpoint
        with jax.transfer_guard("allow"):
            if curr_step > 0:
                async_checkpoint_manager.save(curr_step, train_state)

        if config.profile and curr_step == 10:
            jax.profiler.start_trace(config.out_dir + "/profile")
        if config.profile and curr_step == 10 + config.n_profile_steps:
            jax.profiler.stop_trace()

        # logging
        if curr_step % 10 == 0 and curr_step > 0:
            train_loss = np.mean(train_losses)
            min_loss = min(min_loss, train_loss)
            grad_norm = np.mean(grad_norms)
            curr_lr = jax.device_get(lr).item()
            curr_tokens = (
                (curr_step + 1) * effective_batch_size * config.model.block_size
            )
            to_log = {
                "train_loss": train_loss,
                "grad_norm": grad_norm,
                "lr": curr_lr,
                "tokens": curr_tokens,
            }

            if curr_step % 100 == 0:
                # add performance metrics
                jax.block_until_ready(train_state.params)
                if start_time is not None:
                    end_time = time.time()

                    seconds_per_step = (end_time - start_time) / 100
                    tokens_per_second = (
                        effective_batch_size
                        * config.model.block_size
                        / seconds_per_step
                    )
                    to_log["seconds_per_step"] = seconds_per_step
                    to_log["tokens_per_second"] = tokens_per_second

                write_note(
                    f"step: {curr_step}, loss: {train_loss:.4f}, "
                    f"grad_norm: {grad_norm:.4f}, "
                    f"lr: {curr_lr:.4f}, tokens: {curr_tokens:.4f}"
                )

                # eval hellaswag
                if (
                    curr_step % config.hellaswag_eval_interval == 0
                    and config.model.block_size >= 1024
                ):
                    hs_accs = []
                    for _ in range(10 if platform == "cpu" else 10042 // hs_batch_size):
                        hs_batch = next(hellaswag_ds)
                        hs_acc = eval_hellaswag_jit(train_state, *hs_batch)
                        hs_accs.append(jax.device_get(hs_acc).item())
                    hellaswag_acc = np.mean(hs_accs)
                    max_hellaswag_acc = max(max_hellaswag_acc, hellaswag_acc)

                    to_log["hellaswag_acc"] = hellaswag_acc
                    if wandb.run is not None and jax.process_index() == 0:
                        wandb.summary["max_hellaswag_acc"] = max_hellaswag_acc

                    write_note(f"step: {curr_step}, hellaswag_acc: {hellaswag_acc:.4f}")

                    jax.block_until_ready(hs_acc)

                # check train state dtypes are consistent
                check_dtypes(orig_dtypes, jax.tree.map(lambda x: x.dtype, train_state))

                start_time = time.time()

            # log to wandb
            if wandb.run is not None and jax.process_index() == 0:
                wandb.log(to_log, step=curr_step)
                wandb.summary["min_train_loss"] = min_loss

            # reset metrics lists
            train_losses = []
            grad_norms = []

    with jax.transfer_guard("allow"):
        async_checkpoint_manager.save(curr_step, train_state)
        async_checkpoint_manager.wait_until_finished()

    wandb.finish()
