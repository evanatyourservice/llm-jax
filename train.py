import builtins
from functools import partial
from pprint import pprint
import shutil
import time
from typing import Callable, Tuple
from dataclasses import asdict
import os
import flax.traverse_util
import numpy as np
import wandb
import tyro

import jax
import jax.numpy as jnp
from jaxlib import xla_client
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import flax
from flax import struct, linen as nn
from flax.training.train_state import TrainState as ts
import optax
import optax.tree_utils as otu
from transformers import AutoTokenizer

from dataset import prepare_hellaswag, fineweb_edu_dataset
from configs import TrainConfig
from optimizers.psgd_affine import affine, _shape_as_matrix
from optimizers.tearfree import optimizer as tearfree_opt
from optimizers.tearfree import shampoo, second_order
from sharding import infer_sharding, fsdp_sharding
from utils import check_dtypes, reshard, write_note, count_params, get_default_config
from model import GPT


# hack to allow pickling of bfloat16 arrays
builtins.bfloat16 = xla_client.bfloat16

wandb.require("core")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"
# Transfer guard will fail the program whenever that data between a host and
# a device is transferred implicitly. This often catches subtle bugs that
# cause slowdowns and memory fragmentation. Explicit transfers are done
# with jax.device_put and jax.device_get.
jax.config.update("jax_transfer_guard", "disallow")
# Fixes design flaw in jax.random that may cause unnecessary d2d comms.
jax.config.update("jax_threefry_partitionable", True)


class TrainState(ts):
    lr_fn: Callable = struct.field(pytree_node=False)


def get_attention_mask_and_positions(
    seq_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Builds the position and attention mask vectors from the given tokens."""
    positions = jnp.cumsum(seq_mask, axis=-1)
    # Subtract one for all positions from the first valid one as they are
    # 0-indexed
    current_token_position = positions - (positions >= 1)

    seq_len = seq_mask.shape[-1]
    attn_mask = seq_mask[..., None, :]
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    # Prefixes can be attended by all tokens
    attn_mask *= causal_mask[None, ...]

    return current_token_position, attn_mask


def main(config: TrainConfig):
    write_note(f"Number of JAX devices: {jax.device_count()}")
    write_note(f"Number of JAX processes: {jax.process_count()}")

    # set seeds
    # random.seed(config.seed)
    # np.random.seed(config.seed)

    # wandb init
    if config.wandb is not None and jax.process_index() == 0:
        wandb.init(**asdict(config.wandb))
        wandb_config = asdict(config)
        wandb_config["jax_n_devices"] = jax.device_count()
        wandb_config["jax_n_processes"] = jax.process_count()
        wandb.config.update(wandb_config)

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

    def make_opt(reshaped_params_sharding=None, scanned_arrays=None):
        write_note(f"using {config.optimizer.type} optimizer")

        optimizer = []
        if config.optimizer.grad_clip > 0.0:
            optimizer.append(optax.clip_by_global_norm(config.optimizer.grad_clip))

        # decays to 0.01 by around 2000 steps
        update_prob_schedule = lambda n: jnp.maximum(jnp.exp(-0.002 * n), 0.01)
        # opposite of update_prob_schedule from 0.01 to 0.1
        # precond_lr_schedule = lambda n: (-0.9 * jnp.exp(-0.002 * n) + 1.0) / 10

        if config.optimizer.type in ["adam", "adamw"]:
            optimizer.append(
                optax.adamw(
                    lr_schedule,
                    *config.optimizer.betas,
                    weight_decay=config.optimizer.weight_decay,
                    mask=None,
                    mu_dtype=jnp.bfloat16,
                )
            )
        elif config.optimizer.type in ["psgd", "psgd_affine", "affine"]:
            optimizer.append(
                affine(
                    lr_schedule,
                    preconditioner_update_probability=update_prob_schedule,
                    b1=config.optimizer.betas[0],
                    nesterov=config.optimizer.nesterov,
                    weight_decay=config.optimizer.weight_decay,
                    mask=None,
                    max_size_triangular=config.optimizer.max_size_triangular,
                    max_skew_triangular=config.optimizer.max_skew_triangular,
                    precond_lr=config.optimizer.precond_lr,
                    precond_init_scale=config.optimizer.precond_init_scale,
                    mu_dtype=jnp.bfloat16,
                    precond_dtype=config.optimizer.preconditioner_dtype,
                    precision="bfloat16",
                    reshaped_params_sharding=reshaped_params_sharding,
                    scanned_arrays=scanned_arrays,
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

    # ====== train state and sharding ======
    write_note("creating and sharding train state")
    repl_sharding = NamedSharding(mesh, P())
    data_sharding = NamedSharding(mesh, P("fsdp"))

    def init_train_state(key):
        """Initialize the train state."""
        model = GPT(config.model)

        dummy_tokens = jnp.zeros((1, config.model.block_size), dtype=jnp.uint16)
        dummy_seq_mask = jnp.ones((1, config.model.block_size), dtype=jnp.bool)

        positions, attention_mask = get_attention_mask_and_positions(dummy_seq_mask)

        params = model.init(key, dummy_tokens, positions, attention_mask)
        params = otu.tree_cast(params, config.params_dtype)

        # delay optimizer creation to pass in preconditioner sharding
        train_state = TrainState(
            step=0,
            apply_fn=model.apply,
            params=params,
            tx=None,
            opt_state=None,
            lr_fn=lr_schedule,
        )
        return train_state

    rng = jax.random.PRNGKey(jax.device_put(config.seed, jax.devices("cpu")[0]))

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

    # get pytree of how many times we scan each array in params
    scanned_arrays = jax.tree.map(lambda _: 0, train_state.params)
    scanned_arrays = flax.traverse_util.ModelParamTraversal(
        lambda path, _: "scan" in path
    ).update(lambda _: 1, scanned_arrays)

    # make optimizer and get its shardings, init psgd with scanned arrays
    optimizer = make_opt(scanned_arrays=scanned_arrays)

    opt_state_shapes = jax.eval_shape(optimizer.init, train_state.params)
    opt_state_sharding, _ = infer_sharding(params=opt_state_shapes, mesh=mesh, op=op)

    opt_state = jax.jit(optimizer.init, out_shardings=opt_state_sharding)(
        train_state.params
    )

    # PSGD reshapes params into matrices. Here we get sharding rules for them
    # similarly to params. We can pass this into PSGD for internal sharding
    # constraints, although it's not absolutely necessary. If all params are
    # already matrices, then this is unnecessary.
    def get_reshaped_params_shapes(params, scanned_layers):
        """Get the shapes of params after PSGD reshapes."""
        # TODO move to psgd file and print which params are being reshaped in a nice way
        affine_reshapers = jax.tree.map(
            _shape_as_matrix, params, scanned_layers
        )  # returns tuples of (reshape_fn, unreshape_fn, shape)
        p_struct = jax.tree.structure(params)
        affine_reshapers = p_struct.flatten_up_to(affine_reshapers)
        matrix_shapes = [
            jax.ShapeDtypeStruct(r[2], jnp.float32) for r in affine_reshapers
        ]
        return p_struct.unflatten(matrix_shapes)

    reshaped_params_shapes = get_reshaped_params_shapes(
        train_state.params, scanned_arrays
    )
    reshaped_params_sharding, _ = infer_sharding(
        params=reshaped_params_shapes, mesh=mesh, op=op
    )

    # remake optimizer with reshaped params sharding and scanned layers passed in
    optimizer = make_opt(
        reshaped_params_sharding=reshaped_params_sharding, scanned_arrays=scanned_arrays
    )

    # finish making train state (pass in optimizer and opt_state)
    train_state = train_state.replace(tx=optimizer, opt_state=opt_state)
    train_state_sharding = train_state_sharding.replace(
        tx=optimizer, opt_state=opt_state_sharding
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
    write_note("creating datasets")

    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2", trust_remote_code=True, use_fast=True
    )
    pad_id = tokenizer.pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.eos_token_id
    write_note(f"pad_id: {pad_id}")

    make_train_ds = partial(
        fineweb_edu_dataset,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        block_size=config.model.block_size,
        flat_devices=devices_flat,
        tf_prefetch=10,
        device_prefetch=2 if platform == "gpu" else 0,
    )

    shard_idx = 0
    train_ds = make_train_ds(shard_idx=shard_idx)

    # hellaswag has 4 seqs per example
    hellaswag_ds, hellaswag_len = prepare_hellaswag(
        tokenizer,
        max(config.batch_size // 4, jax.device_count()),
        config.model.block_size,
        devices_flat,
        tf_prefetch=4,
    )

    # ====== train and eval steps ======
    def train_step(
        state: TrainState,
        tokens: jnp.ndarray,
        seq_lens: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, TrainState, jnp.ndarray, jnp.ndarray]:

        def loss_fn(params):
            seq_mask = jnp.arange(tokens.shape[1])[None, :] < seq_lens[:, None]

            positions, attention_mask = get_attention_mask_and_positions(seq_mask)

            logits = state.apply_fn(
                otu.tree_cast(params, config.compute_dtype),
                tokens,
                positions,
                attention_mask,
            )
            assert logits.dtype == config.compute_dtype

            # Exclude the last step as it does not appear in the targets.
            logits = logits[:, :-1]

            # Similarly, the first token cannot be predicteds.
            target_tokens = tokens[:, 1:]
            target_mask = seq_mask[:, 1:]

            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, target_tokens
            )

            # Don't update on unwanted tokens (all tokens are wanted in this case)
            loss = loss * target_mask.astype(loss.dtype)

            # Normalization factor
            norm_factor = jnp.sum(target_mask)
            norm_factor = jnp.reciprocal(jnp.where(norm_factor == 0, 1, norm_factor))

            # Calculate the loss
            loss = jnp.sum(loss) * norm_factor

            # Palm style z-loss
            zloss = jax.scipy.special.logsumexp(logits, axis=-1)
            zloss = zloss * target_mask.astype(zloss.dtype)
            zloss = jnp.sum(zloss) * norm_factor
            loss += 1e-4 * zloss**2

            return loss

        before_dtypes = jax.tree.map(lambda x: x.dtype, state)

        loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(state.params)
        new_state = state.apply_gradients(grads=grads)

        check_dtypes(before_dtypes, jax.tree.map(lambda x: x.dtype, new_state))

        grad_norm = optax.global_norm(grads)
        lr = state.lr_fn(state.step)

        return loss, new_state, grad_norm, lr

    def eval_step_unreduced(
        state: TrainState, tokens: jnp.ndarray, seq_lens: jnp.ndarray
    ) -> jnp.ndarray:
        seq_mask = jnp.arange(tokens.shape[1])[None, :] < seq_lens[:, None]

        # Get attention mask and positions
        positions, attention_mask = get_attention_mask_and_positions(seq_mask)

        # Forward pass
        logits = state.apply_fn(
            otu.tree_cast(state.params, config.compute_dtype),
            tokens,
            positions,
            attention_mask,
        )
        assert logits.dtype == config.compute_dtype

        # Exclude the last step as it does not appear in the targets
        logits = logits[:, :-1]

        # Similarly, the first token cannot be predicted
        target_tokens = tokens[:, 1:]
        target_mask = seq_mask[:, 1:]

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_tokens)

        # Don't use loss past the sequence length
        loss = loss * target_mask.astype(loss.dtype)

        @jax.vmap
        def unreduced_losses(loss, mask):
            norm_factor = jnp.sum(mask)
            norm_factor = jnp.reciprocal(jnp.where(norm_factor == 0, 1, norm_factor))
            return jnp.sum(loss) * norm_factor

        return unreduced_losses(loss, target_mask)  # [b * 4]

    def eval_hellaswag(state: TrainState, data, seq_lens, labels):
        """Evaluate the hellaswag dataset."""
        # data comes in shape (b, 4, block_size)
        # masks comes in shape (b, 4, block_size)
        # labels comes in shape (b,)
        bs_in = data.shape[0]
        data = jnp.reshape(data, (-1, data.shape[-1]))
        seq_lens = jnp.reshape(seq_lens, (-1,))
        losses = eval_step_unreduced(state, data, seq_lens)
        choices = jnp.argmin(jnp.reshape(losses, (bs_in, 4)), axis=-1)
        correct = jnp.sum(choices == labels)
        accuracy = correct / bs_in
        return accuracy

    # ====== jit functions ========
    # we specify in_shardings for sake of clarity, but they are inferred
    train_step_jit = jax.jit(
        train_step,
        donate_argnums=(0,),
        in_shardings=(train_state_sharding, data_sharding, data_sharding),
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
        ),
        out_shardings=repl_sharding,
    )

    @partial(
        jax.jit,
        donate_argnums=(0,),
        in_shardings=(train_state_sharding,),
        out_shardings=train_state_sharding,
    )
    def step_minus_1(state):
        return state.replace(step=state.step - 1)

    # ======= train ========
    # grab start step from train state
    step = jax.device_get(train_state.step).item()
    write_note(f"starting training at step {step}")

    orig_dtypes = jax.tree.map(lambda x: x.dtype, train_state)
    min_loss = float("inf")
    max_hellaswag_acc = 0.0
    train_losses = []
    grad_norms = []
    start_time = None  # skip first loop for compile
    write_note("starting training")
    for step in range(step, config.train_steps):
        for accum_step in range(config.optimizer.gradient_accumulation_steps):
            try:
                tokens, seq_lens = next(train_ds)
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

                tokens, seq_lens = next(train_ds)

            loss, train_state, g_norm, lr = train_step_jit(train_state, tokens, seq_lens)
            train_losses.append(jax.device_get(loss).item())
            grad_norms.append(jax.device_get(g_norm).item())
            if accum_step < config.optimizer.gradient_accumulation_steps - 1:
                # only update step if we're on the last accumulation step
                train_state = step_minus_1(train_state)

        # log to wandb every 10 steps
        if config.wandb is not None and jax.process_index() == 0 and step % 10 == 0:
            jax.block_until_ready(train_state.params)
            end_time = time.time()

            train_loss = np.mean(train_losses)
            min_loss = min(min_loss, train_loss)
            grad_norm = np.mean(grad_norms)

            tokens_per_batch = (
                config.optimizer.gradient_accumulation_steps
                * config.batch_size
                * jax.device_count()
                * config.model.block_size
            )
            to_log = {
                "train_loss": train_loss,
                "grad_norm": grad_norm,
                "lr": jax.device_get(lr).item(),
                "tokens": step * tokens_per_batch,
            }

            # performance metrics
            if start_time is not None:
                seconds_per_step = (end_time - start_time) / 10
                to_log["seconds_per_step"] = seconds_per_step
                to_log["tokens_per_second"] = tokens_per_batch / seconds_per_step

            wandb.log(to_log, step=step)
            wandb.summary["min_train_loss"] = min_loss

            train_losses = []
            grad_norms = []

            # print every 100 steps
            if step % 100 == 0:
                write_note(f"step: {step}, loss: {train_loss:.4f}")

            start_time = time.time()

        # eval hellaswag
        if step % config.hellaswag_eval_interval == 0:
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

            # double check dtypes are consistent
            check_dtypes(orig_dtypes, jax.tree.map(lambda x: x.dtype, train_state))

            start_time = time.time()


if __name__ == "__main__":
    config = tyro.cli(TrainConfig, default=get_default_config(), use_underscores=True)
    main(config)
