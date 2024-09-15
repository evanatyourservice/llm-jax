from typing import Callable
import numpy as np
from flax import struct
import transformers
import jax
import jax.numpy as jnp
import optax
from transformers import AutoTokenizer
from tqdm import tqdm
import tensorflow as tf
import json


@struct.dataclass
class State:
    params: dict
    apply_fn: Callable = struct.field(pytree_node=False)


def prepare_hellaswag(
    batch_size: int,
    block_size: int,
    tf_prefetch: int = 2,
):
    """Read file and tokenize the hellaswag dataset."""

    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2", trust_remote_code=True, use_fast=True
    )

    all_data = []
    all_beginning_lengths = []
    all_seq_lengths = []
    all_labels = []
    with open("data/hellaswag_val.jsonl", "r") as f:
        # iterate over lines and tokenize
        for line in tqdm(f, total=10042):
            item = json.loads(line)

            context = item["ctx"]
            endings = item["endings"]
            correct_end = item["label"]

            beginning_length = len(tokenizer(context)["input_ids"])

            data_to_concat = []
            beginning_lengths_to_concat = []
            seq_lengths_to_concat = []
            for ending in endings:
                output = tokenizer(context + " " + ending)["input_ids"]
                output_len = len(output)

                # pad to block_size
                if output_len < block_size:
                    output = output + [tokenizer.eos_token_id] * (
                        block_size - output_len
                    )
                # max length is block_size
                output = output[:block_size]

                data_to_concat.append(output)
                beginning_lengths_to_concat.append(beginning_length)
                seq_lengths_to_concat.append(output_len)

            all_data.append(np.array(data_to_concat, dtype=np.uint16))
            all_beginning_lengths.append(
                np.array(beginning_lengths_to_concat, dtype=np.int32)
            )
            all_seq_lengths.append(
                np.array(seq_lengths_to_concat, dtype=np.int32)
            )
            all_labels.append(int(correct_end))

    all_data = np.array(all_data, dtype=np.uint16)  # (10042, 4, seq_len)
    all_beginning_lengths = np.array(
        all_beginning_lengths, dtype=np.int32
    )  # (10042, 4)
    all_seq_lengths = np.array(all_seq_lengths, dtype=np.int32)  # (10042, 4)
    all_labels = np.array(all_labels, dtype=np.int32)  # (10042,)

    ds = tf.data.Dataset.from_tensor_slices(
        (all_data, all_beginning_lengths, all_seq_lengths, all_labels)
    )
    ds = ds.repeat()

    ds = ds.batch(
        batch_size // jax.process_count(),
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.prefetch(tf_prefetch)
    ds = ds.as_numpy_iterator()

    return ds


def hs_eval_step_unreduced(
    state,
    tokens: jnp.ndarray,
    begin_lens: jnp.ndarray,
    seq_lens: jnp.ndarray,
) -> jnp.ndarray:
    logits = state.apply_fn(tokens[:, :-1], params=state.params, train=False)["logits"]

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


@jax.jit
def eval_hellaswag(state, data, begin_lens, seq_lens, labels):
    """Evaluate the hellaswag dataset."""
    # data comes in shape (b, 4, block_size + 1)
    # seq lens come in shape (b, 4)
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


def test_hellaswag():
    print("loading model")
    model = transformers.FlaxAutoModelForCausalLM.from_pretrained("gpt2")
    params = model.params

    state = State(params, model.__call__)

    ds = prepare_hellaswag(2, 1024)

    accuracies = []
    for _ in range(10000 // 2):
        batch = next(ds)
        accuracy = eval_hellaswag(state, *batch)
        accuracies.append(accuracy)
        print(
            f"Avg accuracy over {len(accuracies) * 2} steps: "
            f"{np.mean(accuracies) * 100:.2f}%"
        )

    assert np.mean(accuracies) * 100 > 28
    print("Test passed!")


if __name__ == "__main__":
    test_hellaswag()
