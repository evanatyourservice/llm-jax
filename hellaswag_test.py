from typing import Callable
import numpy as np
from flax import struct
import flax.jax_utils
import transformers

from dataset import prepare_hellaswag
from train import eval_hellaswag, TrainConfig


@struct.dataclass
class State:
    params: dict
    apply_fn: Callable = struct.field(pytree_node=False)


def test_hellaswag():
    config = TrainConfig(batch_size=1)

    print("loading model")
    model = transformers.FlaxAutoModelForCausalLM.from_pretrained("gpt2")
    params = model.params

    state = State(params, model.__call__)
    state = flax.jax_utils.replicate(state)

    ds = prepare_hellaswag(config)

    accuracies = []
    for _ in range(100):
        batch = next(ds)
        accuracy = eval_hellaswag(state, *batch)[0].item()
        accuracies.append(accuracy)
        print(
            f"Avg accuracy over {len(accuracies)} steps: "
            f"{np.mean(accuracies) * 100:.2f}%"
        )

    assert np.mean(accuracies) * 100 > 28
    print("Test passed!")


if __name__ == "__main__":
    test_hellaswag()
