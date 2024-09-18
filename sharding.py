# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Big vision sharding utilities.

Modified for simple fsdp sharding."""
import numpy as np

import jax
from jax.sharding import NamedSharding, PartitionSpec as P
import flax.linen as nn

from utils import tree_flatten_with_names, write_note


def infer_sharding(params, mesh, op):
    """Infer sharding spec for the given parameters.

    Return a sharding tree and a spec tree.
    """
    x_with_names, tree_def = tree_flatten_with_names(params)
    names = tree_def.unflatten(list(zip(*x_with_names))[0])

    specs = jax.tree.map(lambda x: (None,) * x.ndim, params)

    specs = jax.tree.map(
        lambda x, name, spec: op(spec, mesh, name, x),
        params,
        names,
        specs,
        is_leaf=lambda v: isinstance(v, nn.Partitioned),
    )

    # Two-level tree_map to prevent it from doing traversal inside the spec.
    specs = jax.tree.map(lambda _, spec: P(*spec), nn.unbox(params), specs)
    sharding = jax.tree.map(lambda spec: NamedSharding(mesh, spec), specs)
    return sharding, specs


def fsdp_sharding(axis, min_size_to_shard_mb=1, psgd_reshaped: bool = False):
    """FSDP sharding rule.

    Shards the largest dimension that is not sharded already and is divisible
    by the total device count.

    Args:
      axis: mesh axis name for FSDP, or a collection of names.
      min_size_to_shard_mb: minimal tensor size to bother with sharding.

    Returns:
      A function that updates the sharding spec.
    """
    axis = axis if isinstance(axis, str) else tuple(axis)
    axis_tuple = axis if isinstance(axis, tuple) else (axis,)

    def _update_spec(cur_spec, mesh, name, x):
        axis_size = np.prod([mesh.shape[a] for a in axis_tuple])
        shape = x.shape

        # Partitioning rules
        # indexed backwards from last dim for scan leading dims friendliness
        if (
            np.prod(shape) * x.dtype.itemsize >= min_size_to_shard_mb * (2**20)
            and len(shape) > 1
        ):
            new_sharding = [None for _ in shape]
            if "scale" in name:
                pass
            elif any(
                s in name
                for s in [
                    "preconditioner",
                    "out_kernel",
                    "gate_kernel",
                    "up_kernel",
                    "embedding",
                ]
            ):
                # shard these on last dim, including PSGD preconditioners so expanding
                # axes stay sharded while applying preconditioner
                if shape[-1] % axis_size == 0:
                    new_sharding[-1] = axis
                    print(f"sharding {name}:{shape} to {new_sharding}")
                    return tuple(new_sharding)
            elif any(
                s in name for s in ["down_kernel", "k_kernel", "v_kernel", "q_kernel"]
            ):
                # shard these on first dim (-2)
                if shape[-2] % axis_size == 0:
                    new_sharding[-2] = axis
                    print(f"sharding {name}:{shape} to {new_sharding}")
                    return tuple(new_sharding)
            else:
                # Partition along largest axis that is divisible and not taken starting
                # from last dimension.
                idx = np.argsort(shape)[::-1]
                for i in idx:
                    if shape[i] % axis_size == 0:
                        if cur_spec[i] is None:
                            new_sharding[i] = axis
                            print(f"sharding {name}:{shape} to {new_sharding}")
                            return tuple(new_sharding)

        write_note(
            f"Parameter {name}:{shape} not sharded because did not meet rules "
            f"or already occupied by other sharding rules: {cur_spec}"
        )
        return cur_spec

    return _update_spec
