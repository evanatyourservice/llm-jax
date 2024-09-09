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
        # CHANGED: assuming preconditioners are kept in lists
        is_leaf=lambda v: isinstance(v, nn.Partitioned) or isinstance(v, list),
    )

    # Two-level tree_map to prevent it from doing traversal inside the spec.
    specs = jax.tree.map(lambda _, spec: P(*spec), nn.unbox(params), specs)
    sharding = jax.tree.map(lambda spec: NamedSharding(mesh, spec), specs)
    return sharding, specs


def fsdp_sharding(axis, min_size_to_shard_mb=1):
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

        # Preconditioner sharding
        # We're assuming preconditioners are kept in lists.
        # This is at least the case for PSGD affine and tearfree shampoo.
        if isinstance(x, list):
            return [
                (
                    (axis,)
                    if len(p.shape) > 1
                    and np.prod(p.shape) * p.dtype.itemsize
                    >= min_size_to_shard_mb * (2**20)
                    and p.shape[0] % axis_size == 0
                    else (None,)
                )
                for p in x
            ]

        shape = x.shape

        # Params sharding
        if np.prod(shape) * x.dtype.itemsize < min_size_to_shard_mb * (2**20):
            return cur_spec

        # Partition along largest axis that is divisible and not taken starting
        # from last dimension.
        idx = np.argsort(shape)[::-1]
        for i in idx:
            if shape[i] % axis_size == 0:
                if cur_spec[i] is None:
                    return cur_spec[:i] + (axis,) + cur_spec[i + 1 :]

        write_note(
            f"Failed to apply `fsdp` rule to the parameter {name}:{shape}, "
            f"as all its dimensions are not divisible by the requested axis: "
            f"{axis}:{axis_size}, or already occupied by other sharding "
            f"rules: {cur_spec}"
        )
        return cur_spec

    return _update_spec
