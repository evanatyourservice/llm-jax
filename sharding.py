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

"""Base sharding functions from big vision changed for our nets and optimizers."""
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
        # Preconditioners for PSGD and tearfree shampoo kept in lists
        is_leaf=lambda v: isinstance(v, nn.Partitioned) or isinstance(v, list),
    )

    # Two-level tree_map to prevent it from doing traversal inside the spec.
    specs = jax.tree.map(lambda _, spec: P(*spec), nn.unbox(params), specs)
    sharding = jax.tree.map(lambda spec: NamedSharding(mesh, spec), specs)
    return sharding, specs


def fsdp_sharding(axis, min_size_to_shard_mb=1):
    """Simple FSDP sharding rules."""
    # TODO consider not overwriting already sharded dims
    axis = axis if isinstance(axis, str) else tuple(axis)
    axis_tuple = axis if isinstance(axis, tuple) else (axis,)

    def _update_spec(cur_spec, mesh, name, x):
        axis_size = np.prod([mesh.shape[a] for a in axis_tuple])

        if isinstance(x, list):
            # Preconditioners for PSGD and tearfree shampoo kept in lists
            precond_specs = []
            # psgd likes last dim sharded, shampoo first
            shard_dim = -1 if "Qs_preconditioners" in name[0] else -2
            for precond in x:
                shape = precond.shape
                new_sharding = [None for _ in shape]
                if (
                    np.prod(shape) * precond.dtype.itemsize
                    >= min_size_to_shard_mb * (2**20)
                    and len(shape) > 1
                    and shape[shard_dim] % axis_size == 0
                ):
                    new_sharding[shard_dim] = axis
                print(f"sharding {name}:{shape} to {new_sharding}")
                precond_specs.append(tuple(new_sharding))
            return precond_specs

        shape = x.shape

        # Partitioning rules, simple FSDP
        # indexed backwards from last dim for friendliness to scanned leading dims
        if (
            np.prod(shape) * x.dtype.itemsize >= min_size_to_shard_mb * (2**20)
            and len(shape) > 1
        ):
            new_sharding = [None for _ in shape]
            if "scale" in name or "bias" in name:
                pass
            elif any(s in name for s in ["embedding", "out_kernel", "down_kernel"]):
                # shard these on last dim (-1)
                if shape[-1] % axis_size == 0:
                    new_sharding[-1] = axis
                    print(f"sharding {name}:{shape} to {new_sharding}")
                    return tuple(new_sharding)
                else:
                    print(
                        f"WARNING: Parameter {name}:{shape} is not sharded because "
                        f"last dimension is not divisible by axis size {axis_size}. "
                        "Consider changing last dim to be divisible by axis size."
                    )
            elif any(
                s in name
                for s in [
                    "q_kernel",
                    "k_kernel",
                    "v_kernel",
                    "gate_kernel",
                    "up_kernel",
                ]
            ):
                # shard these on first dim (-2)
                if shape[-2] % axis_size == 0:
                    new_sharding[-2] = axis
                    print(f"sharding {name}:{shape} to {new_sharding}")
                    return tuple(new_sharding)
                else:
                    print(
                        f"WARNING: Parameter {name}:{shape} is not sharded because "
                        f"first dimension is not divisible by axis size {axis_size}. "
                        "Consider changing first dim to be divisible by axis size."
                    )
            else:
                # If not explicitly sharded above, infer here by partitioning
                # along largest axis that is divisible and not taken.
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
