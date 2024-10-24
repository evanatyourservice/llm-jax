# Copyright 2024 The precondition Authors.
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

"""Parameter reshaping module."""

import dataclasses
import functools
import numpy as np

import jax
from jax import numpy as jnp
import optax


@dataclasses.dataclass
class Options:
    """Parameter reshaping options.

    Attributes:
      merge_dims: Collapse dimensions smaller than this number left-to-right,
        e.g., [3, 1, 5, 2, 2] becomes [3, 5, 4] with `merge_dims = 4`. Notice
        ordering, [2, 3, 2] becomes [6, 2] with `merge_dims = 6`, not its reverse.
      block_size: If nonzero, pads all dimensions larger than the block size to a
        multiple of the block size.
    """

    merge_dims: int = 8192
    block_size: int = 256


@dataclasses.dataclass
class _Shapes:
    """Shape container."""

    original_shape: list[int]
    merged_shape: list[int]
    padded_shape: list[int]


def _derive_shapes(options: Options, param: jax.Array) -> _Shapes:
    """Derive desired shapes from options."""
    merged = _merge_small_dims(param.shape, options.merge_dims)
    if merged == [1]:
        return _Shapes(
            original_shape=list(param.shape), merged_shape=[], padded_shape=[]
        )
    if options.block_size == 0:
        padded = merged
    else:
        padded = []
        for s in merged:
            if s >= options.block_size:
                s = (s + options.block_size - 1) // options.block_size
                s *= options.block_size
            padded.append(s)
    return _Shapes(
        original_shape=list(param.shape), merged_shape=merged, padded_shape=padded
    )


def merge(options: Options) -> optax.GradientTransformation:
    """Merge and maybe pad gradients, leaving params alone."""

    if options.merge_dims < 2:
        raise ValueError(
            "merge_dims ({}) must be at least 2".format(options.merge_dims)
        )

    if options.block_size < 2 and options.block_size != 0:
        raise ValueError(
            "block_size ({}) must be at least 2 (or 0 to disable)".format(
                options.block_size
            )
        )

    def _merge(update: jax.Array, shapes: _Shapes) -> jax.Array:
        assert list(update.shape) == shapes.original_shape, (update.shape, shapes)
        merged = update.reshape(shapes.merged_shape)
        padding = [(0, p - m) for p, m in zip(shapes.padded_shape, shapes.merged_shape)]
        if padding and options.block_size > 0:
            return jnp.pad(merged, padding)
        return merged

    def update(
        updates: optax.Updates, state: optax.MaskedNode, params: optax.Params
    ) -> tuple[optax.Updates, optax.MaskedNode]:
        shapes = jax.tree.map(functools.partial(_derive_shapes, options), params)
        new_updates = jax.tree.map(_merge, updates, shapes)
        return new_updates, state

    return optax.GradientTransformation(lambda _: optax.MaskedNode(), update)


def unmerge(options: Options) -> optax.GradientTransformation:
    """Unmerge and unpad gradients, leaving params alone."""

    def _unmerge(update: jax.Array, shapes: _Shapes) -> jax.Array:
        assert list(update.shape) == shapes.padded_shape, (update.shape, shapes)
        if options.block_size == 0:
            merged = update
        else:
            merged = update[tuple(slice(0, m) for m in shapes.merged_shape)]
        return merged.reshape(shapes.original_shape)

    def update(
        updates: optax.Updates, state: optax.MaskedNode, params: optax.Params
    ) -> tuple[optax.Updates, optax.MaskedNode]:
        shapes = jax.tree.map(functools.partial(_derive_shapes, options), params)
        new_updates = jax.tree.map(_unmerge, updates, shapes)
        return new_updates, state

    return optax.GradientTransformation(lambda _: optax.MaskedNode(), update)


def _merge_small_dims(shape_to_merge, max_dim):
    """Merge small dimensions.

    If there are some small dimensions, we collapse them:
    e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if max_dim = 1024
         [1, 2, 768, 1, 2048] --> [2, 768, 2048]

    Args:
      shape_to_merge: Shape to merge small dimensions.
      max_dim: Maximal dimension of output shape used in merging.

    Returns:
      Merged shape.
    """
    if shape_to_merge and np.all(np.array(shape_to_merge) == 1):
        return [1]

    resulting_shape = []
    product = 1
    for d in shape_to_merge:
        if product * d <= max_dim:
            product *= d
        else:
            if product > 1:
                resulting_shape.append(product)
            product = d
    if product > 1:
        resulting_shape.append(product)
    return resulting_shape
