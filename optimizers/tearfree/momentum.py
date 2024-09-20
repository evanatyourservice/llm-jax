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

"""Momentum configuration and transform."""

import copy
import dataclasses
from typing import Any, NamedTuple, Union, Optional

import jax
import jax.tree_util as jtu
import optax
from optax._src import base, utils
import optax.tree_utils as otu
from optimizers.tearfree import praxis_shim


@dataclasses.dataclass
class Options:
    """Configuration dataclass for momentum.

    Notably, this class contains weight decay parameters. Why?

    In classical convex literature, Nesterov acceleration applied to gradient
    descent can be viewed as "revising" the last iterate's momentum based on
    the gradient we observe immediately after taking a momentum "gamble"
    (see viz, https://stats.stackexchange.com/a/191727).

    To maintain this interpretation exactly, we would need to go against
    the grain on how weight decay is implemented. Momentum must be the last*
    gradient transformation applied to the iterate, which would require the
    weight decay to be applied to the update before it's used to change
    the velocity (momentum's state, the first moment).

    In particular, AdamW and Adafactor suggest direct weight downscaling,
    excluding weight decay from the velocity accumulation.

    As a result, the true meaning of Nesterov acceleration here is better
    understood literally, described in its parameter doc.

    *Technically, some optimizers include the learning rate in the update used to
    update the velocity (e.g., Adafactor), but others apply the learning rate
    scaling last, after momentum (e.g., Adam). We can recover the former from the
    latter by dividing the decay by the root of the learning rate, so this
    particular "gradient transformation" shouldn't be viewed as affecting
    the Nesterov interpretation, up to tuning constants.

    Attributs:
      ema: If true, momentum is computed as an exponential moving
        average: `velocity(t+1) = decay * velocity(t) + (1 - decay) * update(t)`
        If false, then uses "trace" accumulation for momentum:
        `velocity(t+1) = decay * velocity(t) + update(t)`. Note that if the
        updates were the same (they aren't) then these would be the same up to a
        factor of `(1 - decay)`. This corresponds to distributed_shampoo argument
        `moving_average_for_momentum`.
      nesterov: Toggle for Nesterov acceleration. If false, then the new
        update `update'(t+1)` simply equals `velocity(t+1)`. If true, then
        `update'(t+1) = maybe_decay * update(t) + decay * velocity(t+1)`, where
        `maybe_decay` is `(1 - decay)` if `ema` and 1 otherwise.
      momentum_decay: The decay referred to in `ema` and `nesterov` formulas.
      weight_decay: Add `weight_decay * x(t)` to the `update(t)` value, where
        `x(t)` is the value of the current parameters.
      weight_decay_after_momentum: Whether weight decay addition is performed
        after the momentum transformation.
      momentum_dtype: str, `float32` or `bfloat16`, dtype of momentum buffer.
    """

    ema: bool = True
    nesterov: bool = False
    momentum_decay: Optional[float] = 0.9
    weight_decay: float = 1e-4
    weight_decay_after_momentum: bool = True
    momentum_dtype: str = "float32"


State = Union[optax.MaskedNode, optax.TraceState]


def apply(options: Options) -> praxis_shim.ShardedGradientTransformation:
    """Generate the momentum update from options."""
    _validate(options)

    momentum_transforms = []
    if options.momentum_decay:
        if options.ema:
            momentum_transforms.append(optax.scale(1 - options.momentum_decay))
        momentum_transforms.append(
            _sharded_trace(
                options.momentum_decay, options.nesterov, options.momentum_dtype
            )
        )

    wd_transforms = [optax.add_decayed_weights(options.weight_decay)] * (
        options.weight_decay > 0.0
    )

    if options.weight_decay_after_momentum:
        transforms = momentum_transforms + wd_transforms
    else:
        transforms = wd_transforms + momentum_transforms

    return praxis_shim.sharded_chain(*transforms)


def _validate(options: Options):
    """Raise ValueError if options are invalid."""
    if options.momentum_decay is not None:
        if not (0 <= options.momentum_decay <= 1):
            raise ValueError(
                "momentum_decay ({}) must be in [0, 1]".format(options.momentum_decay)
            )

    if not (options.weight_decay >= 0):
        raise ValueError("weight_decay ({}) must be >= 0".format(options.weight_decay))


def _sharded_trace(
    momentum: float, nesterov: bool, accumulator_dtype: str
) -> praxis_shim.ShardedGradientTransformation:
    """Extend optax's trace to allow sharding."""
    trace_transform = trace(momentum, nesterov, accumulator_dtype=accumulator_dtype)

    def init_pspec_fn(mdl_params):
        def _opt_state_sharding_spec(var_hparams):
            s_var_hparams = copy.deepcopy(var_hparams)
            s_var_hparams.init = None
            return s_var_hparams

        mdl_sharding = jax.tree.map(_opt_state_sharding_spec, mdl_params)
        return TraceState(trace=mdl_sharding)

    return praxis_shim.ShardedGradientTransformation(
        trace_transform.init, trace_transform.update, init_pspec_fn
    )


class TraceState(NamedTuple):
    """Holds an aggregation of past updates."""

    trace: base.Params


def trace(
    decay: float, nesterov: bool = False, accumulator_dtype: Optional[Any] = None
) -> base.GradientTransformation:
    """Compute a trace of past updates.

    Note: `trace` and `ema` have very similar but distinct updates;
    `trace = decay * trace + t`, while `ema = decay * ema + (1-decay) * t`.
    Both are frequently found in the optimization literature.

    Args:
      decay: Decay rate for the trace of past updates.
      nesterov: Whether to use Nesterov momentum.
      accumulator_dtype: Optional `dtype` to be used for the accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.

    Returns:
      A `GradientTransformation` object.
    """

    accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)

    def init_fn(params):
        trace = otu.tree_zeros_like(params, dtype=accumulator_dtype)

        # Calculate and print size for trace
        trace_n_elements = sum(leaf.size for leaf in jax.tree.leaves(trace))
        trace_size_MB = sum(
            leaf.size * leaf.dtype.itemsize / (2**20) for leaf in jax.tree.leaves(trace)
        )
        if jax.process_index() == 0:
            print(f"Momentum size: {trace_n_elements} elements, {trace_size_MB:.2f} MB")

        return TraceState(trace=trace)

    def update_fn(updates, state, params=None):
        del params
        f = lambda g, t: g + decay * t
        new_trace = jtu.tree_map(f, updates, state.trace)
        updates = jtu.tree_map(f, updates, new_trace) if nesterov else new_trace
        new_trace = otu.tree_cast(new_trace, accumulator_dtype)
        return updates, TraceState(trace=new_trace)

    return base.GradientTransformation(init_fn, update_fn)
