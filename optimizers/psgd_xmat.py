from typing import Any, Optional, Union, Callable, NamedTuple
import numpy as np

import jax
from jax import numpy as jnp
from jax.random import PRNGKey

from optax import tree_utils as otu
from optax._src import base, transform, clipping
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype
from optax._src.combine import chain

from optimizers.utils import add_eps, apply_momentum


class PSGDXMatState(NamedTuple):
    count: jax.Array
    key: PRNGKey
    mu: Optional[base.Updates]
    a: jax.Array
    b: jax.Array


def scale_by_xmat(
    preconditioner_update_probability: Union[float, Callable[[int], float]] = 1.0,
    b1: float = 0.9,
    nesterov: bool = False,
    precond_lr: Union[float, Callable[[int], float]] = 0.1,
    precond_init_scale: Optional[float] = None,
    update_global_norm_clip: Optional[float] = None,
    momentum_before_precond_update: bool = True,
    step_normalizer_order: str = "2nd",
    seed: Optional[PRNGKey] = None,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precision: str = "tensorfloat32",
    precond_sharding: Any = None,
) -> base.GradientTransformationExtraArgs:
    """
    Implements XMat PSGD from https://github.com/lixilinx/psgd_torch.

    Args:
        preconditioner_update_probability: float, probability of updating the
            preconditioner.
        b1: float, momentum parameter.
        nesterov: bool, whether to use Nesterov momentum.
        precond_lr: float or callable, learning rate for the preconditioner.
        precond_init_scale: optional float, initial scale for the preconditioner.
        update_global_norm_clip: optional float, clip updates by global norm.
        momentum_before_precond_update: bool, whether to pass momentum into
            preconditioner instead of raw gradients.
        step_normalizer_order: str, '1st' or '2nd'.
        seed: Optional PRNGKey, random seed.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum accumulator.
            Defaults to the same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioner.
        precision: str, precision for matmul, 'bfloat16', 'tensorfloat32', 'float32'.
        precond_sharding: optional Any, sharding spec for parameters.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    mu_dtype = canonicalize_dtype(mu_dtype)
    precond_dtype = canonicalize_dtype(precond_dtype)

    def init_fn(params):
        key = seed if seed is not None else jax.random.PRNGKey(36)

        # momentum
        mu = None
        if b1 > 0:
            print("PSGD: Using momentum.")
            mu = otu.tree_zeros_like(params, mu_dtype)

        # preconditioner
        a = jax.tree.map(lambda x: jnp.ones_like(x, dtype=precond_dtype), params)
        b = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=precond_dtype), params)

        # initial state
        return PSGDXMatState(count=jnp.zeros([], jnp.int32), key=key, mu=mu, a=a, b=b)

    def update_fn(
        updates: base.Updates,
        state: PSGDXMatState,
        params: base.Params = None,
        Hvp: Optional[base.Updates] = None,
        vector: Optional[base.Updates] = None,
        update_preconditioner: Optional[bool] = None,
    ):
        del params
        # use hessian preconditioning if hessian provided
        # otherwise use gg^T whitening type preconditioning
        hessian_based_preconditioning = Hvp is not None
        if hessian_based_preconditioning and (
            vector is None or update_preconditioner is None
        ):
            raise ValueError(
                "If using Hessian-based preconditioning, must also pass in random vector and "
                "update_preconditioner to PSGD's update function. See README for more info."
            )

        count_inc = safe_int32_increment(state.count)
        key = state.key

        update_prob_in = preconditioner_update_probability
        if isinstance(preconditioner_update_probability, Callable):
            update_prob_in = preconditioner_update_probability(count_inc)

        precond_lr_in = precond_lr
        if isinstance(precond_lr, Callable):
            precond_lr_in = precond_lr(count_inc)

        def _update_precond(key: PRNGKey, a, b, Hvs, vs):
            def init_a(a, h, v):
                if precond_init_scale is not None:
                    init_scale = precond_init_scale
                else:
                    if hessian_based_preconditioning:
                        init_scale = (
                            jnp.sum(jnp.square(v)) / jnp.sum(jnp.square(h))
                        ) ** 0.25
                    else:
                        init_scale = (h.size / jnp.sum(jnp.square(h))) ** 0.25
                return a * init_scale

            # init a
            a = jax.lax.cond(
                state.count == 0,
                lambda a, h, v: jax.tree.map(init_a, a, h, v),
                lambda a, h, v: a,
                a,
                Hvs,
                vs,
            )

            # update preconditioner
            new_as = []
            new_bs = []
            for ai, bi, v, h in zip(a, b, vs, Hvs):
                new_a, new_b = _update_precond_Xmat_math_(
                    ai, bi, v, h, precond_lr_in, step_normalizer_order, precision
                )
                new_as.append(new_a)
                new_bs.append(new_b)

            return key, new_as, new_bs

        def _dont_update_precond(key, a, b, Hvs, vs):
            return key, a, b

        # momentum
        mu = None
        momentum_updates = updates
        if state.mu is not None:
            momentum_updates, mu = apply_momentum(
                updates, state.mu, count_inc, b1, nesterov
            )

        if not hessian_based_preconditioning:
            # update cond and vector not passed in, create here
            key, subkey = jax.random.split(key)
            update_preconditioner = jnp.logical_or(
                jax.random.uniform(subkey) < update_prob_in, state.count < 2
            )
            key, subkey = jax.random.split(key)
            flat_updates, updates_struct = jax.tree.flatten(updates)
            key_list = jax.random.split(subkey, len(flat_updates))
            vector = [
                jax.random.normal(k, x.shape, x.dtype)
                for k, x in zip(key_list, flat_updates)
            ]
            vector = jax.tree_unflatten(updates_struct, vector)
            if precond_sharding is not None:
                vector = jax.lax.with_sharding_constraint(vector, precond_sharding)

            # use grads as Hvp
            if momentum_before_precond_update:
                Hvp = momentum_updates
            else:
                Hvp = updates

        flat_a, a_struct = jax.tree.flatten(state.a)
        flat_b, _ = jax.tree.flatten(state.b)
        flat_h, _ = jax.tree.flatten(Hvp)
        flat_v, _ = jax.tree.flatten(vector)
        key, a, b = jax.lax.cond(
            update_preconditioner,
            _update_precond,
            _dont_update_precond,
            key,
            flat_a,
            flat_b,
            flat_h,
            flat_v,
        )
        a = a_struct.unflatten(a)
        b = a_struct.unflatten(b)

        # preconditioning
        updates = jax.tree.map(_precond_grad_Xmat_math, a, b, momentum_updates)

        # clipping
        if update_global_norm_clip:
            updates, _ = clipping.clip_by_global_norm(update_global_norm_clip).update(
                updates, base.EmptyState
            )

        # elementwise clipping
        updates = jax.tree.map(lambda x: jnp.clip(x, -1.0, 1.0), updates)

        mu = otu.tree_cast(mu, mu_dtype)
        a = otu.tree_cast(a, precond_dtype)
        b = otu.tree_cast(b, precond_dtype)
        state = PSGDXMatState(count=count_inc, key=key, mu=mu, a=a, b=b)
        return updates, state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def xmat(
    learning_rate: Union[float, Callable[[int], float]] = 0.01,
    preconditioner_update_probability: float = 1.0,
    b1: float = 0.9,
    nesterov: bool = False,
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    precond_lr: Union[float, Callable[[int], float]] = 0.1,
    precond_init_scale: Optional[float] = None,
    update_global_norm_clip: Optional[float] = None,
    momentum_before_precond_update: bool = True,
    step_normalizer_order: str = "2nd",
    seed: Optional[PRNGKey] = None,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precision: str = "tensorfloat32",
    precond_sharding: Any = None,
) -> base.GradientTransformationExtraArgs:
    """
    Implements XMat PSGD from https://github.com/lixilinx/psgd_torch.

    Args:
        learning_rate: float or callable, learning rate for the optimizer.
        preconditioner_update_probability: float, probability of updating the
            preconditioner.
        b1: float, momentum parameter.
        nesterov: bool, whether to use Nesterov momentum.
        weight_decay: float, weight decay.
        mask: optional mask for weight decay.
        precond_lr: float or callable, learning rate for the preconditioner.
        precond_init_scale: optional float, initial scale for the preconditioner.
        update_global_norm_clip: optional float, clip updates by global norm.
        momentum_before_precond_update: bool, whether to pass momentum into
            preconditioner instead of raw gradients.
        step_normalizer_order: str, '1st' or '2nd'.
        seed: Optional PRNGKey, random seed.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum accumulator.
            Defaults to the same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioner.
        precision: str, precision for matmul, 'bfloat16', 'tensorfloat32', 'float32'.
        precond_sharding: optional Any, sharding spec for parameters.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    opt = [
        scale_by_xmat(
            preconditioner_update_probability=preconditioner_update_probability,
            b1=b1,
            nesterov=nesterov,
            precond_lr=precond_lr,
            precond_init_scale=precond_init_scale,
            update_global_norm_clip=update_global_norm_clip,
            momentum_before_precond_update=momentum_before_precond_update,
            step_normalizer_order=step_normalizer_order,
            seed=seed,
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
            precision=precision,
            precond_sharding=precond_sharding,
        )
    ]
    if weight_decay > 0:
        opt.append(transform.add_decayed_weights(weight_decay, mask=mask))
    opt.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*opt)


def flip(x):
    return jnp.flip(x, list(np.arange(x.ndim)))


def _update_precond_Xmat_math_(a, b, v, h, precond_lr, step_normalizer, precision):
    """
    Update preconditioner Q = diag(a) + adiag(b) with (vector, Hessian-vector product) = (v, h).
    """
    with jax.default_matmul_precision(precision):
        Qh = a * h + b * flip(h)
        aflip, bflip = flip(a), flip(b)
        invQtv = (aflip * v - bflip * flip(v)) / (a * aflip - b * bflip)

        u, v = Qh * Qh, invQtv * invQtv
        nablaA = u - v
        nablaB = Qh * flip(Qh) - invQtv * flip(invQtv)
        orig_shape = nablaB.shape
        shape_without_ones = [s for s in orig_shape if s != 1]
        if len(shape_without_ones) == 1:
            nablaB = nablaB.reshape((-1,))
            q, r = jnp.divmod(len(nablaB), 2)
            nablaB = jnp.where(r == 1, nablaB.at[q].set(0), nablaB)
            nablaB = nablaB.reshape(orig_shape)

        if step_normalizer == "2nd":
            mu = precond_lr / add_eps(jnp.max(u + v))
        else:
            mu = precond_lr / add_eps(
                jnp.maximum(jnp.max(jnp.abs(nablaA)), jnp.max(jnp.abs(nablaB)))
            )

        a -= mu * (nablaA * a + nablaB * bflip)
        b -= mu * (nablaA * b + nablaB * aflip)

        return a, b


def _precond_grad_Xmat_math(a, b, g):
    """
    Preconditioning gradient g with Q = diag(a) + adiag(b).

    All variables here are either matrices or column vectors.
    """
    if g.size > 1:
        ab = a * b
        return (a * a + flip(b * b)) * g + (ab + flip(ab)) * flip(g)
    else:
        return g
