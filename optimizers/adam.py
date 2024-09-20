from typing import Optional, Any, Callable, Union

import chex
import jax
import jax.numpy as jnp
from optax._src import base, numerics, utils, transform, combine
from optax import tree_utils as otu


def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
    nesterov: bool = False,
) -> base.GradientTransformation:
    """Same as optax version but doesn't create momentum buffer if b1 == 0."""
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        if b1 > 0:
            mu = otu.tree_zeros_like(params, dtype=mu_dtype)
        else:
            mu = None
        nu = otu.tree_zeros_like(params)
        state = transform.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

        # Calculate sizes for nu (preconditioner) and mu (momentum)
        nu_n_elements = sum(leaf.size for leaf in jax.tree.leaves(nu))
        nu_size_MB = sum(
            leaf.size * leaf.dtype.itemsize / (2**20) for leaf in jax.tree.leaves(nu)
        )
        if jax.process_index() == 0:
            print(
                f"Adam Preconditioner (nu) size: {nu_n_elements} elements, {nu_size_MB:.2f} MB"
            )
        if mu is not None:
            mu_n_elements = sum(leaf.size for leaf in jax.tree.leaves(mu))
            mu_size_MB = sum(
                leaf.size * leaf.dtype.itemsize / (2**20)
                for leaf in jax.tree.leaves(mu)
            )
            if jax.process_index() == 0:
                print(
                    f"Adam Momentum (mu) size: {mu_n_elements} elements, {mu_size_MB:.2f} MB"
                )

        return state

    def update_fn(updates, state, params=None):
        del params
        count_inc = numerics.safe_int32_increment(state.count)
        if b1 > 0:
            mu = otu.tree_update_moment(updates, state.mu, b1, 1)
            if nesterov:
                mu_hat = jax.tree.map(
                    lambda m, g: b1 * m + (1 - b1) * g,
                    otu.tree_bias_correction(
                        mu, b1, numerics.safe_int32_increment(count_inc)
                    ),
                    otu.tree_bias_correction(updates, b1, count_inc),
                )
            else:
                mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
            mu = otu.tree_cast(mu, mu_dtype)
        else:
            mu = None
            mu_hat = updates
        nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
        nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
        updates = jax.tree.map(
            lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat
        )
        return updates, transform.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


def adamw(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    nesterov: bool = False,
) -> base.GradientTransformation:
    return combine.chain(
        scale_by_adam(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
        ),
        transform.add_decayed_weights(weight_decay, mask),
        transform.scale_by_learning_rate(learning_rate),
    )
