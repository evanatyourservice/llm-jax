import jax
import jax.numpy as jnp


def parallel_stabilized_simple(
    queries: jnp.ndarray,
    keys: jnp.ndarray,
    values: jnp.ndarray,
    igate_preact: jnp.ndarray,
    fgate_preact: jnp.ndarray,
    stabilize_rowwise: bool = True,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """mLSTM cell in parallel form (JAX/Flax version)."""
    S, DH = queries.shape

    log_fgates = jax.nn.log_sigmoid(fgate_preact)  # (S, 1)
    log_fgates_cumsum = jnp.concatenate(
        [jnp.zeros((1, 1), dtype=log_fgates.dtype), jnp.cumsum(log_fgates, axis=0)],
        axis=0,
    )  # (S+1, 1)

    log_D_matrix = jnp.where(
        jnp.tri(S, k=0, dtype=bool),
        (log_fgates_cumsum - log_fgates_cumsum.T)[1:, 1:] + igate_preact.T,
        -jnp.inf,
    )  # (S, S)

    max_log_D = jnp.max(
        log_D_matrix, axis=-1 if stabilize_rowwise else None, keepdims=True
    )  # (S, 1) or (1, 1)

    D_matrix = jnp.exp(log_D_matrix - max_log_D)
    C_matrix = (queries @ (keys.T / jnp.sqrt(DH))) * D_matrix

    normalizer = jnp.maximum(
        jnp.sum(C_matrix, axis=-1, keepdims=True), jnp.exp(-max_log_D)
    )

    return (C_matrix / (normalizer + eps)) @ values
