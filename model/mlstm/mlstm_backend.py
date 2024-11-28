import jax
import jax.numpy as jnp


def parallel_stabilized_simple(
    queries: jnp.ndarray,
    keys: jnp.ndarray,
    values: jnp.ndarray,
    igate_preact: jnp.ndarray,
    fgate_preact: jnp.ndarray,
    lower_triangular_matrix: jnp.ndarray = None,
    stabilize_rowwise: bool = True,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """This is the mLSTM cell in parallel form (JAX/Flax version).

    Args:
        queries: (S, DH)
        keys: (S, DH)
        values: (S, DH)
        igate_preact: (S, 1)
        fgate_preact: (S, 1)
        lower_triangular_matrix: (S,S). Defaults to None.
        stabilize_rowwise: Whether to stabilize the combination matrix C rowwise (take maximum per row).
            Alternative: Subtract the maximum over all rows. Defaults to True.

    Returns:
        jnp.ndarray: (S, DH), h_tilde_state
    """
    S, DH = queries.shape

    log_fgates = jax.nn.log_sigmoid(fgate_preact)  # (S, 1)

    if lower_triangular_matrix is None or S < lower_triangular_matrix.shape[-1]:
        ltr = jnp.tri(S, k=0, dtype=bool)
    else:
        ltr = lower_triangular_matrix

    log_fgates_cumsum = jnp.concatenate(
        [jnp.zeros((1, 1), dtype=queries.dtype), jnp.cumsum(log_fgates, axis=-2)],
        axis=-2,
    )  # (S+1, 1)

    _log_fg_matrix = log_fgates_cumsum.repeat(S + 1, axis=-1)  # (S+1, S+1)
    _log_fg_matrix = _log_fg_matrix - _log_fg_matrix.T

    log_fg_matrix = jnp.where(ltr, _log_fg_matrix[1:, 1:], -jnp.inf)  # (S, S)

    log_D_matrix = log_fg_matrix + igate_preact.T  # (S, S)

    if stabilize_rowwise:
        max_log_D = jnp.max(log_D_matrix, axis=-1, keepdims=True)  # (S, 1)
    else:
        max_log_D = jnp.max(log_D_matrix.reshape(-1), axis=-1, keepdims=True)[
            ..., None
        ]  # (1, 1)

    log_D_matrix_stabilized = log_D_matrix - max_log_D
    D_matrix = jnp.exp(log_D_matrix_stabilized)

    keys_scaled = keys / jnp.sqrt(DH)
    qk_matrix = jnp.einsum("sd,md->sm", queries, keys_scaled)
    C_matrix = qk_matrix * D_matrix

    normalizer = jnp.maximum(
        jnp.abs(jnp.sum(C_matrix, axis=-1, keepdims=True)), jnp.exp(-max_log_D)
    )
    C_matrix_normalized = C_matrix / (normalizer + eps)

    h_tilde_state = jnp.einsum("sm,md->sd", C_matrix_normalized, values)

    return h_tilde_state
