import jax
from jax import numpy as jnp, vmap
import flax.linen as nn
from flax.linen.linear import Dense
from flax.linen.module import Module, compact
from flax.typing import Array


class RMSNorm(nn.Module):
    """RMSNorm layer.

    Upcasts to float32 and back."""

    @nn.compact
    def __call__(self, x):
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = x * jax.lax.rsqrt(var + 1e-06)
        normed_inputs = normed_inputs.astype(x.dtype)

        scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
        scale = jnp.expand_dims(scale, axis=range(len(x.shape) - 1))
        normed_inputs = normed_inputs * (1 + scale)
        return normed_inputs


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


class mLSTMCell(Module):
    """Multiplicative LSTM cell.

    Attributes:
        embedding_dim: number of output features
        num_heads: number of attention heads
        context_length: length of the context window
    """

    embedding_dim: int
    num_heads: int
    context_length: int

    @compact
    def __call__(self, q: Array, k: Array, v: Array) -> Array:
        """Apply the mLSTM cell.

        Args:
            q: Query tensor of shape (B, S, H)
            k: Key tensor of shape (B, S, H)
            v: Value tensor of shape (B, S, H)

        Returns:
            Output tensor of shape (B, S, H)
        """
        B, S, _ = q.shape
        head_dim = self.embedding_dim // self.num_heads

        igate = Dense(
            features=self.num_heads,
            name="igate",
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.normal(stddev=0.1),
        )
        fgate = Dense(
            features=self.num_heads,
            name="fgate",
            kernel_init=nn.initializers.zeros_init(),
            bias_init=bias_linspace_init(),
        )

        if_gate_input = jnp.concatenate((q, k, v), axis=-1)
        igate_preact = igate(if_gate_input)  # (B, S, NH)
        igate_preact = igate_preact.mT[..., None]  # (B, NH, S, 1)
        fgate_preact = fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = fgate_preact.mT[..., None]  # (B, NH, S, 1)

        q = q.reshape(B, S, self.num_heads, head_dim)
        k = k.reshape(B, S, self.num_heads, head_dim)
        v = v.reshape(B, S, self.num_heads, head_dim)

        q = jnp.swapaxes(q, -3, -2)
        k = jnp.swapaxes(k, -3, -2)
        v = jnp.swapaxes(v, -3, -2)

        h_state = vmap(vmap(parallel_stabilized_simple))(
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
        )  # (B, NH, S, DH)

        h_state = RMSNorm()(h_state)

        # Fix the final reshape to maintain the original embedding dimension
        h_state = jnp.swapaxes(h_state, -3, -2)  # (B, S, NH, DH)
        h_state = h_state.reshape(B, S, self.embedding_dim)  # (B, S, H)

        return h_state


def bias_linspace_init(start: float = 3.0, end: float = 6.0):
    """Linearly spaced bias initialization function for Flax.

    Args:
        start: Starting value for linear spacing
        end: Ending value for linear spacing

    Returns:
        An initialization function compatible with Flax
    """

    def init(_, shape, *args):
        assert (
            len(shape) == 1
        ), f"param must be 1-dimensional (typically a bias), got {len(shape)}"
        return jnp.linspace(start, end, shape[0])

    return init
