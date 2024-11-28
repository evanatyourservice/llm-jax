import jax
from jax import numpy as jnp, vmap
import flax.linen as nn
from flax.linen.linear import Dense
from flax.linen.module import Module, compact
from flax.typing import Array

from model.mlstm.mlstm_backend import parallel_stabilized_simple


def bias_linspace_init(start: float = 3.0, end: float = 6.0):
    """Linearly spaced bias initialization function for Flax.
    
    Args:
        start: Starting value for linear spacing
        end: Ending value for linear spacing
    
    Returns:
        An initialization function compatible with Flax
    """
    def init(_, shape, *args):
        assert len(shape) == 1, f"param must be 1-dimensional (typically a bias), got {len(shape)}"
        return jnp.linspace(start, end, shape[0])

    return init


class RMSNorm(nn.Module):
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x):
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)
        x *= jax.lax.rsqrt(
            jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.epsilon
        )
        return x.astype(orig_dtype)


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

        # Gate layers should be initialized before reshape operations
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

        # Vectorized gate computation across batch dimension
        def gate_fn(x):
            if_gate_input = jnp.concatenate(x, axis=-1)
            return igate(if_gate_input), fgate(if_gate_input)

        igate_preact, fgate_preact = vmap(gate_fn)((q, k, v))  # (B, S, NH)

        # First reshape, then use einsum for transpose
        q = q.reshape(B, S, self.num_heads, head_dim)
        k = k.reshape(B, S, self.num_heads, head_dim)
        v = v.reshape(B, S, self.num_heads, head_dim)

        q = jnp.swapaxes(q, -3, -2)
        k = jnp.swapaxes(k, -3, -2)
        v = jnp.swapaxes(v, -3, -2)

        # Reshape gate preactivations using swapaxes like qkv
        igate_preact = jnp.swapaxes(igate_preact, -2, -1)[..., None]  # (B, NH, S, 1)
        fgate_preact = jnp.swapaxes(fgate_preact, -2, -1)[..., None]  # (B, NH, S, 1)

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

    @property
    def num_feature_axes(self) -> int:
        return 1
