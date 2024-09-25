import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding as NS, PartitionSpec as P
import flax.linen as nn


initializer = nn.initializers.normal(0.02)
constrain = lambda x, mesh, spec: jax.lax.with_sharding_constraint(x, NS(mesh, spec))


def _get_large_negative(dtype):
    dtype_max = jnp.finfo(dtype).max
    return jnp.asarray(-0.7 * dtype_max, dtype=dtype)


def _get_causal_mask(T, S):
    mask = jnp.tril(jnp.ones((T, S), dtype=jnp.bool_))
    return mask[None, None, :, :]


def _get_window_mask(T: int, S: int, local_window_size: tuple[int, int]):
    query_pos = jnp.array(range(T))
    key_pos = jnp.array(range(S))
    left_window, right_window = local_window_size
    left_mask = query_pos[..., None] <= key_pos[..., None, :] + left_window
    right_mask = query_pos[..., None] >= key_pos[..., None, :] - right_window
    return jnp.logical_and(right_mask, left_mask)[None, None, :, :]


def _get_padding_mask_logits(T, S, q_seqlen, kv_seqlen):
    q_mask = True
    kv_mask = True
    if q_seqlen is not None:
        q_indices = jnp.arange(0, T)[None, :, None]
        q_mask = q_indices < q_seqlen[:, None, None]
    if kv_seqlen is not None:
        kv_indices = jnp.arange(0, S)[None, None, :]
        kv_mask = kv_indices < kv_seqlen[:, None, None]
    mask = jnp.logical_and(q_mask, kv_mask)
    return mask[:, None, :, :]


def _apply_masks(logits, mask, is_causal, q_seqlen, kv_seqlen, local_window_size):
    if mask is None and not is_causal and q_seqlen is None and kv_seqlen is None:
        return logits

    combined_mask = jnp.ones_like(logits, dtype=jnp.bool_)
    if mask is not None:
        assert mask.dtype == jnp.bool_
        combined_mask = jnp.logical_and(combined_mask, mask)

    T, S = logits.shape[2], logits.shape[3]

    if is_causal:
        mask = _get_causal_mask(T, S)
        combined_mask = jnp.logical_and(combined_mask, mask)

    if local_window_size is not None:
        mask = _get_window_mask(T, S, local_window_size)
        combined_mask = jnp.logical_and(combined_mask, mask)

    if q_seqlen is not None or kv_seqlen is not None:
        mask = _get_padding_mask_logits(T, S, q_seqlen, kv_seqlen)
        combined_mask = jnp.logical_and(combined_mask, mask)

    large_negative_number = _get_large_negative(logits.dtype)
    padded_logits = jnp.where(combined_mask, logits, large_negative_number)
    return padded_logits


def _dot_product_attention_core(query, key, value, is_causal, local_window_size):
    # gemma style
    query *= jnp.array(query.shape[-1] ** -0.5, dtype=query.dtype)
    logits = jnp.einsum("BTNH,BSNH->BNTS", query, key)
    logits = jnp.tanh(logits / 50.0) * 50.0

    padded_logits = _apply_masks(logits, None, is_causal, None, None, local_window_size)

    probs = jax.nn.softmax(padded_logits.astype(jnp.float32)).astype(
        padded_logits.dtype
    )

    encoded = jnp.einsum("BNTS,BSNH->BTNH", probs, value)
    return encoded


def _sine_table(features, length, min_timescale=1.0, max_timescale=10000.0):
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction
    rotational_frequency = 1.0 / timescale
    # Must use high precision einsum here, bfloat16 rounding is catastrophic.
    sinusoid_inp = jnp.einsum(
        "i,j->ij",
        jnp.arange(length),
        rotational_frequency,
        precision=jax.lax.Precision.HIGHEST,
    )
    sinusoid_inp = jnp.concatenate([sinusoid_inp, sinusoid_inp], axis=-1)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def _rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    x = jnp.concatenate([-x2, x1], axis=-1)
    return x


def _apply_rotary_embedding(q, k, cos, sin):
    qlen = q.shape[-4]
    klen = k.shape[-3]

    qcos = jnp.expand_dims(cos[:qlen, :], range(len(q.shape) - 2))
    qsin = jnp.expand_dims(sin[:qlen, :], range(len(q.shape) - 2))
    kcos = jnp.expand_dims(cos[:klen, :], range(len(k.shape) - 2))
    ksin = jnp.expand_dims(sin[:klen, :], range(len(k.shape) - 2))

    qcos = jnp.swapaxes(qcos, -2, -4)
    qsin = jnp.swapaxes(qsin, -2, -4)
    kcos = jnp.swapaxes(kcos, -2, -3)
    ksin = jnp.swapaxes(ksin, -2, -3)

    out_q = q * qcos + _rotate_half(q) * qsin
    out_k = k * kcos + _rotate_half(k) * ksin

    return out_q.astype(q.dtype), out_k.astype(k.dtype)


class Attention(nn.Module):
    """Multi-head attention with RoPE and GQA.

    Upcasts to float32 and back for softmax."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float
    sliding_window_size: int
    mesh: Mesh

    @nn.compact
    def __call__(self, x):
        B, T, C = x.shape
        N = self.num_heads
        K = self.num_kv_heads
        G = N // K
        H = self.head_dim

        q_params = self.param("q_kernel", initializer, (C, N * H))
        k_params = self.param("k_kernel", initializer, (C, K * H))
        v_params = self.param("v_kernel", initializer, (C, K * H))
        out_params = self.param("out_kernel", initializer, (N * H, C))

        q = jnp.dot(x, q_params)
        k = jnp.dot(x, k_params)
        v = jnp.dot(x, v_params)

        q = jnp.reshape(q, (B, T, K, G, H))
        k = jnp.reshape(k, (B, T, K, H))
        v = jnp.reshape(v, (B, T, K, H))

        with jax.named_scope("rope"):
            sin, cos = _sine_table(H, T, max_timescale=self.rope_theta)
            q, k = _apply_rotary_embedding(q, k, cos, sin)

        # vmapped_fn = jax.vmap(
        #     _dot_product_attention_core, in_axes=(3, None, None, None, None), out_axes=3
        # )
        # encoded = vmapped_fn(q, k, v, True, (self.sliding_window_size, 0))
        encoded = []
        for i in range(G):
            encoded.append(
                _dot_product_attention_core(
                    q[:, :, :, i], k, v, True, (self.sliding_window_size, 0)
                )
            )
        encoded = jnp.stack(encoded, axis=3)

        encoded = jnp.reshape(encoded, (B, T, N * H))

        out = jnp.dot(encoded, out_params)
        out = constrain(out, self.mesh, P("fsdp"))
        return out
