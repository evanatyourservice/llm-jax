import jax
import jax.numpy as jnp


def sine_table(features, length, min_timescale=1.0, max_timescale=10000.0):
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


def rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    x = jnp.concatenate([-x2, x1], axis=-1)
    return x


def apply_rotary_embedding(q, k, cos, sin):
    """Helper function to apply Rotary Embeddings."""
    batch, rep, qheads, qlen, d = q.shape
    kbatch, kheads, klen, kd = k.shape

    qcos = jax.lax.broadcast_in_dim(cos[:qlen, :], (1, 1, 1, qlen, d), (3, 4))
    qsin = jax.lax.broadcast_in_dim(sin[:qlen, :], (1, 1, 1, qlen, d), (3, 4))

    kcos = jax.lax.broadcast_in_dim(cos[:klen, :], (1, 1, klen, d), (2, 3))
    ksin = jax.lax.broadcast_in_dim(sin[:klen, :], (1, 1, klen, d), (2, 3))

    out_q = q * qcos + rotate_half(q) * qsin
    out_k = k * kcos + rotate_half(k) * ksin

    return out_q.astype(q.dtype), out_k.astype(k.dtype)
