import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P


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


def apply_rotary_embedding(q, k, cos, sin, seq_first=True, mesh=None):
    """Helper function to apply Rotary Embeddings.

    Inputs are (batch, seq, heads, head_dim) if seq_first is True (default),
    else (batch, heads, seq, head_dim).
    """
    assert q.ndim == 4
    assert k.ndim == 4
    if seq_first:
        qlen = q.shape[-3]
        klen = k.shape[-3]
    else:
        qlen = q.shape[-2]
        klen = k.shape[-2]

    qcos = jnp.expand_dims(cos[:qlen, :], range(len(q.shape) - 2))
    qsin = jnp.expand_dims(sin[:qlen, :], range(len(q.shape) - 2))

    kcos = jnp.expand_dims(cos[:klen, :], range(len(k.shape) - 2))
    ksin = jnp.expand_dims(sin[:klen, :], range(len(k.shape) - 2))

    if seq_first:
        qcos = jnp.swapaxes(qcos, -2, -3)
        qsin = jnp.swapaxes(qsin, -2, -3)
        kcos = jnp.swapaxes(kcos, -2, -3)
        ksin = jnp.swapaxes(ksin, -2, -3)

    if mesh is not None:
        qcos = jnp.broadcast_to(qcos, q.shape)
        qsin = jnp.broadcast_to(qsin, q.shape)
        kcos = jnp.broadcast_to(kcos, k.shape)
        ksin = jnp.broadcast_to(ksin, k.shape)

        qcos = jax.lax.with_sharding_constraint(qcos, NamedSharding(mesh, P("fsdp")))
        qsin = jax.lax.with_sharding_constraint(qsin, NamedSharding(mesh, P("fsdp")))
        kcos = jax.lax.with_sharding_constraint(kcos, NamedSharding(mesh, P("fsdp")))
        ksin = jax.lax.with_sharding_constraint(ksin, NamedSharding(mesh, P("fsdp")))

    out_q = q * qcos + rotate_half(q) * qsin
    out_k = k * kcos + rotate_half(k) * ksin

    return out_q.astype(q.dtype), out_k.astype(k.dtype)
