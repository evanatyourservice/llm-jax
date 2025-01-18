from typing import Optional
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding as NS, PartitionSpec as P
import flax.linen as nn

from configs import ModelConfig


init_fn = lambda dim: nn.initializers.normal(jnp.sqrt(2 / (5 * dim)))
wang_fn = lambda dim, n_layers: nn.initializers.normal(2 / n_layers / jnp.sqrt(dim))
constrain = lambda x, mesh, spec: jax.lax.with_sharding_constraint(x, NS(mesh, spec))


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


class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int
    mesh: Mesh

    def setup(self):
        self.embedding = self.param(
            "embedding", init_fn(self.embed_dim), (self.vocab_size, self.embed_dim)
        )

    def encode(self, x: jax.Array) -> jax.Array:
        x = jnp.take(self.embedding, x, axis=0)
        if self.mesh is not None:
            x = constrain(x, self.mesh, P("fsdp"))
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x: jax.Array) -> jax.Array:
        x = jnp.dot(x, self.embedding.T)
        if self.mesh is not None:
            x = constrain(x, self.mesh, P("fsdp"))
        x = jnp.tanh(x / 30) * 30
        return x


def _get_large_negative(dtype):
    dtype_max = jnp.finfo(dtype).max
    return jnp.asarray(-0.7 * dtype_max, dtype=dtype)


def _get_causal_mask(T, S):
    mask = jnp.tril(jnp.ones((T, S), dtype=jnp.bool_))
    return mask[None, None, :, :]


def _dot_product_attention_core(query, key, value):
    head_dim = query.shape[-1]
    query *= jax.lax.rsqrt(jnp.array(head_dim, dtype=jnp.float32)).astype(query.dtype)
    logits = jnp.einsum("BTNH,BSNH->BNTS", query, key)
    logits = jnp.tanh(logits / 50) * 50
    causal_mask = _get_causal_mask(logits.shape[-2], logits.shape[-1])
    logits = jnp.where(causal_mask, logits, _get_large_negative(logits.dtype))
    probs = jax.nn.softmax(logits.astype(jnp.float32)).astype(logits.dtype)
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
    # come in as (B, T, K, G, H) and (B, T, K, H)
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

    # done in float32
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
    n_layers: int
    mesh: Mesh

    @nn.compact
    def __call__(self, x):
        B, T, C = x.shape
        N = self.num_heads
        K = self.num_kv_heads
        G = N // K
        H = self.head_dim

        q_params = self.param("q_kernel", init_fn(C), (C, N * H))
        k_params = self.param("k_kernel", init_fn(C), (C, K * H))
        v_params = self.param("v_kernel", init_fn(C), (C, K * H))
        out_params = self.param("out_kernel", wang_fn(N * H, self.n_layers), (N * H, C))

        q = jnp.dot(x, q_params)
        k = jnp.dot(x, k_params)
        v = jnp.dot(x, v_params)

        q = jnp.reshape(q, (B, T, K, G, H))
        k = jnp.reshape(k, (B, T, K, H))
        v = jnp.reshape(v, (B, T, K, H))

        sin, cos = _sine_table(H, T, max_timescale=self.rope_theta)
        q, k = _apply_rotary_embedding(q, k, cos, sin)

        vmapped_fn = jax.vmap(
            _dot_product_attention_core, in_axes=(3, None, None), out_axes=3
        )
        encoded = vmapped_fn(q, k, v)
        encoded = jnp.reshape(encoded, (B, T, N * H))
        out = jnp.dot(encoded, out_params)
        if self.mesh is not None:
            out = constrain(out, self.mesh, P("fsdp"))
        return out


class MLP(nn.Module):
    hidden_dim: int
    n_layers: int
    mesh: Mesh

    @nn.compact
    def __call__(self, x):
        C = x.shape[-1]

        gate_kernel = self.param("gate_kernel", init_fn(C), (C, self.hidden_dim))
        up_kernel = self.param("up_kernel", init_fn(C), (C, self.hidden_dim))
        down_kernel = self.param(
            "down_kernel", wang_fn(self.hidden_dim, self.n_layers), (self.hidden_dim, C)
        )

        gate = jnp.dot(x, gate_kernel)
        gate = nn.silu(gate)

        up = jnp.dot(x, up_kernel)
        x = gate * up

        down = jnp.dot(x, down_kernel)
        if self.mesh is not None:
            down = constrain(down, self.mesh, P("fsdp"))
        return down


class Block(nn.Module):
    """Transformer block."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_dim: int
    rope_theta: float
    n_layers: int
    mesh: Mesh
    use_scan: bool = False

    @nn.compact
    def __call__(self, x):
        attn_layer = Attention(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rope_theta,
            self.n_layers,
            self.mesh,
        )
        x += attn_layer(RMSNorm()(x))
        x += MLP(self.hidden_dim, self.n_layers, self.mesh)(RMSNorm()(x))
        if self.use_scan:
            return (x, None)
        return x


class Transformer(nn.Module):
    config: ModelConfig
    mesh: Mesh = None
    using_grad_accum: bool = False

    @nn.compact
    def __call__(self, tokens):
        remat_policy = None
        if not self.config.remat_everything:
            remat_policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims

        if self.config.remat:
            embedder = nn.remat(
                Embedder, prevent_cse=not self.using_grad_accum, policy=remat_policy
            )(
                self.config.vocab_size,
                self.config.num_embeds,
                self.mesh,
            )
        else:
            embedder = Embedder(
                self.config.vocab_size,
                self.config.num_embeds,
                self.mesh,
            )

        x = embedder.encode(tokens)

        if self.config.remat:
            prevent_cse = True
            if self.using_grad_accum or self.config.scan_layers:
                prevent_cse = False
            BlockModule = nn.remat(Block, prevent_cse=prevent_cse, policy=remat_policy)
        else:
            BlockModule = Block

        if self.config.scan_layers:
            x, _ = nn.scan(
                BlockModule,
                variable_axes={True: 0},
                split_rngs={True: True},
                length=self.config.num_layers,
            )(
                self.config.num_heads,
                self.config.num_kv_heads,
                self.config.head_dim,
                self.config.hidden_dim,
                self.config.rope_theta,
                self.config.num_layers,
                self.mesh,
                use_scan=True,
            )(
                x
            )
        else:
            for _ in range(self.config.num_layers):
                x = BlockModule(
                    self.config.num_heads,
                    self.config.num_kv_heads,
                    self.config.head_dim,
                    self.config.hidden_dim,
                    self.config.rope_theta,
                    self.config.num_layers,
                    self.mesh,
                )(x)

        x = RMSNorm()(x)
        logits = embedder.decode(x)
        return logits
