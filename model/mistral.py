from functools import partial

import flax
import flax.linen
import jax
import jax.numpy as jnp
import flax.linen as nn

from configs import ModelConfig
from model.rotary_embedding import apply_rotary_embedding, sine_table


initializer = nn.initializers.normal(0.02)


class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "embedding", initializer, (self.vocab_size, self.embed_dim)
        )

    def encode(self, x: jax.Array) -> jax.Array:
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x: jax.Array) -> jax.Array:
        return jnp.dot(x, self.input_embedding_table.T)


class RMSNorm(nn.Module):
    """RMSNorm layer.

    Upcasts to float32 and back."""

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))

        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = x * jax.lax.rsqrt(var + 1e-06)
        normed_inputs = normed_inputs.astype(x.dtype)

        scale = jnp.expand_dims(scale, axis=range(len(x.shape) - 1))
        normed_inputs = normed_inputs * (1 + scale)

        return normed_inputs


class Attention(nn.Module):
    """Multi-head attention with RoPE and GQA.

    Upcasts to float32 and back for softmax."""

    num_heads: int
    num_kv_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, x, mask):
        B, T, C = x.shape

        q_params = self.param(
            "q_kernel",
            initializer,
            (C, self.num_heads // self.num_kv_heads, self.num_kv_heads, self.head_dim),
        )
        k_params = self.param(
            "k_kernel", initializer, (C, self.num_kv_heads, self.head_dim)
        )
        v_params = self.param(
            "v_kernel", initializer, (C, self.num_kv_heads, self.head_dim)
        )
        out_params = self.param(
            "out_kernel",
            initializer,
            (self.num_heads // self.num_kv_heads, self.num_kv_heads, self.head_dim, C),
        )

        q = jnp.einsum("bsm,mrhk->brhsk", x, q_params)
        k = jnp.einsum("bdm,mhk->bhdk", x, k_params)
        v = jnp.einsum("bdm,mhv->bhdv", x, v_params)

        sin, cos = sine_table(self.head_dim, T, max_timescale=1000000.0)
        q, k = apply_rotary_embedding(q, k, cos, sin)

        scale = jax.lax.rsqrt(jnp.array(self.head_dim, dtype=x.dtype))
        qk = jnp.einsum("brhsk,bhdk->brhsd", q, k) * scale

        qk = jnp.tanh(qk / 50) * 50  # gemma style soft cap
        mask = jnp.expand_dims(mask, axis=1)
        qk = jax.nn.softmax(qk.astype(jnp.float32), where=mask).astype(x.dtype)
        qkv = jnp.einsum("brhsd,bhdv->brhsv", qk, v)

        out = jnp.einsum("brhsv,rhvm->bsm", qkv, out_params)

        return out


class MLP(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        C = x.shape[-1]
        gate = nn.Dense(self.hidden_dim, use_bias=False, kernel_init=initializer)(x)
        gate = nn.silu(gate)

        x = nn.Dense(self.hidden_dim, use_bias=False, kernel_init=initializer)(x)
        x = x * gate

        x = nn.Dense(C, use_bias=False, kernel_init=initializer)(x)
        return x


def excess_kurtosis(emb):
    mean = jnp.mean(emb, axis=-1, keepdims=True)
    std = jnp.std(emb, axis=-1, keepdims=True)
    centralized = emb - mean
    fourth_moment = jnp.mean(centralized**4, axis=-1, keepdims=True)
    kurtosis = jnp.squeeze(fourth_moment / (std**4 + 1e-6), axis=-1)
    kurtosis = kurtosis.reshape(-1) - 3
    kurtosis = jnp.maximum(kurtosis, 0.0)
    return jnp.sum(kurtosis)


class Block(nn.Module):
    num_heads: int
    num_kv_heads: int
    head_dim: int
    sliding_window_size: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x, kurtosis_sum = x

        attn_layer = Attention(self.num_heads, self.num_kv_heads, self.head_dim)

        attn_mask = nn.make_causal_mask(x[:1, :, 0], dtype=jnp.bool)

        all_ones = jnp.ones_like(attn_mask)
        sliding_mask = jnp.triu(all_ones, -1 * self.sliding_window_size + 1) * jnp.tril(
            all_ones, self.sliding_window_size - 1
        )
        attn_mask = sliding_mask * attn_mask

        attn_in = RMSNorm()(x)
        x = x + attn_layer(attn_in, attn_mask)
        mlp_in = RMSNorm()(x)
        x = x + MLP(self.hidden_dim)(mlp_in)

        return x, kurtosis_sum + excess_kurtosis(x)


class Mistral(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, tokens):
        wte = Embedder(self.config.vocab_size, self.config.num_embeds)

        x = wte.encode(tokens)  # [B, T, num_embeds]

        kurtosis_sum = jnp.array(0.0)

        if self.config.scan_layers:
            x, kurtosis_sum = flax_scan(Block, self.config.num_layers, unroll=2)(
                self.config.num_heads,
                self.config.num_kv_heads,
                self.config.head_dim,
                self.config.sliding_window_size,
                self.config.hidden_dim,
            )((x, kurtosis_sum))
        else:
            for _ in range(self.config.num_layers):
                x, kurtosis_sum = Block(
                    self.config.num_heads,
                    self.config.num_kv_heads,
                    self.config.head_dim,
                    self.config.sliding_window_size,
                    self.config.hidden_dim,
                )((x, kurtosis_sum))

        x = RMSNorm()(x)

        logits = wte.decode(x)

        # gemma style soft cap
        soft_cap_scaler = jnp.array(30.0, dtype=logits.dtype)
        logits = jnp.tanh(logits / soft_cap_scaler) * soft_cap_scaler

        return logits, kurtosis_sum


def _flax_scan(
    body_fn,
    length: int,
    variable_broadcast=False,
    variable_carry=False,
    variable_axes={True: 0},
    split_rngs={True: True},
    unroll: int = 1,
):
    scan_fn = partial(
        flax.core.lift.scan,
        variable_broadcast=variable_broadcast,
        variable_carry=variable_carry,
        variable_axes=variable_axes,
        split_rngs=split_rngs,
        unroll=unroll,
    )

    def wrapper(scope, carry):
        return body_fn(scope, carry), None

    fn = lambda scope, c: scan_fn(wrapper, length=length)(scope, c)[0]

    return fn


def flax_scan(
    target,
    length: int,
    variable_broadcast=False,
    variable_carry=False,
    variable_axes={True: 0},
    split_rngs={True: True},
    unroll: int = 1,
):
    return nn.transforms.lift_transform(
        _flax_scan,
        target,
        length=length,
        variable_broadcast=variable_broadcast,
        variable_carry=variable_carry,
        variable_axes=variable_axes,
        split_rngs=split_rngs,
        unroll=unroll,
    )
