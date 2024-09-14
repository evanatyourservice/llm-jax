from functools import partial

import flax
import flax.linen
import jax
import jax.numpy as jnp
import flax.linen as nn

from configs import ModelConfig
from model.rotary_embedding import apply_rotary_embedding, sine_table


initializer = nn.initializers.normal(0.02)


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
        attn_layer = Attention(self.num_heads, self.num_kv_heads, self.head_dim)

        attn_mask = nn.make_causal_mask(x[:1, :, 0], dtype=jnp.bool)

        all_ones = jnp.ones_like(attn_mask, dtype=jnp.bool)
        sliding_mask = jnp.triu(all_ones, -1 * self.sliding_window_size + 1) * jnp.tril(
            all_ones, self.sliding_window_size - 1
        )
        attn_mask = (sliding_mask * attn_mask).astype(jnp.bool)

        attn_in = nn.RMSNorm(epsilon=1e-6)(x)
        x = x + attn_layer(attn_in, attn_mask)
        mlp_in = nn.RMSNorm(epsilon=1e-6)(x)
        x = x + MLP(self.hidden_dim)(mlp_in)

        return x


class Mistral(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, tokens):
        wte = nn.Embed(
            self.config.vocab_size, self.config.num_embeds, embedding_init=initializer
        )

        x = wte(tokens)  # [B, T, num_embeds]

        if self.config.scan_layers:
            x = flax_scan(Block, self.config.num_layers, unroll=self.config.scan_unroll)(
                self.config.num_heads,
                self.config.num_kv_heads,
                self.config.head_dim,
                self.config.sliding_window_size,
                self.config.hidden_dim,
            )(x)
        else:
            for _ in range(self.config.num_layers):
                x = Block(
                    self.config.num_heads,
                    self.config.num_kv_heads,
                    self.config.head_dim,
                    self.config.sliding_window_size,
                    self.config.hidden_dim,
                )(x)

        x = nn.RMSNorm(epsilon=1e-6)(x)

        logits = wte.attend(x)

        # gemma style soft cap
        soft_cap_scaler = jnp.array(30.0, dtype=logits.dtype)
        logits = jnp.tanh(logits / soft_cap_scaler) * soft_cap_scaler

        return logits


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
