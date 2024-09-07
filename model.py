from typing import Any, Optional, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

from configs import ModelConfig


initializer = nn.initializers.normal()


class RMSNorm(nn.Module):
    """RMSNorm layer."""

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)

        # Jax.lax.rsqrt is used because it returns different floats than
        # jnp.reciprocal(jnp.sqrt(var + 1e-06))
        normed_inputs = x * jax.lax.rsqrt(var + 1e-06)

        # normed_inputs is a rank-K tensor, K > 1 (K is typically 2 or 3). scale is
        # a rank-1 tensor. To avoid implicit rank-promotion, reshape scale to
        # a (1, ..., 1, D) tensor, so the rank of scale matches normed_inputs.
        scale = jnp.expand_dims(scale, axis=range(len(x.shape) - 1))
        normed_inputs = normed_inputs * (1 + scale)
        return normed_inputs


def apply_rope(
    inputs: jax.Array,  # [B, L]
    positions: jax.Array,  # [B, L]
    head_dim: int,
    max_wavelength: int = 10000,
) -> jax.Array:
    """Applies RoPE."""
    fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = max_wavelength**fraction

    sinusoid_inp = positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(inputs.dtype)


class Attention(nn.Module):
    """Attention module."""

    num_heads: int

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        segment_pos: jax.Array,
        attn_mask: jax.Array,
    ) -> jax.Array:
        B, T, C = x.shape
        head_dim = C // self.num_heads

        qkv = nn.Dense(3 * C, use_bias=False, kernel_init=initializer)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = jnp.reshape(q, (B, T, self.num_heads, head_dim))
        k = jnp.reshape(k, (B, T, self.num_heads, head_dim))
        v = jnp.reshape(v, (B, T, self.num_heads, head_dim))

        # normalize qk
        q = RMSNorm()(q)
        k = RMSNorm()(k)

        query_proj = apply_rope(q, segment_pos, head_dim=head_dim)
        query_scaled = query_proj * jax.lax.rsqrt(
            jnp.array(head_dim, dtype=query_proj.dtype)
        )
        key_proj = apply_rope(k, segment_pos, head_dim=head_dim)

        logits = jnp.einsum("...qhd,...khd->...hqk", query_scaled, key_proj)

        padded_logits = jnp.where(
            jnp.expand_dims(attn_mask, -3), logits, jnp.finfo(logits.dtype).min
        )
        probs = jax.nn.softmax(padded_logits, axis=-1)
        encoded = jnp.einsum("...hqk,...khd->...qhd", probs, v)
        encoded = jnp.reshape(encoded, (B, T, C))

        attn_output = nn.Dense(C, use_bias=False, kernel_init=initializer)(encoded)

        return attn_output


class MLP(nn.Module):

    @nn.compact
    def __call__(self, x):
        C = x.shape[-1]
        x = nn.Dense(4 * C, use_bias=False, kernel_init=initializer)(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(C, use_bias=False, kernel_init=initializer)(x)
        return x


class Block(nn.Module):
    num_heads: int

    @nn.compact
    def __call__(self, x, pos, mask):
        attn_layer = Attention(self.num_heads)
        x = x + attn_layer(RMSNorm()(x), pos, mask)
        x = x + MLP()(RMSNorm()(x))
        return x


class GPT(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, tokens, positions, attention_mask):
        wte = nn.Embed(
            self.config.vocab_size,
            self.config.num_embeds,
        )
        wpe = nn.Embed(
            self.config.block_size,
            self.config.num_embeds,
        )

        token_embed = wte(tokens)  # [B, T, num_embeds]
        pos_embed = wpe(positions)  # [B, T, num_embeds]

        x = token_embed + pos_embed
        for i in range(self.config.num_layers):
            x = Block(self.config.num_heads)(x, positions, attention_mask)

        x = RMSNorm()(x)
        logits = wte.attend(x)
        return logits


def convert_hf_params(hf_params: FrozenDict, num_heads, num_embeds) -> FrozenDict:
    params = unfreeze(hf_params["transformer"])
    for k, v in params.pop("h", {}).items():
        params[k] = v

    params = flatten_dict(params, sep=".")
    for k in params.keys():
        # if k.endswith('attn.c_attn.bias'):
        #    params[k] = params[k].reshape(num_heads, -1)
        if k.endswith("attn.c_attn.kernel"):
            # params[k] = params[k].reshape(num_embeds, num_heads, -1)
            params[k] = params[k].T
        elif k.endswith("attn.c_proj.kernel"):
            # params[k] = params[k].reshape(num_heads, -1, num_embeds)
            params[k] = params[k].T
        elif k.split(".")[1] == "mlp" and k.endswith("kernel"):
            params[k] = params[k].T

    params = unflatten_dict({f"params.{k}": v for k, v in params.items()}, sep=".")
    return freeze(params)


def get_pretrained_params(model_type: str) -> Tuple[ModelConfig, FrozenDict]:
    """
    returns config and pretrained parameters from huggingface gpt models
    """
    assert model_type in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
    # only dropout can be overridden see more notes below
    from transformers import FlaxGPT2LMHeadModel

    print("loading weights from pretrained gpt: %s" % model_type)

    config = {
        "gpt2": ModelConfig(num_layers=12, num_heads=12, num_embeds=768),  # 124M params
        "gpt2-medium": ModelConfig(
            num_layers=24, num_heads=16, num_embeds=1024
        ),  # 350M params
        "gpt2-large": ModelConfig(
            num_layers=36, num_heads=20, num_embeds=1280
        ),  # 774M params
        "gpt2-xl": ModelConfig(
            num_layers=48, num_heads=25, num_embeds=1600
        ),  # 1558M params
    }[model_type]

    model_hf = FlaxGPT2LMHeadModel.from_pretrained(model_type)
    hf_params = model_hf.params["transformer"]
    params = convert_hf_params(hf_params, config.num_heads, config.num_embeds)
    return config, params
