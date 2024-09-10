import enum
from functools import partial
from typing import Any, Optional, Tuple
import flax
import flax.linen
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


class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding", initializer, (self.vocab_size, self.embed_dim)
        )

    def encode(self, x: jax.Array) -> jax.Array:
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x: jax.Array) -> jax.Array:
        return jnp.dot(x, self.input_embedding_table.T)


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
    num_heads: int

    @nn.compact
    def __call__(self, x, mask):
        B, T, C = x.shape
        assert C % self.num_heads == 0
        head_dim = C // self.num_heads

        qkv = nn.Dense(3 * C, use_bias=False)(x)
        qkv = qkv.reshape(B, T, 3 * self.num_heads, head_dim)
        q, k, v = jnp.split(qkv, 3, axis=2)

        q = apply_rope(q, jnp.arange(T)[None, :], head_dim)
        k = apply_rope(k, jnp.arange(T)[None, :], head_dim)

        # calculate attention matrix
        scale = jnp.reciprocal(jnp.sqrt(head_dim).astype(x.dtype))
        # attn weight shape is (batch..., num_heads, q_length, kv_length)
        attn = jnp.einsum("...qhd,...khd->...hqk", q, k) * scale

        # gemma style soft cap
        attn = jnp.tanh(attn / 50) * 50

        # mask out attention to future tokens
        attn = jnp.where(mask, attn, jnp.finfo(x.dtype).min)
        attn = jax.nn.softmax(attn).astype(x.dtype)

        # return weighted sum over values for each query position
        x = jnp.einsum("...hqk,...khd->...qhd", attn, v).reshape(B, T, C)
        x = nn.Dense(C, use_bias=False)(x)

        return x


class MLP(nn.Module):

    @nn.compact
    def __call__(self, x):
        C = x.shape[-1]
        x = nn.Dense(4 * C, use_bias=False, kernel_init=initializer)(x)
        x = nn.gelu(x)
        x = nn.Dense(C, use_bias=False, kernel_init=initializer)(x)
        return x


class Block(nn.Module):
    num_heads: int

    @nn.compact
    def __call__(self, x):
        attn_layer = Attention(self.num_heads)

        with jax.ensure_compile_time_eval():
            attn_mask = nn.make_causal_mask(x[:, :, 0], dtype=bool)

        x = x + attn_layer(RMSNorm()(x), attn_mask)
        x = x + MLP()(RMSNorm()(x))
        return x


class GPT(nn.Module):
    config: ModelConfig

    @nn.checkpoint
    @nn.compact
    def __call__(self, tokens):
        wte = Embedder(
            self.config.vocab_size,
            self.config.num_embeds,
        )

        x = wte.encode(tokens)  # [B, T, num_embeds]

        x = flax_scan(Block, length=self.config.num_layers, unroll=1)(
            self.config.num_heads
        )(x)

        x = RMSNorm()(x)
        logits = wte.decode(x)
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
