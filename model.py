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


class Einsum(nn.Module):
    """Einsum is a convenience module for parameterized tensor multiplication."""

    shape: tuple[int, ...]

    @nn.compact
    def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
        w = self.param("w", initializer, self.shape)
        return jnp.einsum(eqn, x, w)


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


LayerCache = dict[str, jax.Array]


class AttentionType(enum.Enum):
    GLOBAL = 1
    LOCAL_SLIDING = 2


class Attention(nn.Module):
    """Attention module."""

    num_heads: int
    num_kv_heads: int
    features: int
    head_dim: int
    attn_type: AttentionType = AttentionType.GLOBAL
    attn_logits_soft_cap: float | None = 50.0
    sliding_window_size: int | None = None

    @property
    def use_qkv_einsum(self):
        return self.num_kv_heads == self.num_heads

    @property
    def use_gqa(self):
        return self.num_kv_heads != self.num_heads and self.num_kv_heads > 1

    def setup(self):
        self.attn_vec_einsum = Einsum(
            shape=(self.num_heads, self.head_dim, self.features)
        )

        if self.use_qkv_einsum:
            self.qkv_einsum = Einsum(
                shape=(3, self.num_heads, self.features, self.head_dim)
            )
        else:
            self.q_einsum = Einsum(
                shape=(self.num_heads, self.features, self.head_dim)
            )
            self.kv_einsum = Einsum(
                shape=(2, self.num_kv_heads, self.features, self.head_dim)
            )

    def __call__(
        self,
        x: jax.Array,
        segment_pos: jax.Array,
        cache: LayerCache | None,
        attn_mask: jax.Array,
    ) -> tuple[LayerCache | None, jax.Array]:
        seq_len = x.shape[1]

        if self.use_qkv_einsum:
            query_proj, key_proj, value_proj = self.qkv_einsum("BTD,SNDH->SBTNH", x)
        else:
            query_proj = self.q_einsum("BTD,NDH->BTNH", x)
            key_proj, value_proj = self.kv_einsum("BSD,CKDH->CBSKH", x)

        query_proj = apply_rope(
            query_proj, segment_pos, head_dim=self.head_dim
        )
        query_scaled = query_proj * jnp.reciprocal(jnp.sqrt(self.head_dim))
        key_proj = apply_rope(
            key_proj, segment_pos, head_dim=self.head_dim
        )

        # Cache is left aligned.
        if cache is not None:
            end_index = cache["end_index"][0]
            slice_indices = (0, end_index % cache["v"].shape[1], 0, 0)
            value_proj = jax.lax.dynamic_update_slice(
                cache["v"], value_proj, slice_indices
            )
            key_proj = jax.lax.dynamic_update_slice(cache["k"], key_proj, slice_indices)

        if self.use_gqa:
            # Reshape matrices to enable einsums over groups.
            b, t, kg, h = query_scaled.shape
            query_scaled = query_scaled.reshape(
                (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
            )
            logits = jnp.einsum("BTKGH,BSKH->BTKGS", query_scaled, key_proj)
            b, t, k, g, s = logits.shape
            logits = logits.reshape((b, t, k * g, s))
        else:
            logits = jnp.einsum("BTNH,BSNH->BTNS", query_scaled, key_proj)

        if self.attn_logits_soft_cap is not None:
            logits = jnp.tanh(logits / self.attn_logits_soft_cap)
            logits = logits * self.attn_logits_soft_cap

        if self.attn_type == AttentionType.LOCAL_SLIDING:
            if self.sliding_window_size is None:
                raise ValueError(
                    "Sliding_window_size must be set if Local Sliding attention type"
                )

            all_ones = jnp.ones_like(attn_mask)
            sliding_mask = jnp.triu(
                all_ones, -1 * self.sliding_window_size + 1
            ) * jnp.tril(all_ones, self.sliding_window_size - 1)
            attn_mask = sliding_mask * attn_mask

        padded_logits = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, jnp.finfo(logits.dtype).min)
        probs = jax.nn.softmax(padded_logits, axis=-1).astype(key_proj.dtype)
        if self.use_gqa:
            # Reshape matrices to enable einsums over groups.
            b, t, kg, h = probs.shape
            probs = probs.reshape(
                (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
            )
            encoded = jnp.einsum("BTKGS,BSKH->BTKGH", probs, value_proj)
            b, t, k, g, h = encoded.shape
            encoded = encoded.reshape((b, t, k * g, h))
        else:
            encoded = jnp.einsum("BTNS,BSNH->BTNH", probs, value_proj)
        attn_output = self.attn_vec_einsum("BTNH,NHD->BTD", encoded)

        if cache is not None:
            new_cache = {
                "v": value_proj,
                "k": key_proj,
                "end_index": cache["end_index"] + seq_len,
            }
        else:
            new_cache = None

        return new_cache, attn_output

    @classmethod
    def init_cache(
        cls,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        batch_size: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> LayerCache:
        del cls  # not used
        return {
            "v": jnp.zeros((batch_size, cache_size, num_heads, head_dim), dtype=dtype),
            "k": jnp.zeros((batch_size, cache_size, num_heads, head_dim), dtype=dtype),
            "end_index": jnp.zeros((batch_size,), dtype=jnp.int32),
        }


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
        x, pos, mask = x

        attn_layer = Attention(
            self.num_heads,
            self.num_heads,
            x.shape[-1],
            x.shape[-1] // self.num_heads,
        )

        x = x + attn_layer(RMSNorm()(x), pos, None, mask)[1]
        x = x + MLP()(RMSNorm()(x))
        return (x, pos, mask)


class GPT(nn.Module):
    config: ModelConfig

    @nn.checkpoint
    @nn.compact
    def __call__(self, tokens, positions, attention_mask):
        wte = nn.Embed(
            self.config.vocab_size,
            self.config.num_embeds,
        )

        x = wte(tokens)  # [B, T, num_embeds]

        x = flax_scan(Block, length=self.config.num_layers, unroll=1)(
            self.config.num_heads
        )((x, positions, attention_mask))[0]

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
