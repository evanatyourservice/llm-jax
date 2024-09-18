import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding as NS, Mesh, PartitionSpec as P
import flax.linen as nn

from configs import ModelConfig
from model.rotary_embedding import apply_rotary_embedding, sine_table


initializer = nn.initializers.normal(0.02)


constrain = lambda x, mesh, spec: jax.lax.with_sharding_constraint(x, NS(mesh, spec))


class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int
    mesh: Mesh
    def setup(self):
        self.embedding_table = self.param(
            "embedding", initializer, (self.vocab_size, self.embed_dim)
        )

    def encode(self, x: jax.Array) -> jax.Array:
        x = self.embedding_table[(x,)]
        x = constrain(x, self.mesh, P("fsdp"))
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x: jax.Array) -> jax.Array:
        x = jnp.dot(x, self.embedding_table.T)
        x = constrain(x, self.mesh, P("fsdp"))
        return x


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


class Attention(nn.Module):
    """Multi-head attention with RoPE and GQA.

    Upcasts to float32 and back for softmax."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float
    mesh: Mesh

    @nn.compact
    def __call__(self, x, mask):
        _, T, C = x.shape

        q_params = self.param(
            "q_kernel", initializer, (C, self.num_heads * self.head_dim)
        )
        k_params = self.param(
            "k_kernel", initializer, (C, self.num_kv_heads * self.head_dim)
        )
        v_params = self.param(
            "v_kernel", initializer, (C, self.num_kv_heads * self.head_dim)
        )
        out_params = self.param(
            "out_kernel", initializer, (self.num_heads * self.head_dim, C)
        )

        q_params = jnp.reshape(
            q_params,
            (C, self.num_heads // self.num_kv_heads, self.num_kv_heads, self.head_dim),
        )
        k_params = jnp.reshape(k_params, (C, self.num_kv_heads, self.head_dim))
        v_params = jnp.reshape(v_params, (C, self.num_kv_heads, self.head_dim))
        out_params = jnp.reshape(
            out_params,
            (self.num_heads // self.num_kv_heads, self.num_kv_heads, self.head_dim, C),
        )
        q_params = constrain(q_params, self.mesh, P("fsdp"))
        k_params = constrain(k_params, self.mesh, P("fsdp"))
        v_params = constrain(v_params, self.mesh, P("fsdp"))
        out_params = constrain(out_params, self.mesh, P(None, None, None, "fsdp"))

        q = jnp.einsum("bsm,mrhk->brhsk", x, q_params)
        k = jnp.einsum("bdm,mhk->bhdk", x, k_params)
        v = jnp.einsum("bdm,mhv->bhdv", x, v_params)
        q = constrain(q, self.mesh, P("fsdp"))
        k = constrain(k, self.mesh, P("fsdp"))
        v = constrain(v, self.mesh, P("fsdp"))

        sin, cos = sine_table(self.head_dim, T, max_timescale=self.rope_theta)
        q, k = apply_rotary_embedding(q, k, cos, sin)
        q = constrain(q, self.mesh, P("fsdp"))
        k = constrain(k, self.mesh, P("fsdp"))

        scale = jax.lax.rsqrt(jnp.array(self.head_dim, dtype=x.dtype))
        qk = jnp.einsum("brhsk,bhdk->brhsd", q, k) * scale
        qk = constrain(qk, self.mesh, P("fsdp"))

        qk = jnp.tanh(qk / 50) * 50  # gemma style soft cap

        mask = jnp.expand_dims(mask, axis=1)
        qk = jax.nn.softmax(qk.astype(jnp.float32), where=mask).astype(x.dtype)
        qk = constrain(qk, self.mesh, P("fsdp"))

        qkv = jnp.einsum("brhsd,bhdv->brhsv", qk, v)
        qkv = constrain(qkv, self.mesh, P("fsdp"))

        out = jnp.einsum("brhsv,rhvm->bsm", qkv, out_params)
        out = constrain(out, self.mesh, P("fsdp"))
        return out


class MLP(nn.Module):
    hidden_dim: int
    mesh: Mesh

    @nn.compact
    def __call__(self, x):
        C = x.shape[-1]

        gate_kernel = self.param("gate_kernel", initializer, (C, self.hidden_dim))
        up_kernel = self.param("up_kernel", initializer, (C, self.hidden_dim))
        down_kernel = self.param("down_kernel", initializer, (self.hidden_dim, C))

        gate = jnp.dot(x, gate_kernel)
        gate = constrain(gate, self.mesh, P("fsdp"))
        gate = nn.silu(gate)

        up = jnp.dot(x, up_kernel)
        up = constrain(up, self.mesh, P("fsdp"))
        x = gate * up

        down = jnp.dot(x, down_kernel)
        down = constrain(down, self.mesh, P("fsdp"))
        return down


class Block(nn.Module):
    """Transformer block."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    sliding_window_size: int
    hidden_dim: int
    rope_theta: float
    mesh: Mesh
    use_scan: bool = False

    @nn.compact
    def __call__(self, x):
        attn_layer = Attention(
            self.num_heads, self.num_kv_heads, self.head_dim, self.rope_theta, self.mesh
        )

        attn_mask = nn.make_causal_mask(x[:1, :, 0], dtype=jnp.bool)
        all_ones = jnp.ones_like(attn_mask, dtype=jnp.bool)
        sliding_mask = jnp.triu(all_ones, -1 * self.sliding_window_size + 1) * jnp.tril(
            all_ones, self.sliding_window_size - 1
        )
        attn_mask = (sliding_mask * attn_mask).astype(jnp.bool)

        attn_in = RMSNorm()(x)
        attn_out = attn_layer(attn_in, attn_mask)
        x = x + attn_out

        mlp_in = RMSNorm()(x)
        mlp_out = MLP(self.hidden_dim, self.mesh)(mlp_in)
        x = x + mlp_out

        x = constrain(x, self.mesh, P("fsdp"))

        if self.use_scan:
            return (x, None)
        return x


class Mistral(nn.Module):
    """Mistral model."""

    config: ModelConfig
    mesh: Mesh

    @nn.compact
    def __call__(self, tokens):
        embedder = nn.remat(Embedder)(
            self.config.vocab_size, self.config.num_embeds, self.mesh
        )

        x = embedder.encode(tokens)

        RemattedBlock = nn.remat(Block, prevent_cse=not self.config.scan_layers)

        if self.config.scan_layers:
            x, _ = nn.scan(
                RemattedBlock,
                variable_axes={True: 0},
                split_rngs={"params": True},
                length=self.config.num_layers,
                unroll=self.config.scan_unroll,
            )(
                self.config.num_heads,
                self.config.num_kv_heads,
                self.config.head_dim,
                self.config.sliding_window_size,
                self.config.hidden_dim,
                self.config.rope_theta,
                self.mesh,
                use_scan=True,
            )(
                x
            )
        else:
            for _ in range(self.config.num_layers):
                x = RemattedBlock(
                    self.config.num_heads,
                    self.config.num_kv_heads,
                    self.config.head_dim,
                    self.config.sliding_window_size,
                    self.config.hidden_dim,
                    self.config.rope_theta,
                    self.mesh,
                )(x)

        x = RMSNorm()(x)

        logits = embedder.decode(x)

        # gemma style soft cap
        logits = jnp.tanh(logits / 30) * 30

        return logits
