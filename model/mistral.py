import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding as NS, Mesh, PartitionSpec as P
import flax.linen as nn

from configs import ModelConfig
from model.rotary_embedding import apply_rotary_embedding, sine_table
from model.jax_attn import dot_product_attention


initializer = nn.initializers.normal(0.02)


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
            "embedding", initializer, (self.vocab_size, self.embed_dim)
        )
        self.final_norm = RMSNorm()

    def encode(self, x: jax.Array) -> jax.Array:
        x = jnp.take(self.embedding, x, axis=0)
        x = constrain(x, self.mesh, P("fsdp"))

        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)

        return x

    def decode(self, x: jax.Array) -> jax.Array:
        x = self.final_norm(x)

        x = jnp.dot(x, self.embedding.T)
        x = constrain(x, self.mesh, P("fsdp"))

        # gemma style soft cap
        x = jnp.tanh(x / 30.0) * 30.0

        return x


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

        q = jnp.dot(x, q_params)
        k = jnp.dot(x, k_params)
        v = jnp.dot(x, v_params)
        q = jnp.reshape(q, (B, T, self.num_heads, self.head_dim))
        k = jnp.reshape(k, (B, T, self.num_kv_heads, self.head_dim))
        v = jnp.reshape(v, (B, T, self.num_kv_heads, self.head_dim))

        sin, cos = sine_table(self.head_dim, T, max_timescale=self.rope_theta)
        q, k = apply_rotary_embedding(q, k, cos, sin, seq_first=True)

        qkv = dot_product_attention(
            q, k, v, is_causal=True, local_window_size=(self.sliding_window_size, 0)
        )

        qkv = jnp.reshape(qkv, (B, T, self.num_heads * self.head_dim))

        out = jnp.dot(qkv, out_params)
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
        gate = nn.silu(gate)

        up = jnp.dot(x, up_kernel)
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
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rope_theta,
            self.sliding_window_size,
            self.mesh,
        )

        attn_in = RMSNorm()(x)
        attn_out = attn_layer(attn_in)
        x = x + attn_out

        mlp_in = RMSNorm()(x)
        mlp_out = MLP(self.hidden_dim, self.mesh)(mlp_in)
        x = x + mlp_out

        if self.use_scan:
            return (x, None)
        return x


class Mistral(nn.Module):
    """Mistral model."""

    config: ModelConfig
    mesh: Mesh
    using_grad_accum: bool = False

    @nn.compact
    def __call__(self, tokens):
        policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims

        # remat if within a scan, otherwise let JAX manage
        if self.using_grad_accum:
            embedder = nn.remat(Embedder, prevent_cse=False, policy=policy)(
                self.config.vocab_size, self.config.num_embeds, self.mesh
            )
        else:
            embedder = Embedder(
                self.config.vocab_size, self.config.num_embeds, self.mesh
            )

        x = embedder.encode(tokens)

        if self.config.scan_layers or self.using_grad_accum:
            BlockModule = nn.remat(Block, prevent_cse=False, policy=policy)
        else:
            BlockModule = Block

        if self.config.scan_layers:
            x, _ = nn.scan(
                BlockModule,
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
                x = BlockModule(
                    self.config.num_heads,
                    self.config.num_kv_heads,
                    self.config.head_dim,
                    self.config.sliding_window_size,
                    self.config.hidden_dim,
                    self.config.rope_theta,
                    self.mesh,
                )(x)

        logits = embedder.decode(x)

        return logits
