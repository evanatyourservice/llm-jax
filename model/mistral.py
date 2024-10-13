import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding as NS, PartitionSpec as P
import flax.linen as nn

from configs import ModelConfig
from model.attention import Attention


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

    def encode(self, x: jax.Array) -> jax.Array:
        x = jnp.take(self.embedding, x, axis=0)
        x = constrain(x, self.mesh, P("fsdp"))
        return x

    def decode(self, x: jax.Array) -> jax.Array:
        x = jnp.dot(x, self.embedding.T)
        x = constrain(x, self.mesh, P("fsdp"))
        return x


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
        remat_policy = None
        if not self.config.remat_everything:
            remat_policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims

        if self.config.remat:
            embedder = nn.remat(
                Embedder, prevent_cse=not self.using_grad_accum, policy=remat_policy
            )(self.config.vocab_size, self.config.num_embeds, self.mesh)
        else:
            embedder = Embedder(
                self.config.vocab_size, self.config.num_embeds, self.mesh
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

        x = RMSNorm()(x)

        logits = embedder.decode(x)

        return logits
