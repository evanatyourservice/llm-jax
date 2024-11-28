import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding as NS, PartitionSpec as P
import flax.linen as nn

from configs import ModelConfig
from model.mlstm.mlstm_layer import mLSTMLayer
from model.mlstm.mlstm_cell import RMSNorm


constrain = lambda x, mesh, spec: jax.lax.with_sharding_constraint(x, NS(mesh, spec))


def small_init():
    def init(key, shape, *args):
        std = jnp.sqrt(2.0 / (5.0 * shape[-1]))
        return jax.random.normal(key, shape) * std

    return init


def wang_init(num_blocks: int):
    def init(key, shape, *args):
        std = 2.0 / num_blocks / jnp.sqrt(shape[-1])
        return jax.random.normal(key, shape) * std

    return init


class Embedder(nn.Module):
    vocab_size: int
    embed_dim: int
    mesh: Mesh

    def setup(self):
        self.embedding = self.param(
            "embedding", small_init(), (self.vocab_size, self.embed_dim)
        )

    def encode(self, x):
        x = jnp.take(self.embedding, x, axis=0)
        if self.mesh is not None:
            x = constrain(x, self.mesh, P("fsdp"))
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        x = jnp.dot(x, self.embedding.T)
        if self.mesh is not None:
            x = constrain(x, self.mesh, P("fsdp"))
        x = jnp.tanh(x / 30) * 30
        return x


class MLP(nn.Module):
    hidden_dim: int
    num_blocks: int
    mesh: Mesh

    @nn.compact
    def __call__(self, x):
        C = x.shape[-1]

        gate_kernel = self.param("gate_kernel", small_init(), (C, self.hidden_dim))
        up_kernel = self.param("up_kernel", small_init(), (C, self.hidden_dim))
        down_kernel = self.param("down_kernel", wang_init(self.num_blocks), (self.hidden_dim, C))

        gate = jnp.dot(x, gate_kernel)
        gate = nn.silu(gate)

        up = jnp.dot(x, up_kernel)
        x = gate * up

        down = jnp.dot(x, down_kernel)
        if self.mesh is not None:
            down = constrain(down, self.mesh, P("fsdp"))
        return down


class mLSTMBlock(nn.Module):
    block_size: int
    num_heads: int
    num_blocks: int
    mesh: Mesh
    use_scan: bool = False

    @nn.compact
    def __call__(self, x):
        B, T, C = x.shape

        lstm_out = mLSTMLayer(
            embedding_dim=C,
            hidden_dim=calculate_proj_up_dim(C),
            num_heads=self.num_heads,
            context_length=self.block_size,
        )(RMSNorm()(x))
        lstm_out = constrain(lstm_out, self.mesh, P("fsdp"))
        x += lstm_out
        x += MLP(calculate_proj_up_dim(C), self.num_blocks, self.mesh)(RMSNorm()(x))

        if self.use_scan:
            return (x, None)
        return x


class mLSTM(nn.Module):
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
            mLSTMBlockModule = nn.remat(
                mLSTMBlock, prevent_cse=prevent_cse, policy=remat_policy
            )
        else:
            mLSTMBlockModule = mLSTMBlock

        if self.config.scan_layers:
            x, _ = nn.scan(
                mLSTMBlockModule,
                variable_axes={True: 0},
                split_rngs={True: True},
                length=self.config.num_layers,
            )(
                self.config.block_size,
                self.config.num_heads,
                self.config.num_layers,
                self.mesh,
                use_scan=True,
            )(
                x
            )
        else:
            for _ in range(self.config.num_layers):
                x = mLSTMBlockModule(
                    self.config.block_size,
                    self.config.num_heads,
                    self.config.num_layers,
                    self.mesh,
                )(x)

        x = RMSNorm()(x)
        logits = embedder.decode(x)
        return logits


def calculate_proj_up_dim(
    embedding_dim: int,
    proj_factor: float = 1.3,
    round_up: bool = True,
    multiple_of: int = 64,
) -> int:
    proj_up_dim = proj_factor * embedding_dim
    multiple_of_multiplier = proj_up_dim / multiple_of
    
    if round_up:
        multiple_of_multiplier = np.ceil(multiple_of_multiplier)
    else:
        multiple_of_multiplier = np.floor(multiple_of_multiplier)
        
    return int(multiple_of_multiplier * multiple_of)
