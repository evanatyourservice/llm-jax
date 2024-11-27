import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding as NS, PartitionSpec as P
import flax.linen as nn

from configs import ModelConfig


init_fn = nn.initializers.he_normal()
constrain = lambda x, mesh, spec: jax.lax.with_sharding_constraint(x, NS(mesh, spec))


class RMSNorm(nn.Module):
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x):
        return x * jax.lax.rsqrt(
            jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.epsilon
        )


class Embedder(nn.Module):
    vocab_size: int
    embed_dim: int
    mesh: Mesh

    def setup(self):
        self.embedding = self.param(
            "embedding", init_fn, (self.vocab_size, self.embed_dim)
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
    mesh: Mesh

    @nn.compact
    def __call__(self, x):
        C = x.shape[-1]

        gate_kernel = self.param("gate_kernel", init_fn, (C, self.hidden_dim))
        up_kernel = self.param("up_kernel", init_fn, (C, self.hidden_dim))
        down_kernel = self.param("down_kernel", init_fn, (self.hidden_dim, C))

        gate = jnp.dot(x, gate_kernel)
        gate = nn.silu(gate)

        up = jnp.dot(x, up_kernel)
        x = gate * up

        down = jnp.dot(x, down_kernel)
        if self.mesh is not None:
            down = constrain(down, self.mesh, P("fsdp"))
        return down


class LSTMBlock(nn.Module):
    hidden_dim: int
    mesh: Mesh
    use_scan: bool = False

    @nn.compact
    def __call__(self, x):
        C = x.shape[-1]

        rnn = nn.RNN(
            nn.OptimizedLSTMCell(features=C, kernel_init=init_fn),
            time_major=False,
            unroll=128,
        )

        x += rnn(RMSNorm()(x))
        x += MLP(self.hidden_dim, self.mesh)(RMSNorm()(x))

        if self.use_scan:
            return (x, None)
        return x


class LSTM(nn.Module):
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
            LSTMBlockModule = nn.remat(LSTMBlock, prevent_cse=prevent_cse, policy=remat_policy)
        else:
            LSTMBlockModule = LSTMBlock

        if self.config.scan_layers:
            x, _ = nn.scan(
                LSTMBlockModule,
                variable_axes={True: 0},
                split_rngs={True: True},
                length=self.config.num_layers,
            )(
                self.config.hidden_dim,
                self.mesh,
                use_scan=True,
            )(
                x
            )
        else:
            for _ in range(self.config.num_layers):
                x = LSTMBlockModule(
                    self.config.hidden_dim,
                    self.mesh,
                )(x)

        x = RMSNorm()(x)
        logits = embedder.decode(x)
        return logits

def test_lstm():
    """Simple test to verify LSTM functionality."""
    import numpy as np
    from configs import ModelConfig
    
    config = ModelConfig(
        vocab_size=100,
        num_embeds=32,
        hidden_dim=64,
        num_layers=2,
        remat=False,
        remat_everything=False,
        scan_layers=True,
    )
    
    model = LSTM(config)
    
    batch_size, seq_len = 2, 16
    tokens = jnp.array(np.random.randint(0, config.vocab_size, (batch_size, seq_len)))
    
    key = jax.random.PRNGKey(0)
    params = model.init(key, tokens)
    
    print("\nTest Configuration:")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Model config: {config}")
    
    logits = model.apply(params, tokens)
    
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
    
    print("\nModel Output Statistics:")
    print(f"Logits shape: {logits.shape}")
    print(f"Logits mean: {jnp.mean(logits):.4f}")
    print(f"Logits std: {jnp.std(logits):.4f}")
    print(f"Logits min: {jnp.min(logits):.4f}")
    print(f"Logits max: {jnp.max(logits):.4f}")
    print("\n✓ LSTM test passed!")
    
    return logits

if __name__ == "__main__":
    test_lstm()
