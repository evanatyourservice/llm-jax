import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.sharding import Mesh, NamedSharding as NS, PartitionSpec as P


constrain = lambda x, mesh, spec: jax.lax.with_sharding_constraint(x, NS(mesh, spec))


def fast_conv_complex(a: jax.Array, b: jax.Array) -> jax.Array:
    """
    Inputs:
        a: array with shape [len1, n] representing len1 n x n diagonal matrices.
        b: another jax array with shape [batch, len2, n].

    Outputs:
        c = conv(a, b) with shape [batch, len1 + len2 - 1, n].
    """
    len1, _ = a.shape
    _, len2, _ = b.shape
    T = 2**int(np.ceil(np.log2(len1 + len2 - 1)))
    A = jax.numpy.fft.fft(a, n=T, axis=0)
    B = jax.numpy.fft.fft(b, n=T, axis=1)
    C = A * B
    c = jax.numpy.fft.ifft(C, n=T, axis=1)
    return c[:, :(len1 + len2 - 1)]


class ComplexStateSpaceModel(nn.Module):
    """
    JAX Flax module for Complex State SSM.

    It returns a fully trainable complex state SSM defined as:

        x_t = A @ x_{t-1} + B @ u_t,
        y_t = C @ x_t + D @ u_t + b,

    where:

        u_t, 1 <= t <= T, are the sequences of (real) inputs,
        x_t, 1 <= t <= T, are the sequence of (complex) states,
        y_t, 1 <= t <= T, are the sequence of (real) outputs,
        matrices A (diagonal complex), B (complex) and C (complex) are mandatory,
        matrix D (real) and bias b (real) are optional,
        and x_0 is the initial (complex) state.

    Note that we use the tranposed version of these equations in the Python code
    by following the 'row major' convention.
    """
    input_size: int
    state_size: int
    output_size: int
    has_matrixD: bool = False
    has_bias: bool = False
    resample_up: int = 1
    resample_down: int = 1
    init_scale_A: float = 1 - 1e-4
    mesh: Mesh = None

    @nn.compact
    def __call__(self, u: jax.Array, x0: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
        """
        Inputs:
            u: the real input array with shape [batch, length, input_size].
            x0: the complex initial state with shape [batch, state_size].

        Outputs:
            y: the real output array with shape [batch, resample_up*length//resample_down, output_size].
            x0: the complex final state with shape [batch, state_size].
        """
        _, length, _ = u.shape

        if self.resample_up > 1:
            u = jnp.repeat(u, self.resample_up, axis=1)
            _, length, _ = u.shape

        A_angles = self.param('A_angles', nn.initializers.uniform(2*jnp.pi), (self.state_size,))
        A = self.init_scale_A * jnp.exp(1j * A_angles)
        B = self.param('B', nn.initializers.normal(stddev=1.0 / jnp.sqrt(self.state_size + self.input_size)), 
                      (self.input_size, self.state_size), dtype=jnp.complex64)
        C = self.param('C', nn.initializers.normal(stddev=1.0 / jnp.sqrt(self.state_size), dtype=jnp.complex64), 
                      (self.state_size, self.output_size), dtype=jnp.complex64)
        D = self.param('D', nn.initializers.zeros_init(), (self.input_size, self.output_size)) if self.has_matrixD else None
        b = self.param('b', nn.initializers.zeros_init(), (self.output_size,)) if self.has_bias else None

        Aps = jnp.power(A, jnp.arange(length + 1)[:, None])
        uB = u.astype(jnp.complex64) @ B  # transpose of math eq B*u
        x = fast_conv_complex(Aps[:-1], uB)[:, :length]
        if x0 is not None:
            x = x + Aps[1:] * x0[:,None,:]

        if self.resample_down > 1:
            x = x[:, self.resample_down-1::self.resample_down]
            if self.has_matrixD:
                u = u[:, self.resample_down-1::self.resample_down]

        y = jnp.real(x @ C)  # transpose of math eq C*x
        if self.has_matrixD:
            y = y + u @ D  # transpose of math eq D*u
        if self.has_bias:
            y = y + b
        
        if self.mesh is not None:
            y = constrain(y, self.mesh, P("fsdp"))
            x = constrain(x, self.mesh, P("fsdp"))
        return y, x[:, -1]


def fast_conv_real(a: jax.Array, b: jax.Array) -> jax.Array:
    """
    Inputs:
        a: array with shape [len1, K, n, n] representing len1 block diagonal matrices,
           each block has size n x n, and a total of K such blocks.
        b: another jax array with shape [batch, len2, K, n] representing K vectors,
           and each has length n.

    Outputs:
        c = conv(b, a) with shape [batch, len1 + len2 - 1, K, n].
        Following the 'row major' convention, here we take the vector-matrix product (b x a).
    """
    len1, _, _, _ = a.shape
    _, len2, _, _ = b.shape
    T = 2**int(np.ceil(np.log2(len1 + len2 - 1)))
    A = jax.numpy.fft.rfft(a, n=T, axis=0)
    B = jax.numpy.fft.rfft(b, n=T, axis=1)
    C = jnp.einsum("blki, lkij->blkj", B, A)
    c = jax.numpy.fft.irfft(C, n=T, axis=1)
    return c[:, :(len1 + len2 - 1)]


def A_powers(A: jax.Array, t: int) -> jax.Array:
    """
    Inputs:
        A: array with shape [K, n, n] representing a block diagonal matrices,
           each block has size n x n, and a total of K such blocks.
        t: a positive integer.

    Outputs:
        jnp.stack([I, A, A^2, ..., A^(t - 1)], axis=0),
        a array with shape [t, K, n, n].
    """
    K, n, _ = A.shape
    eye = jnp.eye(n, dtype=A.dtype)
    identity = jnp.repeat(jnp.expand_dims(eye, axis=0), K, axis=0)
    result = jnp.stack([identity])
    lift = A
    
    while len(result) < t:
        result_lift = jnp.einsum('lkij,kjm->lkim', result, lift)
        result = jnp.concatenate([result, result_lift], axis=0)
        lift = lift @ lift
        
    return result[:t]


class RealStateSpaceModel(nn.Module):
    """
    JAX Flax module for Real State SSM.

    It returns a fully trainable real state SSM defined as:

        x_t = A @ x_{t-1} + B @ u_t,
        y_t = C @ x_t + D @ u_t + b,

    where:

        u_t, 1 <= t <= T, are the sequences of (real) inputs,
        x_t, 1 <= t <= T, are the sequence of (real) states,
        y_t, 1 <= t <= T, are the sequence of (real) outputs,
        matrices A (block diagonal real), B (real) and C (real) are mandatory,
        matrix D (real) and bias b (real) are optional,
        and x_0 is the initial (real) state.

    Note that we use the tranposed version of these equations in the Python code
    by following the 'row major' convention.
    """
    input_size: int
    state_size: int
    output_size: int
    has_matrixD: bool = False
    has_bias: bool = False
    resample_up: int = 1
    resample_down: int = 1
    init_scale_A: float = 1 - 1e-4
    state_blk_size: int = 2
    mesh: Mesh = None

    @nn.compact
    def __call__(self, u: jax.Array, x0: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
        """
        Inputs:
            u: the real input array with shape [batch, length, input_size].
            x0: the real initial state with shape [batch, state_num_blk, state_blk_size].

        Outputs:
            y: the real output array with shape [batch, resample_up*length//resample_down, output_size].
            x0: the real final state with shape [batch, state_num_blk, state_blk_size].
        """
        _, length, _ = u.shape
        
        if self.resample_up > 1:
            u = jnp.repeat(u, self.resample_up, axis=1)
            _, length, _ = u.shape
            
        assert(self.state_blk_size % 2 == 0)
        assert(self.state_size % self.state_blk_size == 0)
        state_num_blks = self.state_size // self.state_blk_size

        theta = self.param('theta', nn.initializers.uniform(2*jnp.pi), (self.state_size//2,))
        A2 = jnp.zeros((self.state_size//2, 2, 2))  # 2 x 2 blocks by default
        A2 = A2.at[:,0,0].set(jnp.cos(theta))
        A2 = A2.at[:,0,1].set(jnp.sin(theta))
        A2 = A2.at[:,1,0].set(-jnp.sin(theta))
        A2 = A2.at[:,1,1].set(jnp.cos(theta))
        
        if self.state_blk_size > 2:
            A = jnp.zeros((state_num_blks, self.state_blk_size, self.state_blk_size))
            for i in range(state_num_blks):
                start_idx = i * self.state_blk_size // 2
                end_idx = (i + 1) * self.state_blk_size // 2
                A = A.at[i].set(jax.scipy.linalg.block_diag(*A2[start_idx:end_idx]))
        else:
            A = A2
            
        A = self.init_scale_A * A

        B = self.param('B', nn.initializers.normal(stddev=1.0 / jnp.sqrt(self.state_size + self.input_size)), 
                      (self.input_size, self.state_size))
        C = self.param('C', nn.initializers.normal(stddev=1.0 / jnp.sqrt(self.state_size)), 
                      (self.state_size, self.output_size))
        D = self.param('D', nn.initializers.zeros_init(), (self.input_size, self.output_size)) if self.has_matrixD else None
        b = self.param('b', nn.initializers.zeros_init(), (self.output_size,)) if self.has_bias else None

        Aps = A_powers(A, length + 1)
        uB = jnp.matmul(u, B)  # transpose of math eq B*u
        uB = jnp.reshape(uB, (-1, length, state_num_blks, self.state_blk_size))
        x = fast_conv_real(Aps[:-1], uB)[:, :length]

        if x0 is not None:
            x = x + jnp.einsum("bki,lkij->blkj", x0, Aps[1:])

        new_state = x[:, -1]
        x = jnp.reshape(x, (-1, length, state_num_blks * self.state_blk_size))

        if self.resample_down > 1:
            x = x[:, self.resample_down-1::self.resample_down]
            if self.has_matrixD:
                u = u[:, self.resample_down-1::self.resample_down]

        y = jnp.matmul(x, C)  # transpose of math eq C*x
        if self.has_matrixD:
            y = y + jnp.matmul(u, D)  # transpose of math eq D*u
        if self.has_bias:
            y = y + b

        if self.mesh is not None:
            y = constrain(y, self.mesh, P("fsdp"))
            x = constrain(x, self.mesh, P("fsdp"))
        return y, new_state


if __name__ == "__main__":
    # test fast_conv_complex
    print("Let's test function fast_conv_complex")
    len1, len2 = 11, 17
    batch, i = 13, 5
    key = jax.random.PRNGKey(4)
    a_key, b_key = jax.random.split(key)
    a = jax.random.normal(a_key, (len1, i))
    b = jax.random.normal(b_key, (batch, len2, i))
    c = fast_conv_complex(a, b)
    c_forloop = jnp.zeros((batch, len1 + len2 - 1, i))
    for m in range(len1):
        for n in range(len2):
            c_forloop = c_forloop.at[:, m + n].add(a[m] * b[:, n])
    print(f"Max errors between fast_conv_complex and for_loop conv is {jnp.max(jnp.abs(c - c_forloop))}\n")

    # test the complex state SMM class.
    print("Let's test class ComplexStateSpaceModel")
    input_size, state_size, output_size = 3, 7, 5
    complex_ssm = ComplexStateSpaceModel(input_size=input_size, state_size=state_size, output_size=output_size,
                                     has_matrixD=True, has_bias=True, resample_up=1, resample_down=1)
    length = 10 # simply set batch size to 1
                # if you set to a huge sequence length, numerical error eventually accumulates if max(abs(A)) is too close to 1
    u_key, x0_key, params_key, apply_key = jax.random.split(key, 4)
    u = jax.random.normal(u_key, (1, length, input_size))
    x0 = jax.random.normal(x0_key, (1, state_size), dtype=jnp.complex64)
    params_complex_ssm = complex_ssm.init(params_key, u, x0)['params']
    @jax.jit
    def complex_ssm_apply(u, x0):
        return complex_ssm.apply({'params': params_complex_ssm}, u, x0, rngs={'params': apply_key})
    y, new_state = complex_ssm_apply(u, x0)

    # recursive implementation with for loop
    state = x0[0]
    A_angles = params_complex_ssm['A_angles']
    A_complex = complex_ssm.init_scale_A * jnp.exp(1j * A_angles)
    B = params_complex_ssm['B']
    C = params_complex_ssm['C']
    D = params_complex_ssm.get('D', jnp.zeros((input_size, output_size)))
    b = params_complex_ssm.get('b', jnp.zeros(output_size))

    for t in range(length):
        state = state * A_complex + (u[0, t].astype(jnp.complex64)) @ B
        out = jnp.real(state @ C) + u[0, t] @ D + b
        print(f"Max output err between conv and recursive views at step {t+1}: {jnp.max(jnp.abs(y[0,t] - out))}")
    print(f"Max final state err between conv and recursive views: {jnp.max(jnp.abs(new_state[0] - state))}\n")

    # test resampling
    print("Let's test resampling in ComplexStateSpaceModel")
    complex_ssm_resample = ComplexStateSpaceModel(input_size=input_size, state_size=state_size, output_size=output_size,
                                              has_matrixD=True, has_bias=True, resample_up=3, resample_down=7)
    u_resample = jax.random.normal(u_key, (1, 14, input_size))
    params_complex_ssm_resample = complex_ssm_resample.init(params_key, u_resample)['params']
    @jax.jit
    def complex_ssm_resample_apply(u):
        return complex_ssm_resample.apply({'params': params_complex_ssm_resample}, u, rngs={'params': apply_key})
    y_resample, _ = complex_ssm_resample_apply(u_resample)
    print(f"Sequence lengths before and after resampling complex state SSM: {u_resample.shape[1]}, {y_resample.shape[1]}\n")

    # test fast_conv_real
    print("Let's test function fast_conv_real")
    len1, len2 = 11, 17
    batch, K, n = 13, 5, 3
    a_key, b_key = jax.random.split(key)
    a = jax.random.normal(a_key, (len1, K, n, n))
    b = jax.random.normal(b_key, (batch, len2, K, n))
    c = fast_conv_real(a, b)
    c_forloop = jnp.zeros((batch, len1 + len2 - 1, K, n))
    for i in range(len2):
        for j in range(len1):
            for k in range(K):
                c_forloop = c_forloop.at[:, i + j, k].add(jnp.matmul(b[:, i, k], a[j, k]))
    print(f"Max errors between fast_conv_real and for_loop conv is {jnp.max(jnp.abs(c - c_forloop))}\n")

    # test A_powers
    print("Test function A_powers")
    K, n, t = 3, 2, 10
    A_key = jax.random.split(key)[0]
    A = jax.random.normal(A_key, (K, n, n))
    powers = A_powers(A, t)
    eye = jnp.eye(n, dtype=A.dtype)
    powers_forloop = jnp.repeat(jnp.expand_dims(eye, axis=0), K, axis=0)
    for i in range(t-1):
        print(f"Max err between fast and forloop powers of A at step {i+1}: {jnp.max(jnp.abs(powers[i] - powers_forloop))}")
        powers_forloop = jnp.matmul(powers_forloop, A)
    print("\n")

    # test the real state SMM class.
    print("Let's test class RealStateSpaceModel")
    input_size, state_size, output_size, state_blk_size = 3, 8, 5, 2
    real_ssm = RealStateSpaceModel(input_size=input_size, state_size=state_size, output_size=output_size,
                              has_matrixD=True, has_bias=True, resample_up=1, resample_down=1, state_blk_size=state_blk_size)
    length = 10 # simply set batch size to 1
                # if you set to a huge sequence length, numerical error eventually accumulates if max(abs(A)) is too close to 1
    u_key, x0_key, params_key, apply_key = jax.random.split(key, 4)
    u = jax.random.normal(u_key, (1, length, input_size))
    x0 = jax.random.normal(x0_key, (1, state_size//state_blk_size, state_blk_size))
    params_real_ssm = real_ssm.init(params_key, u, x0)['params']
    @jax.jit
    def real_ssm_apply(u, x0):
        return real_ssm.apply({'params': params_real_ssm}, u, x0, rngs={'params': apply_key})
    y, new_state = real_ssm_apply(u, x0)

    # Get state size info for our test
    state_num_blks = state_size // state_blk_size
    
    # recursive implementation with for loop
    state = x0[0].reshape(-1) # remove batch dim and flatten block state
    A_val = params_real_ssm['theta']
    A_blocks = jnp.zeros((state_num_blks, 2, 2))
    A_blocks = A_blocks.at[:,0,0].set(jnp.cos(A_val))
    A_blocks = A_blocks.at[:,0,1].set(jnp.sin(A_val))
    A_blocks = A_blocks.at[:,1,0].set(-jnp.sin(A_val))
    A_blocks = A_blocks.at[:,1,1].set(jnp.cos(A_val))
    A_blocks = real_ssm.init_scale_A * A_blocks
    A_matrix = jax.scipy.linalg.block_diag(*A_blocks)
    B = params_real_ssm['B']
    C = params_real_ssm['C']
    D = params_real_ssm.get('D', jnp.zeros((input_size, output_size)))
    b = params_real_ssm.get('b', jnp.zeros(output_size))

    for t in range(length):
        state = jnp.matmul(state, A_matrix) + jnp.matmul(u[0, t], B)
        out = jnp.matmul(state, C) + jnp.matmul(u[0, t], D) + b
        print(f"Max output err between conv and recursive views at step {t+1}: {jnp.max(jnp.abs(y[0,t] - out))}")
    print(f"Max final state err between conv and recursive views: {jnp.max(jnp.abs(new_state[0].reshape(-1) - state))}")

    # test resampling
    print("\nLet's test resampling in RealStateSpaceModel")
    real_ssm_resample = RealStateSpaceModel(input_size=input_size, state_size=state_size, output_size=output_size,
                                       has_matrixD=True, has_bias=True, resample_up=3, resample_down=7, state_blk_size=state_blk_size)
    u_resample = jax.random.normal(u_key, (1, 14, input_size))
    params_real_ssm_resample = real_ssm_resample.init(params_key, u_resample)['params']
    @jax.jit
    def real_ssm_resample_apply(u):
        return real_ssm_resample.apply({'params': params_real_ssm_resample}, u, rngs={'params': apply_key})
    y_resample, _ = real_ssm_resample_apply(u_resample)
    print(f"Sequence length before and after resampling real state SSM: {u_resample.shape[1]}, {y_resample.shape[1]}")
