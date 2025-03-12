import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


state_space_model_type = "complex"
optimizer = "adam"

print(f"Domain of state vectors: {state_space_model_type}")
print(f"Optimizer: {optimizer}\n")

if state_space_model_type == "complex":
    from ssm import ComplexStateSpaceModel as SSM
    increase_state_size = 1
else:
    from ssm import RealStateSpaceModel as SSM
    increase_state_size = 2

if optimizer == "psgd":
    print("Need to download the psgd optimizer (not yet implemented in JAX)")


mnist_transform = transforms.Compose([transforms.ToTensor(), lambda x: np.array(x)])
train_dataset = datasets.MNIST("../data", train=True, download=True, transform=mnist_transform)
test_dataset = datasets.MNIST("../data", train=False, transform=mnist_transform)

batch_size = 60
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class SSMNetJAX(nn.Module):
    state_space_model_type: str
    
    @nn.compact
    def __call__(self, u):
        if self.state_space_model_type == "complex":
            ssm1 = SSM(input_size=1, state_size=increase_state_size * 16, output_size=16)
            ssm2 = SSM(input_size=16, state_size=increase_state_size * 128, output_size=128)
        else:
            ssm1 = SSM(input_size=1, state_size=increase_state_size * 16, output_size=16)
            ssm2 = SSM(input_size=16, state_size=increase_state_size * 128, output_size=128)

        x, _ = ssm1(u)
        x = x * jax.lax.rsqrt(1 + x*x)
        x, _ = ssm2(x)
        x = x[:, -1]
        x = x * jax.lax.rsqrt(1 + x*x)
        linear = nn.Dense(features=10)
        x = linear(x)
        return x


key = jax.random.PRNGKey(0)
key_params, key_apply = jax.random.split(key)
ssmnet = SSMNetJAX(state_space_model_type=state_space_model_type)
dummy_input = jnp.ones([1, 28*28, 1])
params = ssmnet.init(key_params, dummy_input)['params']

lr0 = 5e-4
if optimizer == "adam":
    opt = optax.adam(learning_rate=lr0)
else:
    raise NotImplementedError("PSGD optimizer is not implemented in this JAX demo.")
opt_state = opt.init(params)

num_epochs = 20
TrainLosses, TestErrs = [], []


@jax.jit
def apply_model(params, key, inputs):
    return ssmnet.apply({'params': params}, inputs, rngs={'params': key})

@jax.jit
def calculate_loss(params, key, batch):
    data, target = batch
    y = apply_model(params, key, jnp.reshape(data, [-1, 28*28, 1]))
    log_probs = nn.log_softmax(y, axis=-1)
    loss = -jnp.mean(jnp.take_along_axis(log_probs, jnp.expand_dims(target, axis=-1), axis=-1))
    return loss

@jax.jit
def train_step(params, opt_state, key, batch):
    key_loss, key_grad = jax.random.split(key)
    loss_value, grads = jax.value_and_grad(calculate_loss)(params, key_loss, batch)
    updates, new_opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_value, key_grad


for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = np.array(data)
        target = np.array(target)
        batch_np = (data, target)
        batch_jax = tuple(jnp.array(item) for item in batch_np)
        params, opt_state, loss_value, key_apply = train_step(params, opt_state, key_apply, batch_jax)
        TrainLosses.append(loss_value)
        if (batch_idx+1) % 100 == 0:
            print(f"Epoch: {epoch + 1}; train loss: {np.mean(TrainLosses[-1000:])}")

    num_errs = 0
    for data, target in test_loader:
        data = np.array(data)
        target = np.array(target)
        y = apply_model(params, key_apply, jnp.reshape(jnp.array(data), [-1, 28*28, 1]))
        pred = jnp.argmax(y, axis=1)
        num_errs += jnp.sum(pred != target)
    test_err_rate = num_errs.item() / len(test_dataset)
    TestErrs.append(test_err_rate)
    print(f"Epoch: {epoch + 1}; test classification error rate: {TestErrs[-1]}")

    if optimizer == "adam":
        lr0 -= lr0 / num_epochs
        opt = optax.adam(learning_rate=lr0)
        opt_state = opt.init(params)
    else:
        pass

plt.plot(TrainLosses)
plt.show()
plt.plot(TestErrs)
plt.show()
