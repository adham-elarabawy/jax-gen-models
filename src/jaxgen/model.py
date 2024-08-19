import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from abc import ABC, abstractmethod

class BaseModel(ABC, nn.Module):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def train_step(self, state: train_state.TrainState, batch: dict[str, jnp.ndarray]) -> tuple[train_state.TrainState, jnp.ndarray]:
        pass

    @abstractmethod
    def sample(self, rng: jax.random.PRNGKey, num_samples: int):
        pass

    @abstractmethod
    def evaluate(self, state: train_state.TrainState, eval_loader):
        pass

    def create_train_state(self, rng: jax.random.PRNGKey, learning_rate: float):
        # TODO: understand what rng is doing here exactly
        params = self.init(rng, jnp.ones([1, *self.input_shape]))['params']
        tx = optax.adam(learning_rate)
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)

class SimpleAutoencoder(BaseModel):
    input_shape: tuple[int]
    encoder: nn.Module
    decoder: nn.Module

    def setup(self):
        self.encoder = nn.Dense(features=128)
        self.decoder = nn.Dense(features=self.input_shape[0])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    @jax.jit
    def train_step(self, state: train_state.TrainState, batch: dict[str, jnp.ndarray]) -> tuple[train_state.TrainState, jnp.ndarray]:
        # TODO: understand what params is doing here, and wtf apply_fn is
        def loss_fn(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
            recon_x = state.apply_fn({'params': params}, batch['images'])
            loss = jnp.mean((batch['images'] - recon_x) ** 2)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        # TODO: log this or smth
        return state, loss

    @jax.jit
    def sample(self, rng, num_samples):
        z = jax.random.normal(rng, (num_samples, 128))
        samples = self.decoder(z)
        return samples

    @jax.jit
    def evaluate(self, state, eval_loader):
        loss_accum = 0
        for batch in eval_loader:
            recon_x = state.apply_fn({'params': state.params}, batch['images'])
            loss = jnp.mean((batch['images'] - recon_x) ** 2)
            loss_accum += loss
        return loss_accum / len(eval_loader)