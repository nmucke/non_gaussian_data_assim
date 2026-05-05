import random
from abc import ABC, abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


class BaseNoise(ABC):
    """Base class for ensemble perturbations."""

    def __init__(self, name: str, num_states: int, state_dim: int) -> None:
        self.name = name
        self.num_states = num_states
        self.state_dim = state_dim

    @abstractmethod
    def sample(self, rng_key: jax.Array, ensemble_size: int) -> jnp.ndarray:
        """Return perturbations of shape [ensemble_size, num_states, state_dim]."""
        raise NotImplementedError


class WhiteNoise(BaseNoise):
    """Spatially white Gaussian noise with standard deviation `scale`."""

    def __init__(
        self,
        num_states: int,
        state_dim: int,
        scale: float,
    ) -> None:
        super().__init__(name="white_noise", num_states=num_states, state_dim=state_dim)
        self.scale = scale

    def sample(self, rng_key: jax.Array, ensemble_size: int) -> jnp.ndarray:
        shape = (ensemble_size, self.num_states, self.state_dim)
        return jax.random.normal(rng_key, shape) * self.scale


class RedNoise(BaseNoise):
    """Spatially correlated red noise with power spectrum P(k) ~ k^{-alpha}.

    Returns perturbations rescaled to standard deviation `scale`.
    """

    def __init__(
        self,
        num_states: int,
        state_dim: int,
        scale: float,
        alpha: float,
    ) -> None:
        super().__init__(name="red_noise", num_states=num_states, state_dim=state_dim)
        self.scale = scale
        self.alpha = alpha

    def sample(self, rng_key: jax.Array, ensemble_size: int) -> jnp.ndarray:
        shape = (self.num_states, self.state_dim)

        k_arrays = [jnp.fft.fftfreq(n) for n in shape]
        k_grids = jnp.meshgrid(*k_arrays, indexing="ij")
        k = jnp.sqrt(sum(kg**2 for kg in k_grids))
        k = k.at[(0,) * len(shape)].set(jnp.inf)
        amplitude = k ** (-self.alpha / 2)

        def one_member(key: jax.Array) -> jnp.ndarray:
            _, key = jax.random.split(key)
            noise_hat = jax.random.normal(key, shape)

            _, key = jax.random.split(key)
            noise_hat = noise_hat + 1j * jax.random.normal(key, shape)
            field = jnp.fft.ifftn(noise_hat * amplitude).real
            return self.scale * field / field.std()

        member_keys = jax.random.split(rng_key, ensemble_size)
        return jax.vmap(one_member)(member_keys)
