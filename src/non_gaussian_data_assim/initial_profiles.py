"""Initial-state profile classes used to seed prior ensembles."""

from abc import ABC, abstractmethod
from typing import Union

import jax
import jax.numpy as jnp


class BaseProfile(ABC):
    """Base class for deterministic initial-state profiles."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def sample(self, rng_key: jax.Array, ensemble_size: int) -> jnp.ndarray:
        """Return a profile of shape [ensemble_size, num_states, state_dim]."""
        raise NotImplementedError


class ConstantProfile(BaseProfile):
    """Profile that returns a constant value tiled across the ensemble."""

    def __init__(
        self,
        num_states: int,
        state_dim: int,
        value: float = 0.0,
    ) -> None:
        super().__init__(name="constant")
        self.num_states = num_states
        self.state_dim = state_dim
        self.value = value

    def sample(self, rng_key: jax.Array, ensemble_size: int) -> jnp.ndarray:
        del rng_key
        return jnp.full((ensemble_size, self.num_states, self.state_dim), self.value)


class CosineProfile(BaseProfile):
    """Two-mode cosine profile.

    `magnitude` may be a scalar (shared across the ensemble) or a 1-D array
    of length `ensemble_size` (one magnitude per member).
    """

    def __init__(
        self,
        state_dim: int,
        domain_length: float,
        magnitude: Union[float, jnp.ndarray],
    ) -> None:
        super().__init__(name="cosine")
        self.state_dim = state_dim
        self.domain_length = domain_length
        self.magnitude = magnitude

    def sample(self, rng_key: jax.Array, ensemble_size: int) -> jnp.ndarray:
        del rng_key
        x = jnp.linspace(0.0, self.domain_length, self.state_dim)

        magnitudes = jnp.asarray(self.magnitude)
        if magnitudes.ndim == 0:
            magnitudes = jnp.broadcast_to(magnitudes, (ensemble_size,))

        def profile(m: jnp.ndarray) -> jnp.ndarray:
            return m * jnp.cos(2 * jnp.pi * x / self.domain_length) + m * jnp.cos(
                4 * jnp.pi * x / self.domain_length
            )

        profiles = jax.vmap(profile)(magnitudes)
        return profiles.reshape(ensemble_size, 1, self.state_dim)
