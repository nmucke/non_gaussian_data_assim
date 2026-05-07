from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np

from non_gaussian_data_assim.ensemble_generation.breeding_vector import (
    BreedingVector,
    NormLike,
)
from non_gaussian_data_assim.forward_models.base import BaseForwardModel


class BasePerturbation(ABC):
    """Base class for ensemble perturbations."""

    def __init__(self, name: str, num_states: int, state_dim: int) -> None:
        self.name = name
        self.num_states = num_states
        self.state_dim = state_dim

    @abstractmethod
    def sample(self, rng_key: jax.Array, ensemble_size: int) -> jnp.ndarray:
        """Return perturbations of shape [ensemble_size, num_states, state_dim]."""
        raise NotImplementedError


class WhiteNoise(BasePerturbation):
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


class RedNoise(BasePerturbation):
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

        if self.alpha > 0:
            return jax.vmap(one_member)(member_keys)
        elif self.alpha == 0:
            whitenoise = WhiteNoise(self.num_states, self.state_dim, self.scale)
            wn_ensemble = whitenoise.sample(
                rng_key=rng_key, ensemble_size=ensemble_size
            )
            return wn_ensemble


class BreedingPerturbation(BasePerturbation):
    """Bred perturbations generated around a best-guess state."""

    def __init__(
        self,
        forward_operator: BaseForwardModel,
        num_states: int,
        state_dim: int,
        perturbation_amplitude: float,
        number_of_intervals: int,
        rescaling_interval: int,
        compute_metrics: bool,
        norm: NormLike | None = None,
    ) -> None:
        super().__init__(
            name="breeding_vector",
            num_states=num_states,
            state_dim=state_dim,
        )
        self.breeder = BreedingVector(
            forward_operator=forward_operator,
            number_of_intervals=number_of_intervals,
            rescaling_interval=rescaling_interval,
            perturbation_amplitude=perturbation_amplitude,
            norm=norm,
            compute_metrics=compute_metrics,
        )

    def sample(
        self,
        rng_key: jax.Array,
        ensemble_size: int,
        x0: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Return bred perturbations.

        Args:
            rng_key: JAX PRNG key.
            ensemble_size: Number of ensemble perturbations.
            x0: Best-guess/base state with shape [num_states, state_dim]
                or [1, num_states, state_dim].

        Returns:
            Bred perturbations with shape
            [ensemble_size, num_states, state_dim].
        """
        if x0 is None:
            raise ValueError(
                "BreedingPerturbation.sample requires x0, the best-guess state."
            )

        return self.breeder(
            x0=x0,
            rng_key=rng_key,
            ensemble_size=ensemble_size,
        )
