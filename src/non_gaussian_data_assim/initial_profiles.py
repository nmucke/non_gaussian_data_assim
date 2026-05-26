"""Initial-state profile classes used to seed prior ensembles."""

from abc import ABC, abstractmethod
from typing import Optional, Union

import jax
import jax.numpy as jnp

# --- If initital profile is random, take functionality from ensemble-generation methods
from non_gaussian_data_assim.ensemble_generation.ensemble_perturbations import (
    CoupledKuramotoPseudo1DPerturbation,
    WhiteNoise,
)
from non_gaussian_data_assim.forward_models.base import BaseForwardModel


class BaseProfile(ABC):
    """Base class for deterministic initial-state profiles."""

    def __init__(
        self, 
        name: str, 
        forward_model: Optional[BaseForwardModel] = None,
        num_states: Optional[int] = None,
        state_dim: Optional[int] = None,
        periodic: bool = False
    ) -> None:
        self.name = name
        self.periodic = periodic
        if forward_model:
            self.num_states = forward_model.num_states
            self.state_dim = forward_model.state_dim
        else:
            self.num_states = num_states
            self.state_dim = state_dim

    def _enforce_periodicity(self, profile: jnp.ndarray) -> jnp.ndarray:
        profile = profile.at[..., -1].set(profile[..., 0])
        return profile

    @abstractmethod
    def sample(self, rng_key: jax.Array) -> jnp.ndarray:
        """Return a profile of shape [ensemble_size=1, num_states, state_dim]."""
        raise NotImplementedError


class ConstantProfile(BaseProfile):
    """Profile that returns a constant value tiled across the ensemble."""

    def __init__(
        self,
        num_states: int,
        state_dim: int,
        periodic: bool,
        value: float = 0.0,
        **kwargs: int,
    ) -> None:
        super().__init__(
            name="constant",
            num_states=num_states,
            state_dim=state_dim,
            periodic=periodic,
        )
        self.value = value

    def sample(self, rng_key: jax.Array) -> jnp.ndarray:
        del rng_key
        return jnp.full((1, self.num_states, self.state_dim), self.value)


class WhiteNoiseProfile(BaseProfile):
    """
    Initial profile is random White-Noise. Created just by calling the white-noise ensemble
    """

    def __init__(
        self,
        num_states: int,
        state_dim: int,
        scale: float,
        periodic: bool,
        **kwargs: int,
    ) -> None:
        super().__init__(
            name="white_noise",
            num_states=num_states,
            state_dim=state_dim,
            periodic=periodic,
        )
        self.scale = scale

    def sample(self, rng_key: jax.Array) -> jnp.ndarray:
        whitenoise = WhiteNoise(
            num_states=self.num_states, state_dim=self.state_dim, scale=self.scale
        )
        field = whitenoise.sample(rng_key=rng_key, ensemble_size=1)
        if self.periodic:
            field = self._enforce_periodicity(field)
        return field


class CosineProfile(BaseProfile):
    """Two-mode cosine profile.

    `magnitude` may be a scalar (shared across the ensemble) or a 1-D array
    of length `ensemble_size` (one magnitude per member).
    """

    def __init__(
        self,
        num_states: int,
        state_dim: int,
        domain_length: float,
        periodic: bool,
        magnitude: Union[float, jnp.ndarray],
        **kwargs: int,
    ) -> None:
        super().__init__(
            name="cosine", num_states=num_states, state_dim=state_dim, periodic=periodic
        )
        self.domain_length = domain_length
        self.magnitude = magnitude

    def sample(self, rng_key: jax.Array) -> jnp.ndarray:
        del rng_key
        ensemble_size = 1
        x = jnp.linspace(0.0, self.domain_length, self.state_dim)

        magnitudes = jnp.asarray(self.magnitude)
        if magnitudes.ndim == 0:
            magnitudes = jnp.broadcast_to(magnitudes, (ensemble_size,))

        def profile(m: jnp.ndarray) -> jnp.ndarray:
            return m * jnp.cos(2 * jnp.pi * x / self.domain_length) + m * jnp.cos(
                4 * jnp.pi * x / self.domain_length
            )

        field = jax.vmap(profile)(magnitudes)
        field = field.reshape(ensemble_size, 1, self.state_dim)

        if self.periodic:
            field = self._enforce_periodicity(field)

        return field


class CoupledKuramotoPseudo1DProfile(BaseProfile):
    """Smooth Gaussian random field profile, matching Evensen's `pseudo1D`.

    Each state field of each ensemble member is an independent zero-mean
    unit-variance Gaussian random field on the periodic domain, smoothed to
    the given decorrelation length and scaled by `scale`.
    """

    def __init__(
        self,
        num_states: int,
        state_dim: int,
        domain_length: float,
        periodic: bool,
        decorrelation_length: float,
        scale: float = 1.0,
    ) -> None:
        super().__init__(
            name="coupled_kuramoto_pseudo1D",
            num_states=num_states,
            state_dim=state_dim,
            periodic=periodic,
        )
        self.domain_length = domain_length
        self.decorrelation_length = decorrelation_length
        self.scale = scale

    def sample(self, rng_key: jax.Array, ensemble_size: int = 1) -> jnp.ndarray:
        cks_smooth_gaussian = CoupledKuramotoPseudo1DPerturbation(
            num_states=self.num_states,
            state_dim=self.state_dim,
            domain_length=self.domain_length,
            decorrelation_length=self.decorrelation_length,
            scale=self.scale,
        )
        profile = cks_smooth_gaussian.sample(rng_key, ensemble_size=ensemble_size)

        if self.periodic:
            profile = self._enforce_periodicity(profile)

        return profile
