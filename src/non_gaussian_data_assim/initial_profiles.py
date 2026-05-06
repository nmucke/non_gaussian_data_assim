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



def _smooth_gaussian_periodic_1d(
    rng_key: jax.Array,
    state_dim: int,
    domain_length: float,
    decorrelation_length: float,
) -> jnp.ndarray:
    """Sample one periodic, mean-zero, unit-variance smooth Gaussian random field.

    Equivalent to Evensen's `pseudo1D`: white noise filtered in Fourier space by a
    Gaussian power spectrum exp(-(k * L)^2 / 2), then renormalized so the realized
    field has empirical mean 0 and std 1.
    """
    dx = domain_length / state_dim
    k = 2.0 * jnp.pi * jnp.fft.rfftfreq(state_dim, d=dx)
    spectrum = jnp.exp(-((k * decorrelation_length) ** 2) / 2.0)

    key_re, key_im = jax.random.split(rng_key)
    n_freq = k.shape[0]
    coefs = (
        jax.random.normal(key_re, (n_freq,)) + 1j * jax.random.normal(key_im, (n_freq,))
    ) * spectrum
    field = jnp.fft.irfft(coefs, n=state_dim)
    return (field - field.mean()) / field.std()


def coupled_kuramoto_pseudo1D_initial_state(
    rng_key: jax.Array,
    state_dim: int,
    domain_length: float,
    decorrelation_length: float,
) -> jnp.ndarray:
    """Smooth Gaussian random field IC for Coupled KS, matching Evensen's pseudo1D.

    Each of the two state fields (Atmos, Ocean) is an independent zero-mean
    unit-variance Gaussian random field on the periodic domain, smoothed to the
    given decorrelation length.
    """
    keys = jax.random.split(rng_key, 2)
    fields = jax.vmap(
        lambda key: _smooth_gaussian_periodic_1d(
            key, state_dim, domain_length, decorrelation_length
        )
    )(keys)
    return fields.reshape(1, 2, state_dim)


def coupled_kuramoto_pseudo1D_prior_ensemble(
    rng_key: jax.Array,
    ensemble_size: int,
    state_dim: int,
    domain_length: float,
    decorrelation_length: float,
    scale: float = 1.0,
) -> jnp.ndarray:
    """Prior ensemble of independent smooth Gaussian random fields for Coupled KS."""
    keys = jax.random.split(rng_key, ensemble_size * 2)
    fields = jax.vmap(
        lambda key: _smooth_gaussian_periodic_1d(
            key, state_dim, domain_length, decorrelation_length
        )
    )(keys)
    return scale * fields.reshape(ensemble_size, 2, state_dim)
