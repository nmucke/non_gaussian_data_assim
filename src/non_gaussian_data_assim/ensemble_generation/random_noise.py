from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp


class BaseMemberNoise(ABC):
    """Base class for perturbations of one ensemble member.

    Returns:
        One perturbation member with shape [num_states, state_dim].
    """

    def __init__(self, name: str, num_states: int, state_dim: int) -> None:
        self.name = name
        self.num_states = num_states
        self.state_dim = state_dim

    @abstractmethod
    def sample_member(self, rng_key: jax.Array) -> jnp.ndarray:
        """Return one perturbation member with shape [num_states, state_dim]."""
        raise NotImplementedError


def _normalize_field(
    field: jnp.ndarray,
    remove_mean: bool = True,
    normalize_std: bool = True,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Normalize one spatial field"""
    if remove_mean:
        field = field - jnp.mean(field)
    if normalize_std:
        std = jnp.std(field)
        field = field / jnp.maximum(std, eps)
    return field


def _sample_periodic_spectral_field_1d(
    rng_key: jax.Array,
    state_dim: int,
    amplitude: jnp.ndarray,
    remove_mean: bool = True,
    normalize_std: bool = True,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Sample one periodic, mean-zero, unit-variance Gaussian random field.

    Args:
        rng_key:
            JAX random key.
        state_dim:
            Number of spatial grid points.
        amplitude:
            Spectral amplitude filter with shape [state_dim // 2 + 1].
            The resulting power spectrum is proportional to amplitude**2.

    Returns:
        Field with shape [state_dim].
    """

    key_re, key_im = jax.random.split(rng_key)

    n_freq = state_dim // 2 + 1

    coefs = jax.random.normal(key_re, (n_freq,)) + 1j * jax.random.normal(
        key_im, (n_freq,)
    )

    # Enforce zero spatial mean.
    coefs = coefs.at[0].set(0.0 + 0.0j)

    # For even state_dim, the Nyquist mode should be real-valued.
    if state_dim % 2 == 0:
        coefs = coefs.at[-1].set(jnp.real(coefs[-1]) + 0.0j)

    field = jnp.fft.irfft(coefs * amplitude, n=state_dim)

    return _normalize_field(
        field,
        remove_mean=remove_mean,
        normalize_std=normalize_std,
        eps=eps,
    )


class BaseSpectralMemberNoise1D(BaseMemberNoise):
    """
    Base class for one-member 1D spectral noise.

    Creates one perturbation member with shape:
        [num_states, state_dim]

    """

    def __init__(
        self,
        name: str,
        num_states: int,
        state_dim: int,
        scale: float,
        domain_length: float = 1.0,
        remove_mean: bool = True,
        normalize_std: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            name=name,
            num_states=num_states,
            state_dim=state_dim,
        )

        self.scale = scale
        self.domain_length = domain_length
        self.remove_mean = remove_mean
        self.normalize_std = normalize_std
        self.eps = eps

    @abstractmethod
    def amplitude(self) -> jnp.ndarray:
        """Return spectral amplitude filter with shape [state_dim // 2 + 1]."""
        raise NotImplementedError

    def sample_member(self, rng_key: jax.Array) -> jnp.ndarray:
        """Sample one ensemble member.

        Returns:
            [num_states, state_dim]
        """

        amplitude = self.amplitude()
        state_keys = jax.random.split(rng_key, self.num_states)
        fields = jax.vmap(
            lambda key: _sample_periodic_spectral_field_1d(
                rng_key=key,
                state_dim=self.state_dim,
                amplitude=amplitude,
                remove_mean=self.remove_mean,
                normalize_std=self.normalize_std,
                eps=self.eps,
            )
        )(state_keys)

        return self.scale * fields
