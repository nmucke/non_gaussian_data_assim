from abc import ABC, abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp

from non_gaussian_data_assim.ensemble_generation.breeding_vector import (
    BreedingVector,
    EnsembleNorm,
)
from non_gaussian_data_assim.forward_models.base import BaseForwardModel


class BasePerturbation(ABC):
    """Base class for ensemble perturbations.

    x0_bg: Best-Guess reference state
        -- if suplied, ensemble-perturbation will be added to it
        -- REQUIRED as starting Conditions for Breeding with shape [num_states, state_dim]
                    or [1, num_states, state_dim].
    """

    def __init__(
        self,
        name: str,
        num_states: int,
        state_dim: int,
    ) -> None:
        self.name = name
        self.num_states = num_states
        self.state_dim = state_dim

    def _add_ensemble_to_bestguess_profile(
        self, bg_profile: jnp.ndarray, ensemble: jnp.ndarray
    ) -> jnp.ndarray:
        return bg_profile + ensemble

    @abstractmethod
    def sample(
        self,
        rng_key: jax.Array,
        ensemble_size: int,
        bg_profile: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Return perturbations of shape [ensemble_size, num_states, state_dim].

        Args:
            rng_key: JAX PRNG key.
            ensemble_size: Number of ensemble perturbations.

        Returns:
            Perturbations with shape
            [ensemble_size, num_states, state_dim].
        """
        raise NotImplementedError


# TODO: Make sure breeding is working as expected with multiple variables!!


class BreedingPerturbation(BasePerturbation):
    """Create Ensemble of Bred perturbations generated around a best-guess state."""

    def __init__(
        self,
        forward_model: BaseForwardModel,
        num_states: int,
        state_dim: int,
        delta0: float,
        breeding_cycles: int,
        outer_steps_per_cycle: int,
        norm_fct: EnsembleNorm | None = None,
        min_norm: float = 1e-10,
    ) -> None:
        super().__init__(
            name="breeding_vector",
            num_states=num_states,
            state_dim=state_dim,
        )

        self.breeder = BreedingVector(
            forward_model=forward_model,
            breeding_cycles=breeding_cycles,
            outer_steps_per_cycle=outer_steps_per_cycle,
            delta0=delta0,
            norm_fct=norm_fct,
            min_norm=min_norm,
        )

    def sample(
        self,
        rng_key: jax.Array,
        ensemble_size: int,
        bg_profile: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Return bred perturbations with shape [ensemble_size, num_states, state_dim]."""

        if bg_profile is None:
            raise ValueError(
                f"Breeding reuires that a best-guess profile is supplied, but it was None"
            )
        return self.breeder.sample_ensemble(
            x0_bg=bg_profile,
            rng_key=rng_key,
            ensemble_size=ensemble_size,
            return_metrics=True,
        )


class WhiteNoise(BasePerturbation):
    """Spatially white Gaussian noise with standard deviation `scale`."""

    def __init__(self, num_states: int, state_dim: int, scale: float) -> None:
        super().__init__(name="white_noise", num_states=num_states, state_dim=state_dim)
        self.scale = scale

    def sample(
        self,
        rng_key: jax.Array,
        ensemble_size: int,
        bg_profile: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # --- Since fields are uncorrelated, we can create for all members, all states and entire state-dim at once
        shape = (ensemble_size, self.num_states, self.state_dim)
        ensemble = jax.random.normal(rng_key, shape) * self.scale

        if bg_profile is not None:
            ensemble = self._add_ensemble_to_bestguess_profile(
                bg_profile=bg_profile, ensemble=ensemble
            )

        return ensemble


# ========================================================================
# ---- Helper-Functions: Produce correlated noise in fourier space


def _powerlaw_rednoise_periodic_1d(
    rng_key: jax.Array,
    state_dim: int,
    domain_length: float,
    alpha: float,
) -> jnp.ndarray:
    """Sample one periodic, mean-zero, unit-variance 1D red-noise field.

    White complex noise is filtered in Fourier space with a power-law spectrum

        amplitude(k) ∝ |k|^(-alpha / 2)

    and transformed back to physical space. The zero mode is removed, so the
    field is mean-zero, then the realized field is normalized to unit empirical
    standard deviation.
    """

    shape = (state_dim,)
    # --- Physical grid spacing. Using physical frequencies makes domain_length explicit
    dx = domain_length / state_dim

    # --- Wave-Numbers for 1d case
    k = jnp.abs(jnp.fft.fftfreq(state_dim, d=dx))

    # --- Avoid a singular zero-frequency mode, explicitly set its amplitude to 0
    k_safe = k.at[0].set(1.0)
    amplitude = k_safe ** (-alpha / 2.0)
    amplitude = amplitude.at[0].set(0.0)

    key_re, key_im = jax.random.split(rng_key)
    # --- Create random noise in fourier space
    noise_hat = jax.random.normal(key_re, shape) + 1j * jax.random.normal(key_im, shape)
    field = jnp.fft.ifft(noise_hat * amplitude).real

    # --- I think it's already zero-mean, but let's make it robust!
    field = field - jnp.mean(field)
    return field / jnp.std(field)


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


class RedNoise(BasePerturbation):
    """Spatially correlated red noise with power spectrum P(k) ~ k^{-alpha}.

    Returns perturbations rescaled to standard deviation `scale`.
    """

    def __init__(
        self,
        num_states: int,
        state_dim: int,
        domain_length: float,
        scale: float,
        alpha: float,
    ) -> None:

        super().__init__(name="red_noise", num_states=num_states, state_dim=state_dim)
        self.scale = scale
        self.domain_length = domain_length
        self.alpha = alpha

    def sample(
        self,
        rng_key: jax.Array,
        ensemble_size: int,
        bg_profile: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Create RedNoise Ensemble by calling function for each member"""

        if self.alpha > 0:
            member_keys = jax.random.split(rng_key, ensemble_size * self.num_states)
            fields = jax.vmap(
                lambda key: _powerlaw_rednoise_periodic_1d(
                    key, self.state_dim, self.domain_length, self.alpha
                )
            )(member_keys)
            ensemble = self.scale * fields.reshape(
                ensemble_size, self.num_states, self.state_dim
            )

            if bg_profile is not None:
                ensemble = self._add_ensemble_to_bestguess_profile(
                    bg_profile=bg_profile, ensemble=ensemble
                )
            return ensemble

        elif self.alpha == 0:
            whitenoise = WhiteNoise(self.num_states, self.state_dim, self.scale)
            wn_ensemble = whitenoise.sample(
                rng_key=rng_key, ensemble_size=ensemble_size, bg_profile=bg_profile
            )
            return wn_ensemble


class CoupledKuramotoPseudo1DPerturbation(BasePerturbation):
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
        decorrelation_length: float,
        scale: float = 1.0,
    ) -> None:
        super().__init__(
            name="coupled_kuramoto_pseudo1D", num_states=num_states, state_dim=state_dim
        )
        self.domain_length = domain_length
        self.decorrelation_length = decorrelation_length
        self.scale = scale

    def sample(
        self,
        rng_key: jax.Array,
        ensemble_size: int,
        bg_profile: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        keys = jax.random.split(rng_key, ensemble_size * self.num_states)
        fields = jax.vmap(
            lambda key: _smooth_gaussian_periodic_1d(
                key, self.state_dim, self.domain_length, self.decorrelation_length
            )
        )(keys)
        ensemble = self.scale * fields.reshape(
            ensemble_size, self.num_states, self.state_dim
        )

        if bg_profile is not None:
            ensemble = self._add_ensemble_to_bestguess_profile(
                bg_profile=bg_profile, ensemble=ensemble
            )
        return ensemble
