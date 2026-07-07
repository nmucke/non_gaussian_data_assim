from typing import Optional

import jax
import jax.numpy as jnp

from non_gaussian_data_assim.perturbations.base import BasePerturbation
from non_gaussian_data_assim.perturbations.white_noise import WhiteNoise


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

        else:
            raise ValueError(
                f"RedNoise requires alpha >= 0 (alpha == 0 falls back to white "
                f"noise, alpha > 0 gives a k^(-alpha) power spectrum). Got "
                f"alpha={self.alpha}."
            )
