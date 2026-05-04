"""Helpers for unified DA experiments: initial states and prior ensembles."""

import jax
import jax.numpy as jnp


def random_normal_initial_state(
    rng_key: jax.Array,
    num_states: int,
    state_dim: int,
    scale: float = 1.0,
    periodic_boundary: bool = False,
) -> jnp.ndarray:
    """Sample a single random-normal initial state of shape [1, num_states, state_dim]."""
    X_0 = jax.random.normal(rng_key, (1, num_states, state_dim)) * scale
    if periodic_boundary:
        X_0 = X_0.at[0, 0, -1].set(X_0[0, 0, 0])
    return X_0


def random_normal_prior_ensemble(
    rng_key: jax.Array,
    ensemble_size: int,
    num_states: int,
    state_dim: int,
    scale: float = 1.0,
    periodic_boundary: bool = False,
) -> jnp.ndarray:
    """Sample a random-normal prior ensemble of shape [ensemble_size, num_states, state_dim]."""
    ensemble = (
        jax.random.normal(rng_key, (ensemble_size, num_states, state_dim)) * scale
    )
    if periodic_boundary:
        ensemble = ensemble.at[:, :, -1].set(ensemble[:, :, 0])
    return ensemble


def kuramoto_cosine_initial_state(
    rng_key: jax.Array,
    state_dim: int,
    domain_length: float,
    magnitude: float,
) -> jnp.ndarray:
    """Deterministic two-mode cosine initial state for Kuramoto-Sivashinsky."""
    del rng_key
    x = jnp.linspace(0.0, domain_length, state_dim)
    profile = magnitude * jnp.cos(2 * jnp.pi * x / domain_length) + magnitude * jnp.cos(
        4 * jnp.pi * x / domain_length
    )
    return profile.reshape(1, 1, state_dim)


def kuramoto_cosine_prior_ensemble(
    rng_key: jax.Array,
    ensemble_size: int,
    state_dim: int,
    domain_length: float,
    magnitude_min: float,
    magnitude_max: float,
) -> jnp.ndarray:
    """Prior ensemble for Kuramoto-Sivashinsky with magnitudes drawn uniformly."""
    x = jnp.linspace(0.0, domain_length, state_dim)
    magnitudes = jax.random.uniform(
        rng_key, (ensemble_size,), minval=magnitude_min, maxval=magnitude_max
    )

    def profile(m: jnp.ndarray) -> jnp.ndarray:
        return m * jnp.cos(2 * jnp.pi * x / domain_length) + m * jnp.cos(
            4 * jnp.pi * x / domain_length
        )

    profiles = jax.vmap(profile)(magnitudes)
    return profiles.reshape(ensemble_size, 1, state_dim)


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
