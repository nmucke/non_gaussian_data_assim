from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from non_gaussian_data_assim.da_methods.base import (
    BaseDataAssimilationMethod,
    make_localization_fn,
)
from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.gaussian_mixture import gaussian_mixt
from non_gaussian_data_assim.observations.observation_operator import (
    ObservationOperator,
)


def uniform_weights(ensemble_size: int) -> np.ndarray:
    """Uniform-weight vector used to initialize AGMF particle weights."""
    return np.ones(ensemble_size) / ensemble_size


class AdaptiveGaussianMixtureFilter(BaseDataAssimilationMethod):
    # AGMF carries the particle weights forward in ``self.w_prev`` across
    # analysis steps, so it is a stateful filter and must not be run via
    # BaseDataAssimilationMethod.rollout (jax.lax.scan cannot trace the update).
    is_stateful: bool = True

    def __init__(
        self,
        ensemble_size: int,
        R: np.ndarray,
        w_prev: Optional[np.ndarray],
        nc_threshold: float,
        obs_operator: ObservationOperator,
        forward_operator: BaseForwardModel,
        name: str = "agmf",
        inflation_factor: float = 1.0,
        localization_distance: Optional[int] = None,
        periodic: bool = False,
    ) -> None:
        """
        Initialize the Adaptive Gaussian Mixture Filter.
        Args:
        ensemble_size (int): Number of ensemble members.
        R (numpy.array): Observation error covariance matrix.
        inflation_factor (float): Inflation factor.
        w_prev (numpy.array): Previous weights of the ensemble members.
        nc_threshold (float): Threshold for deciding whether resampling is necessary.
        obs_operator (Callable[[np.ndarray], np.ndarray]): Observation operator.
        forward_operator (Callable[[np.ndarray], np.ndarray]): Forward operator.
        localization_distance (int): Localization distance.
        """
        super().__init__(name, obs_operator, forward_operator)
        self.ensemble_size = ensemble_size
        self.R = R
        self.inflation_factor = inflation_factor
        # Starting weights for a freshly-constructed filter. Configs pass the
        # uniform weights via ``uniform_weights(ensemble_size)``; fall back to
        # that here so the recursion always starts from a valid state.
        self.w_prev = (
            uniform_weights(ensemble_size) if w_prev is None else w_prev
        )
        self.nc_threshold = nc_threshold
        self.localization_distance = localization_distance
        self.num_states = forward_operator.num_states
        self.state_dim = forward_operator.state_dim
        self.periodic = periodic

        self.localization = make_localization_fn(
            self.localization_distance,
            self.state_dim,
            self.num_states,
            self.periodic,
        )

    def _analysis_step(
        self,
        prior_ensemble: np.ndarray,
        obs_vect: np.ndarray,
        rng_key: jax.random.PRNGKey,
        **kwargs: Any,
    ) -> np.ndarray:
        """Analysis step of the Adaptive Gaussian Mixture Filter."""

        # Preparing the prior state vector (ensemble matrix)
        prior_ensemble = prior_ensemble.reshape(self.ensemble_size, -1).T

        # Calculating the mean and covariance of the prior
        cov_prior = (self.inflation_factor**2) * jnp.cov(prior_ensemble)
        cov_prior = self.localization(cov_prior)

        # Filter and perturb the observation vector
        rng_key, key = jax.random.split(rng_key)
        obs_vect_perturbed = obs_vect + jax.random.multivariate_normal(
            key,
            jnp.zeros(self.obs_operator.num_obs),  # type: ignore[attr-defined]
            self.R,
            shape=(self.ensemble_size,),
        )
        obs_vect_perturbed = obs_vect_perturbed.T

        # Calculating the observation operator matrix
        obs_matrix = self.obs_operator.obs_matrix  # type: ignore[attr-defined]

        # Calculate the Kalman gain
        k_left = cov_prior @ obs_matrix.T
        k_right = obs_matrix @ cov_prior @ obs_matrix.T + self.R

        # K_left * K_right^-1
        kalman_gain = jnp.linalg.solve(k_right, k_left.T).T

        # Calculate the innovation
        innovation = obs_vect_perturbed - obs_matrix @ prior_ensemble

        # Calculate the posterior ensemble
        posterior_ensemble = prior_ensemble + kalman_gain @ innovation

        cov_posterior = jnp.cov(posterior_ensemble)

        # Recalculating weights
        w_t = gaussian_mixt(
            self.w_prev,
            self.obs_operator.num_obs,  # type: ignore[attr-defined]
            posterior_ensemble,
            obs_vect_perturbed,
            obs_matrix,
            self.R,
        )

        # Evaluating degeneracy and calculating the bridging alpha
        N_eff = 1 / jnp.sum(w_t**2)
        alpha = N_eff / self.ensemble_size

        # Adjusting weights
        w_t = w_t * alpha + (1 - alpha) * (1 / self.ensemble_size)
        # Carry the weights forward for the next analysis step. This in-place
        # update makes AGMF stateful and is only valid under the eager
        # ``da_model(...)`` path (a fresh filter is built per experiment, so the
        # recursion restarts from the constructor's w_prev). It must not run
        # under jax.lax.scan; ``rollout`` rejects stateful filters (is_stateful).
        self.w_prev = w_t

        # Resampling if necessary
        def resample_fn(
            rng_key: jax.random.PRNGKey,
            posterior_ensemble: jnp.ndarray,
            cov_posterior: jnp.ndarray,
            w_t: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            """Resample the ensemble when degeneracy is detected."""
            J = jax.random.choice(
                rng_key, jnp.arange(self.ensemble_size), (self.ensemble_size,), p=w_t
            )
            rng_key, key = jax.random.split(rng_key)
            epsc = jax.random.normal(key, (self.ensemble_size,)) * 0.1
            # Resample: replace each column i with column J[i] plus noise
            noise = jnp.sqrt(jnp.diag(cov_posterior))[:, None] * epsc[None, :]
            posterior_ensemble = posterior_ensemble[:, J] + noise
            cov_posterior = (self.inflation_factor**2) * jnp.cov(posterior_ensemble)
            return posterior_ensemble, cov_posterior

        def no_resample_fn(
            rng_key: jax.random.PRNGKey,
            posterior_ensemble: jnp.ndarray,
            cov_posterior: jnp.ndarray,
            w_t: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            """No resampling needed."""
            return posterior_ensemble, cov_posterior

        # Use JAX conditional instead of Python if
        should_resample = N_eff < self.nc_threshold
        posterior_ensemble, cov_posterior = jax.lax.cond(
            should_resample,
            resample_fn,
            no_resample_fn,
            rng_key,
            posterior_ensemble,
            cov_posterior,
            w_t,
        )

        posterior_ensemble = posterior_ensemble.T
        posterior_ensemble = posterior_ensemble.reshape(
            self.ensemble_size, self.num_states, self.state_dim
        )

        return posterior_ensemble
