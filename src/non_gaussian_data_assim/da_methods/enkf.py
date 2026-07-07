from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from non_gaussian_data_assim.da_methods.base import (
    BaseDataAssimilationMethod,
    make_localization_fn,
)
from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.observations.observation_operator import (
    ObservationOperator,
)


class EnsembleKalmanFilter(BaseDataAssimilationMethod):
    """Ensemble Kalman Filter."""

    def __init__(
        self,
        ensemble_size: int,
        R: np.ndarray,
        forward_operator: BaseForwardModel,
        obs_operator: ObservationOperator,
        name: str = "enkf",
        inflation_factor: float = 1.0,
        localization_distance: Optional[int] = None,
        periodic: bool = False,
    ) -> None:
        """
        Initialize the Ensemble Kalman Filter.
        Args:
        ensemble_size (int): Number of ensemble members.
        R (numpy.array): Observation error covariance matrix.
        inflation_factor (float): Inflation factor.
        forward_operator (Callable[[np.ndarray], np.ndarray]): Forward operator.
        obs_operator (Callable[[np.ndarray], np.ndarray]): Observation operator.
        localization_distance (int): Localization distance.
        """
        super().__init__(name, obs_operator, forward_operator)
        self.ensemble_size = ensemble_size
        self.inflation_factor = inflation_factor
        self.num_states = forward_operator.num_states
        self.state_dim = forward_operator.state_dim
        self.dofs = self.num_states * self.state_dim
        self.R = R
        self.localization_distance = localization_distance
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
        """Analysis step of the Ensemble Kalman Filter.

        Args:
        mem (int): Number of ensemble members.
        nx (int): The size of the state vector.
        ensemble (numpy.array): Ensemble of state estimates.
        obs_vect (numpy.array): Observation vector.
        R (numpy.array): Observation error covariance matrix.
        rng_key (jax.random.PRNGKey): RNG key.

        Returns:
        dict: A dictionary containing the posterior ensemble, Kalman gain, innovation,
            mean and covariance of the posterior.
        """
        # Identify indices of valid observations

        # Prepare the prior state vector (ensemble matrix) -> shape [dofs, N]
        prior_ensemble = prior_ensemble.reshape(self.ensemble_size, -1).T

        # Standard multiplicative prior inflation: inflate the forecast ensemble
        # ANOMALIES about their mean by inflation_factor (lambda). This scales the
        # standard deviation by lambda (covariance by lambda^2) AND, because the
        # analysis update below starts from these inflated members, it actually
        # increases the posterior spread -- unlike scaling only cov_prior, which
        # would inflate the gain and shrink the posterior spread (the opposite of
        # the intended effect).
        mean = jnp.mean(prior_ensemble, axis=1, keepdims=True)
        prior_ensemble = mean + self.inflation_factor * (prior_ensemble - mean)

        # Calculate the covariance of the (inflated) prior
        cov_prior = jnp.cov(prior_ensemble)
        cov_prior = self.localization(cov_prior)

        # Filter and perturb the observation vector
        rng_key, key = jax.random.split(rng_key)
        perturb = jax.random.multivariate_normal(
            key, jnp.zeros(self.obs_operator.num_obs), self.R, shape=(self.ensemble_size,)  # type: ignore[attr-defined]
        )
        # Center the perturbations exactly across the ensemble axis. The raw draws
        # carry an O(1/sqrt(N)) sample mean that would otherwise add a spurious
        # shift to the analysis mean; subtracting the ensemble mean removes it.
        perturb = perturb - perturb.mean(axis=0, keepdims=True)
        obs_vect_perturbed = obs_vect + perturb
        obs_vect_perturbed = obs_vect_perturbed.T

        # Observation operator matrix
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
        posterior_ensemble = posterior_ensemble.T

        posterior_ensemble = posterior_ensemble.reshape(
            self.ensemble_size, self.num_states, self.state_dim
        )

        return posterior_ensemble
