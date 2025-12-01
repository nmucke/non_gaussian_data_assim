import pdb
from typing import Any, Dict, Optional

import numpy as np

from non_gaussian_data_assim.da_methods.base import BaseDataAssimilationMethod
from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.gaussian_mixture import gaussian_mixt
from non_gaussian_data_assim.localization import distance_based_localization
from non_gaussian_data_assim.observation_operator import ObservationOperator
from non_gaussian_data_assim.rand_utils import randsample

# def agmf(
#     mem: int,
#     nx: int,
#     ensemble: np.ndarray,
#     obs_vect: np.ndarray,
#     R: np.ndarray,
#     h: float,
#     w_prev: np.ndarray,
#     nc_threshold: float,
# ) -> Dict[str, Any]:
#     """
#     Implement the Adaptive Gaussian Mixture Filter.

#     Args:
#     mem (int): Number of ensemble members.
#     nx (int): Size of the state vector.
#     ensemble (numpy.array): Ensemble of state estimates.
#     obs_vect (numpy.array): Observation vector.
#     R (numpy.array): Observation error covariance matrix.
#     h (float): Scale factor for covariance matrix.
#     w_prev (numpy.array): Previous weights of the ensemble members.
#     nc_threshold (float): Threshold for deciding whether resampling is necessary.

#     Returns:
#     dict: Dictionary containing the posterior ensemble, Kalman gain, innovation,
#           mean and covariance of the posterior, weights, and alpha value.
#     """
#     # Identifying indices of valid observations
#     index_obs = np.where(obs_vect > -999)[0]
#     num_obs = len(index_obs)

#     # Preparing the prior state vector (ensemble matrix)
#     prior_vect = ensemble

#     # Calculating the mean and covariance of the prior
#     mean_prior = np.mean(prior_vect, axis=1)
#     cov_prior = (h**2) * np.cov(prior_vect)

#     # Filtering and perturbing the observation vector
#     obs_vect_filtered = obs_vect[index_obs]
#     obs_vect_perturbed = np.zeros((num_obs, mem))
#     for i in range(mem):
#         for j in range(num_obs):
#             obs_vect_perturbed[j, i] = obs_vect_filtered[j]

#     # Observation error covariance matrix
#     cov_obs = R[:, :]

#     # Calculating the observation operator matrix
#     h_matrix = h_operator(nx, obs_vect)

#     # Calculating the Kalman gain
#     k_left = cov_prior.dot(np.transpose(h_matrix))
#     k_right_inv = np.linalg.inv(
#         h_matrix.dot(cov_prior).dot(np.transpose(h_matrix)) + cov_obs
#     )
#     kalman_gain = k_left.dot(k_right_inv)

#     # Calculating the innovation
#     innovation = obs_vect_perturbed - h_matrix.dot(prior_vect)

#     # Calculating the posterior ensemble
#     posterior_vect = prior_vect + kalman_gain.dot(innovation)
#     mean_posterior = np.mean(posterior_vect, axis=1)
#     cov_posterior = np.cov(posterior_vect)

#     # Recalculating weights
#     w_t = gaussian_mixt(
#         w_prev, num_obs, posterior_vect, obs_vect_perturbed, h_matrix, R
#     )

#     # Evaluating degeneracy and calculating the bridging alpha
#     N_eff = 1 / np.sum(w_t**2)
#     alpha = N_eff / mem

#     # Adjusting weights
#     w_t = w_t * alpha + (1 - alpha) * (1 / mem)

#     # Resampling if necessary
#     resamp = 0
#     if N_eff < nc_threshold:
#         J = randsample(mem, w_t)
#         epsc = np.random.normal(0, 0.1, mem)
#         for i in range(mem):
#             posterior_vect[:, i] = (
#                 posterior_vect[:, int(J[i])] + np.sqrt(np.diag(cov_posterior)) * epsc[i]
#             )
#         cov_posterior = (h**2) * np.cov(posterior_vect)
#         resamp = 1

#     # Result output
#     agmf_output = {
#         "posterior": posterior_vect,
#         "kalman_gain": kalman_gain,
#         "innovation": innovation,
#         "mean_post": mean_posterior,
#         "cov_post": cov_posterior,
#         "weights": w_t,
#         "alpha": alpha,
#     }

#     return agmf_output


class AdaptiveGaussianMixtureFilter(BaseDataAssimilationMethod):
    def __init__(
        self,
        ensemble_size: int,
        R: np.ndarray,
        w_prev: np.ndarray,
        nc_threshold: float,
        obs_operator: ObservationOperator,
        forward_operator: BaseForwardModel,
        inflation_factor: float = 1.0,
        localization_distance: Optional[int] = None,
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
        super().__init__(obs_operator, forward_operator)
        self.ensemble_size = ensemble_size
        self.R = R
        self.inflation_factor = inflation_factor
        self.w_prev = w_prev
        self.nc_threshold = nc_threshold
        self.localization_distance = localization_distance
        self.num_states = forward_operator.num_states
        self.state_dim = forward_operator.state_dim

        if self.localization_distance is None:
            self.localization = lambda x: x
        else:
            self.localization = lambda x: distance_based_localization(
                self.localization_distance, self.state_dim, x  # type: ignore[arg-type]
            )

    def _analysis_step(
        self, prior_ensemble: np.ndarray, obs_vect: np.ndarray
    ) -> np.ndarray:
        """Analysis step of the Adaptive Gaussian Mixture Filter."""

        # Preparing the prior state vector (ensemble matrix)
        prior_ensemble = prior_ensemble.reshape(self.ensemble_size, -1).T

        # Calculating the mean and covariance of the prior
        cov_prior = (self.inflation_factor**2) * np.cov(prior_ensemble)
        cov_prior = self.localization(cov_prior)

        # Filter and perturb the observation vector
        obs_vect_perturbed = obs_vect + np.random.multivariate_normal(
            np.zeros(self.obs_operator.num_obs), self.R, size=self.ensemble_size
        )
        obs_vect_perturbed = obs_vect_perturbed.T

        # Calculating the observation operator matrix
        obs_matrix = self.obs_operator.obs_matrix

        # Calculate the Kalman gain
        k_left = cov_prior @ obs_matrix.T
        k_right = obs_matrix @ cov_prior @ obs_matrix.T + self.R

        # K_left * K_right^-1
        kalman_gain = np.linalg.solve(k_right, k_left.T).T

        # Calculate the innovation
        innovation = obs_vect_perturbed - obs_matrix @ prior_ensemble

        # Calculate the posterior ensemble
        posterior_ensemble = prior_ensemble + kalman_gain @ innovation

        cov_posterior = np.cov(posterior_ensemble)

        # Recalculating weights
        w_t = gaussian_mixt(
            self.w_prev,
            self.obs_operator.num_obs,
            posterior_ensemble,
            obs_vect_perturbed,
            obs_matrix,
            self.R,
        )

        # Evaluating degeneracy and calculating the bridging alpha
        N_eff = 1 / np.sum(w_t**2)
        alpha = N_eff / self.ensemble_size

        # Adjusting weights
        w_t = w_t * alpha + (1 - alpha) * (1 / self.ensemble_size)
        self.w_prev = w_t

        # Resampling if necessary
        resamp = 0
        if N_eff < self.nc_threshold:
            J = randsample(self.ensemble_size, w_t)
            epsc = np.random.normal(0, 0.1, self.ensemble_size)
            for i in range(self.ensemble_size):
                posterior_ensemble[:, i] = (
                    posterior_ensemble[:, int(J[i])]
                    + np.sqrt(np.diag(cov_posterior)) * epsc[i]
                )
            cov_posterior = (self.inflation_factor**2) * np.cov(posterior_ensemble)
            resamp = 1

        # Result output
        # agmf_output = {
        #     "posterior": posterior_vect,
        #     "kalman_gain": kalman_gain,
        #     "innovation": innovation,
        #     "mean_post": mean_posterior,
        #     "cov_post": cov_posterior,
        #     "weights": w_t,
        #     "alpha": alpha,
        # }

        posterior_ensemble = posterior_ensemble.T
        posterior_ensemble = posterior_ensemble.reshape(
            self.ensemble_size, self.num_states, self.state_dim
        )

        return posterior_ensemble
