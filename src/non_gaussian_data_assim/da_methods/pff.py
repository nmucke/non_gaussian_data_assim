import pdb
from typing import Any, Callable, Dict

import numpy as np

from non_gaussian_data_assim.da_methods.base import BaseDataAssimilationMethod
from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.observation_operator import ObservationOperator


def grad_log_post(
    H: np.ndarray,
    R: np.ndarray,
    R_inv: np.ndarray,
    y: np.ndarray,
    y_i: np.ndarray,
    B: np.ndarray,
    x_s_i: np.ndarray,
    x0_mean: np.ndarray,
) -> np.ndarray:
    """
    Calculate the gradient of the log posterior distribution.

    Args:
    H (numpy.array): Observation operator matrix.
    R (numpy.array): Observation error covariance matrix.
    R_inv (numpy.array): Inverse of R.
    y (numpy.array): Observation vector.
    y_i (numpy.array): Individual observation vector for a particle.
    B (numpy.array): Covariance matrix of the ensemble.
    x_s_i (numpy.array): Current state of a particle.
    x0_mean (numpy.array): Mean state of the prior distribution.

    Returns:
    numpy.array: Gradient of the log posterior.
    """
    obs_part = B.dot(H.transpose()).dot(R_inv).dot(y - y_i)[:, 0]
    prior_part = x_s_i - x0_mean
    grad_log_post_est = obs_part - prior_part

    return grad_log_post_est


# def pff(
#     mem: int,
#     n_states: int,
#     ensemble: np.ndarray,
#     obs_vect: np.ndarray,
#     R: np.ndarray,
# ) -> Dict[str, Any]:
#     """
#     Implement the Particle Flow Filter.

#     Args:
#     mem (int): Number of ensemble members.
#     n_states (int): Number of states.
#     ensemble (numpy.array): Initial ensemble of states.
#     obs_vect (numpy.array): Observation vector.
#     R (numpy.array): Observation error covariance matrix.

#     Returns:
#     dict: Dictionary containing the posterior ensemble, mean, and covariance.
#     """
#     index_obs = np.where(obs_vect > -999)[0]
#     B = np.cov(ensemble)
#     x0_mean = np.mean(ensemble, axis=1)

#     # Pseudo-time flow parameters
#     s = 0
#     max_s = 100
#     ds = 0.05 / 10
#     alpha = 0.05 / 10  # Tuning parameter for the covariance of the kernel

#     x_s = ensemble.copy()
#     python_pseudoflow = np.zeros((n_states, mem, max_s + 1))
#     python_pseudoflow[:, :, 0] = x_s.copy()

#     n_obs = np.sum(obs_vect > -999)
#     R_inv = np.linalg.inv(R)

#     # Pseudo-time for data assimilation
#     while s < max_s:

#         H = np.zeros((n_obs, n_states))
#         Hx = np.zeros((n_obs, mem))
#         dHdx = np.zeros((n_obs, n_states, mem))
#         dpdx = np.zeros((n_states, mem))

#         for i in range(mem):
#             H = h_operator(n_states, obs_vect)
#             Hx[:, :] = x_s[index_obs, :]
#             y = np.ones((n_obs, 1))
#             y[:, 0] = obs_vect[index_obs]
#             y_i = np.ones((n_obs, 1))
#             y_i[:, 0] = Hx[:, i]

#             dpdx[:, i] = grad_log_post(H, R, R_inv, y, y_i, B, x_s[:, i], x0_mean)

#         # Kernel calculation
#         B_d = np.zeros((n_states))
#         for d in range(n_states):
#             B_d[d] = B[d, d]

#         kernel = np.zeros((n_states, mem, mem))
#         dkdx = np.zeros((n_states, mem, mem))
#         I_f = np.zeros((n_states, mem))

#         for i in range(mem):
#             for j in range(i, mem):
#                 kernel[:, i, j] = np.exp(
#                     (-1 / 2) * ((x_s[:, i] - x_s[:, j]) ** 2) / (alpha * B_d[:])
#                 )
#                 dkdx[:, i, j] = ((x_s[:, i] - x_s[:, j]) / alpha) * kernel[:, i, j]
#                 if j != i:
#                     kernel[:, i, j] = kernel[:, j, i]
#                     dkdx[:, i, j] = -dkdx[:, j, i]

#             attractive_term = (1 / mem) * (kernel[:, i, :] * dpdx)
#             repelling_term = (1 / mem) * dkdx[:, i, :]
#             I_f[:, i] = np.sum(attractive_term + repelling_term, axis=1)

#         # Update the state vector for next pseudo time step
#         fs = I_f
#         x_s += ds * fs
#         python_pseudoflow[:, :, s + 1] = x_s

#         s += 1

#     # Gathering final results
#     posterior_vect = python_pseudoflow[:, :, -1]
#     mean_posterior = np.mean(posterior_vect, axis=1)
#     cov_posterior = np.cov(posterior_vect)

#     pff_pseudoflow = {
#         "posterior": posterior_vect,
#         "mean_post": mean_posterior,
#         "cov_post": cov_posterior,
#     }

#     return pff_pseudoflow


class ParticleFlowFilter(BaseDataAssimilationMethod):
    def __init__(
        self,
        ensemble_size: int,
        R: np.ndarray,
        obs_operator: ObservationOperator,
        forward_operator: BaseForwardModel,
    ) -> None:
        """
        Initialize the Particle Flow Filter.
        Args:
        mem (int): Number of ensemble members.
        nx (int): Size of the state vector.
        R (numpy.array): Observation error covariance matrix.
        obs_operator (Callable[[np.ndarray], np.ndarray]): Observation operator.
        """
        super().__init__(obs_operator, forward_operator)
        self.ensemble_size = ensemble_size
        self.R = R
        self.num_states = forward_operator.num_states
        self.state_dim = forward_operator.state_dim
        self.dofs = self.num_states * self.state_dim

    def _analysis_step(
        self, prior_ensemble: np.ndarray, obs_vect: np.ndarray
    ) -> np.ndarray:

        prior_ensemble = prior_ensemble.reshape(self.ensemble_size, -1).T

        B = np.cov(prior_ensemble)
        x0_mean = np.mean(prior_ensemble, axis=1)

        # Pseudo-time flow parameters
        s = 0
        max_s = 100
        ds = 0.05 / 10
        alpha = 0.05 / 10  # Tuning parameter for the covariance of the kernel

        x_s = prior_ensemble.copy()
        python_pseudoflow = np.zeros((self.dofs, self.ensemble_size, max_s + 1))
        python_pseudoflow[:, :, 0] = x_s.copy()

        n_obs = np.sum(obs_vect > -999)
        R_inv = np.linalg.inv(self.R)

        # Pseudo-time for data assimilation
        while s < max_s:

            H = np.zeros((n_obs, self.dofs))
            Hx = np.zeros((n_obs, self.ensemble_size))
            dHdx = np.zeros((n_obs, self.dofs, self.ensemble_size))
            dpdx = np.zeros((self.dofs, self.ensemble_size))

            for i in range(self.ensemble_size):
                H = self.obs_operator.obs_matrix
                Hx[:, :] = self.obs_operator(x_s).T

                # x_s[index_obs, :]
                y = np.ones((n_obs, 1))
                y[:, 0] = obs_vect
                y_i = np.ones((n_obs, 1))
                y_i[:, 0] = Hx[:, i]

                dpdx[:, i] = grad_log_post(
                    H, self.R, R_inv, y, y_i, B, x_s[:, i], x0_mean
                )

            # Kernel calculation
            B_d = np.zeros((self.dofs))
            for d in range(self.dofs):
                B_d[d] = B[d, d]

            kernel = np.zeros((self.dofs, self.ensemble_size, self.ensemble_size))
            dkdx = np.zeros((self.dofs, self.ensemble_size, self.ensemble_size))
            I_f = np.zeros((self.dofs, self.ensemble_size))

            for i in range(self.ensemble_size):
                for j in range(i, self.ensemble_size):
                    kernel[:, i, j] = np.exp(
                        (-1 / 2) * ((x_s[:, i] - x_s[:, j]) ** 2) / (alpha * B_d[:])
                    )
                    dkdx[:, i, j] = ((x_s[:, i] - x_s[:, j]) / alpha) * kernel[:, i, j]
                    if j != i:
                        kernel[:, i, j] = kernel[:, j, i]
                        dkdx[:, i, j] = -dkdx[:, j, i]

                attractive_term = (1 / self.ensemble_size) * (kernel[:, i, :] * dpdx)
                repelling_term = (1 / self.ensemble_size) * dkdx[:, i, :]
                I_f[:, i] = np.sum(attractive_term + repelling_term, axis=1)

            # Update the state vector for next pseudo time step
            fs = I_f
            x_s += ds * fs
            python_pseudoflow[:, :, s + 1] = x_s

            s += 1

        # Gathering final results
        posterior_vect = python_pseudoflow[:, :, -1]
        mean_posterior = np.mean(posterior_vect, axis=1)
        cov_posterior = np.cov(posterior_vect)

        pff_pseudoflow = {
            "posterior": posterior_vect,
            "mean_post": mean_posterior,
            "cov_post": cov_posterior,
        }

        pdb.set_trace()

        return pff_pseudoflow
