from typing import Any, Callable, Dict

import numpy as np

from non_gaussian_data_assim.da_methods.base import BaseDataAssimilationMethod
from non_gaussian_data_assim.da_methods.pff import grad_log_post
from non_gaussian_data_assim.localization import localization
from non_gaussian_data_assim.observation_operator import h_operator


def pff(
    mem: int,
    n_states: int,
    ensemble: np.ndarray,
    obs_vect: np.ndarray,
    N: int,
    r_influ: int,
    R: np.ndarray,
) -> Dict[str, Any]:
    """
    Implement the Particle Flow Filter.

    Args:
    mem (int): Number of ensemble members.
    n_states (int): Number of states.
    ensemble (numpy.array): Initial ensemble of states.
    obs_vect (numpy.array): Observation vector.
    R (numpy.array): Observation error covariance matrix.
    N (int): The number of grid points.
    r_influ (int): Radius of influence for localization -- grid cells.
    """
    index_obs = np.where(obs_vect > -999)[0]
    B = np.cov(ensemble)
    # Apply localization to the prior covariance matrix
    B = localization(r_influ, N, B)

    x0_mean = np.mean(ensemble, axis=1)

    # Pseudo-time flow parameters
    s = 0
    max_s = 100
    ds = 0.05 / 10
    alpha = 0.05 / 10  # Tuning parameter for the covariance of the kernel

    x_s = ensemble.copy()
    python_pseudoflow = np.zeros((n_states, mem, max_s + 1))
    python_pseudoflow[:, :, 0] = x_s.copy()

    n_obs = np.sum(obs_vect > -999)
    R_inv = np.linalg.inv(R)

    # Pseudo-time for data assimilation
    while s < max_s:

        H = np.zeros((n_obs, n_states))
        Hx = np.zeros((n_obs, mem))
        dHdx = np.zeros((n_obs, n_states, mem))
        dpdx = np.zeros((n_states, mem))

        for i in range(mem):
            H = h_operator(n_states, obs_vect)
            Hx[:, :] = x_s[index_obs, :]
            y = np.ones((n_obs, 1))
            y[:, 0] = obs_vect[index_obs]
            y_i = np.ones((n_obs, 1))
            y_i[:, 0] = Hx[:, i]

            dpdx[:, i] = grad_log_post(H, R, R_inv, y, y_i, B, x_s[:, i], x0_mean)

        # Kernel calculation
        B_d = np.zeros((n_states))
        for d in range(n_states):
            B_d[d] = B[d, d]

        kernel = np.zeros((n_states, mem, mem))
        dkdx = np.zeros((n_states, mem, mem))
        I_f = np.zeros((n_states, mem))

        for i in range(mem):
            for j in range(i, mem):
                kernel[:, i, j] = np.exp(
                    (-1 / 2) * ((x_s[:, i] - x_s[:, j]) ** 2) / (alpha * B_d[:])
                )
                dkdx[:, i, j] = ((x_s[:, i] - x_s[:, j]) / alpha) * kernel[:, i, j]
                if j != i:
                    kernel[:, i, j] = kernel[:, j, i]
                    dkdx[:, i, j] = -dkdx[:, j, i]

            attractive_term = (1 / mem) * (kernel[:, i, :] * dpdx)
            repelling_term = (1 / mem) * dkdx[:, i, :]
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

    return pff_pseudoflow


class ParticleFlowFilterLocalization(BaseDataAssimilationMethod):
    def __init__(
        self,
        mem: int,
        nx: int,
        R: np.ndarray,
        N: int,
        r_influ: int,
        obs_operator: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """
        Initialize the Particle Flow Filter with localization.
        Args:
        mem (int): Number of ensemble members.
        nx (int): Size of the state vector.
        R (numpy.array): Observation error covariance matrix.
        obs_operator (Callable[[np.ndarray], np.ndarray]): Observation operator.
        """
        super().__init__(obs_operator)
        self.mem = mem
        self.nx = nx
        self.R = R
        self.N = N
        self.r_influ = r_influ

    def _assimilate_data(
        self, prior_ensemble: np.ndarray, obs_vect: np.ndarray
    ) -> np.ndarray:
        return pff(
            mem=self.mem,
            n_states=self.nx,
            ensemble=prior_ensemble,
            obs_vect=obs_vect,
            R=self.R,
            N=self.N,
            r_influ=self.r_influ,
        )["posterior"]
