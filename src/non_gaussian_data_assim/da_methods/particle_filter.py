from typing import Any, Callable, Dict

import numpy as np

from non_gaussian_data_assim.da_methods.base import BaseDataAssimilationMethod
from non_gaussian_data_assim.observation_operator import h_operator


def observation_likelihood(
    obs_vect: np.ndarray,
    pred_vect: np.ndarray,
    R: np.ndarray,
) -> Any:
    """
    Compute the likelihood of an observation given a predicted state.

    Args:
    obs_vect (numpy.array): The actual observation vector.
    pred_vect (numpy.array): The predicted state vector.
    R (numpy.array): Observation error covariance matrix.

    Returns:
    float: The likelihood of observing `obs_vect` given the predicted state `pred_vect`.
    """
    # Calculate the residual (difference) between the observation and prediction
    residual = obs_vect - pred_vect

    # Compute the likelihood assuming a Gaussian observation model
    likelihood = np.exp(-0.5 * np.dot(residual.T, np.linalg.inv(R)).dot(residual))
    return likelihood


def particle_filter(
    mem: int,
    nx: int,
    particles: np.ndarray,
    obs_vect: np.ndarray,
    R: np.ndarray,
) -> Dict[str, Any]:
    """
    Implement the Particle Filter algorithm.

    Args:
    mem (int): Number of particles.
    nx (int): Number of state dimensions.
    particles (numpy.array): Array representing the particles.
    obs_vect (numpy.array): Observation vector.
    R (numpy.array): Observation error covariance matrix.

    Returns:
    dict: Dictionary containing updated particles, weights, a flag indicating resampling,
          and the covariance of the updated particles.
    """
    # Find indices of valid observations
    index_obs = np.where(obs_vect > -999)[0]

    # Observation operator matrix
    h_matrix = h_operator(nx, obs_vect)

    # Update weights for each particle based on observation likelihood
    w_t = np.zeros(mem)
    for i in range(mem):
        pred_vect = h_matrix.dot(particles[:, i])
        pred_vect = pred_vect.reshape(len(pred_vect), 1)
        w_t[i] = observation_likelihood(obs_vect[index_obs], pred_vect[:, 0], R)

    # Normalize the weights
    w_t /= np.sum(w_t) + 1e-10

    # Compute covariance before resampling
    cov_posterior = np.cov(particles)

    # Degeneracy evaluation to determine if resampling is needed
    N_eff = 1 / (np.sum(w_t**2) + 1e-10)
    nc_threshold = 0.8 * mem
    resamp = 0
    if N_eff < nc_threshold:
        # Perform resampling
        J = np.random.choice(mem, size=mem, p=w_t)
        for i in range(mem):
            particles[:, i] = particles[:, int(J[i])] + np.sqrt(
                np.diag(cov_posterior)
            ) * np.random.normal(0, 0.1)
        resamp = 1

    # Update the particles with the resampled values
    posterior_vect = particles.copy()
    cov_posterior = np.cov(particles)

    # Construct the return object
    pf_output = {
        "posterior": posterior_vect,
        "weights": w_t,
        "resamp": resamp,
        "cov_post": cov_posterior,
    }

    return pf_output


class ParticleFilter(BaseDataAssimilationMethod):
    def __init__(
        self,
        mem: int,
        nx: int,
        R: np.ndarray,
        obs_operator: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        super().__init__(obs_operator)
        self.mem = mem
        self.nx = nx
        self.R = R

    def _assimilate_data(
        self, prior_ensemble: np.ndarray, obs_vect: np.ndarray
    ) -> np.ndarray:
        return particle_filter(
            mem=self.mem,
            nx=self.nx,
            particles=prior_ensemble,
            obs_vect=obs_vect,
            R=self.R,
        )["posterior"]
