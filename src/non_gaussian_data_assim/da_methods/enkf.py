from typing import Any, Dict

import numpy as np

from non_gaussian_data_assim.observation_operator import h_operator


# Function to implement the Ensemble Kalman Filter
def enkf(
    mem: int,
    nx: int,
    ensemble: np.ndarray,
    obs_vect: np.ndarray,
    R: np.ndarray,
) -> Dict[str, Any]:
    """
    Implement the Ensemble Kalman Filter.

    Args:
    mem (int): Number of ensemble members.
    nx (int): The size of the state vector.
    ensemble (numpy.array): Ensemble of state estimates.
    obs_vect (numpy.array): Observation vector.
    R (numpy.array): Observation error covariance matrix.

    Returns:
    dict: A dictionary containing the posterior ensemble, Kalman gain, innovation,
          mean and covariance of the posterior.
    """
    # Identify indices of valid observations
    index_obs = np.where(obs_vect > -999)[0]
    num_obs = len(index_obs)

    # Prepare the prior state vector (ensemble matrix)
    prior_vect = ensemble

    # Calculate the mean and covariance of the prior
    mean_prior = np.mean(prior_vect, axis=1)
    cov_prior = np.cov(prior_vect)

    # Filter and perturb the observation vector
    obs_vect_filtered = obs_vect[index_obs]
    obs_vect_perturbed = np.zeros((num_obs, mem))
    r_obs_vect = np.diag(R)

    for i in range(mem):
        for j in range(num_obs):
            obs_vect_perturbed[j, i] = obs_vect_filtered[j]

    # Observation error covariance matrix
    cov_obs = R[:, :]

    # Calculate the observation operator matrix
    h_matrix = h_operator(nx, obs_vect)

    # Calculate the Kalman gain
    k_left = cov_prior.dot(np.transpose(h_matrix))
    k_right = h_matrix.dot(cov_prior).dot(np.transpose(h_matrix)) + cov_obs
    k_right_inv = np.linalg.inv(k_right)
    kalman_gain = k_left.dot(k_right_inv)

    # Calculate the innovation
    innovation = obs_vect_perturbed - h_matrix.dot(prior_vect)

    # Calculate the posterior ensemble
    posterior_vect = prior_vect + kalman_gain.dot(innovation)

    # Compute mean and covariance of the posterior
    mean_posterior = np.mean(posterior_vect, axis=1)
    cov_posterior = np.cov(posterior_vect)

    # Return a dictionary of EnKF outputs
    enkf_output = {
        "posterior": posterior_vect,
        "kalman_gain": kalman_gain,
        "innovation": innovation,
        "mean_post": mean_posterior,
        "cov_post": cov_posterior,
    }

    return enkf_output
