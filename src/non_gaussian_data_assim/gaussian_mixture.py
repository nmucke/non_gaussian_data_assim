import numpy as np
from numpy.typing import NDArray

from non_gaussian_data_assim.observation_operator import h_operator


def gaussian_mixt(
    weight_vect: np.ndarray,
    n_obs: int,
    ens_vect: np.ndarray,
    obs_vect: np.ndarray,
    h_matrix: np.ndarray,
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute the weights for ensemble members using Gaussian Mixture Model.

    Args:
    weight_vect (numpy.array): Current weights of the ensemble members.
    n_obs (int): Number of observations.
    ens_vect (numpy.array): Ensemble matrix.
    obs_vect (numpy.array): Observation vector.
    h_matrix (numpy.array): Observation operator matrix.
    cov_matrix (numpy.array): Covariance matrix of the observations.

    Returns:
    numpy.array: Updated weights for the ensemble members.
    """
    # Normalizing factor for Gaussian probability density function
    norm_factor = 1 / np.sqrt(((2 * np.pi) ** n_obs) * np.linalg.det(cov_matrix))
    weight_mixt = np.zeros(len(weight_vect))
    prob_dens = np.zeros(len(weight_vect))

    # Calculating the weights based on the Gaussian distribution
    for i in range(ens_vect.shape[1]):
        innovation = obs_vect[:, 0] - h_matrix.dot(ens_vect[:, i])
        prob_dens[i] = norm_factor * np.exp(
            -(1 / 2)
            * (
                (np.transpose(innovation)).dot(
                    np.linalg.inv(cov_matrix).dot(innovation)
                )
            )
        )
        weight_mixt[i] = prob_dens[i] * weight_vect[i]

    # Normalizing the weights
    weight_final = weight_mixt / np.sum(weight_mixt)

    return weight_final
