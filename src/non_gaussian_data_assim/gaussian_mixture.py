import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


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
    norm_factor = 1 / jnp.sqrt(((2 * np.pi) ** n_obs) * jnp.linalg.det(cov_matrix))

    # Invert the covariance once, then evaluate every ensemble member at once.
    cov_inv = jnp.linalg.inv(cov_matrix)

    # innovations[:, i] = obs - H @ ens_vect[:, i]  -> shape (n_obs, n_ens)
    innovations = obs_vect[:, 0:1] - h_matrix @ ens_vect

    # Per-member quadratic form: innovation_i.T @ cov_inv @ innovation_i
    quad_form = jnp.einsum("ai,ab,bi->i", innovations, cov_inv, innovations)

    prob_dens = norm_factor * jnp.exp(-0.5 * quad_form)
    weight_mixt = prob_dens * weight_vect

    # Normalizing the weights
    weight_final = weight_mixt / jnp.sum(weight_mixt)

    return weight_final
