import numpy as np


def distance_based_localization(
    r_influ: int, state_dim: int, cov_prior: np.ndarray
) -> np.ndarray:
    """
    Apply localization to the covariance matrix.

    Args:
    r_influ (int): The radius of influence for localization -- grid cells.
    state_dim (int): The dimension of the state vector.
    cov_prior (numpy.array): The prior covariance matrix.

    Returns:
    numpy.array: Localized covariance matrix.
    """
    # Create a localization mask with Gaussian-like decay
    tmp = np.zeros((state_dim, state_dim))
    for i in range(1, 3 * r_influ + 1):
        tmp += np.exp(-(i**2) / r_influ**2) * (
            np.diag(np.ones(state_dim - i), i) + np.diag(np.ones(state_dim - i), -i)
        )
    mask = tmp + np.diag(np.ones(state_dim))

    # Apply the localization mask to the prior covariance matrix
    cov_prior_loc = np.zeros(cov_prior.shape)
    for i in range(1, 2):
        for j in range(1, 2):
            cov_prior_loc[
                (i - 1) * state_dim : i * state_dim, (j - 1) * state_dim : j * state_dim
            ] = np.multiply(
                cov_prior[
                    (i - 1) * state_dim : i * state_dim,
                    (j - 1) * state_dim : j * state_dim,
                ],
                mask,
            )

    return cov_prior_loc
