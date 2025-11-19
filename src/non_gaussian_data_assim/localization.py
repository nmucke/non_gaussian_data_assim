import numpy as np
from numpy.typing import NDArray


def localization(r_influ: int, N: int, cov_prior: np.ndarray) -> np.ndarray:
    """
    Apply localization to the covariance matrix.

    Args:
    r_influ (int): The radius of influence for localization -- grid cells.
    N (int): The number of grid points.
    cov_prior (numpy.array): The prior covariance matrix.

    Returns:
    numpy.array: Localized covariance matrix.
    """
    # Create a localization mask with Gaussian-like decay
    tmp = np.zeros((N, N))
    for i in range(1, 3 * r_influ + 1):
        tmp += np.exp(-(i**2) / r_influ**2) * (
            np.diag(np.ones(N - i), i) + np.diag(np.ones(N - i), -i)
        )
    mask = tmp + np.diag(np.ones(N))

    # Apply the localization mask to the prior covariance matrix
    cov_prior_loc = np.zeros(cov_prior.shape)
    for i in range(1, 2):
        for j in range(1, 2):
            cov_prior_loc[(i - 1) * N : i * N, (j - 1) * N : j * N] = np.multiply(
                cov_prior[(i - 1) * N : i * N, (j - 1) * N : j * N], mask
            )

    return cov_prior_loc
