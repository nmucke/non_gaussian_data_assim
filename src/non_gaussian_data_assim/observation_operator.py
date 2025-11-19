import numpy as np
from numpy.typing import NDArray


def h_operator(nx: int, obs_vect: np.ndarray) -> np.ndarray:
    """
    Create the observation operator matrix H.

    Args:
    nx (int): Size of the state vector.
    obs_vect (numpy.array): Observation vector, where -999 indicates missing data.

    Returns:
    numpy.array: The observation operator matrix.
    """
    # Identifying indices of valid observations (not -999)
    index_obs = np.where(obs_vect > -999)[0]
    num_obs = len(index_obs)

    # Initializing the H matrix with zeros
    h_matrix = np.zeros((num_obs, nx))

    # Setting 1 at positions corresponding to actual observations
    for i in range(num_obs):
        h_matrix[i, index_obs[i]] = 1

    return h_matrix
