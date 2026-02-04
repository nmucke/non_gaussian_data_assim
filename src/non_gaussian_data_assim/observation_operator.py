import pdb

import numpy as np
from jax.experimental import sparse
from numpy.typing import NDArray


class ObservationOperator:
    """Observation operator for selecting specific states and indices from a 3D state array.

    The state array is expected to have shape [ensemble_size, num_states, state_dim].
    This operator selects:
    - Specific states from the num_states dimension (via obs_states)
    - Specific indices from the state_dim dimension (via obs_indices)

    The output is flattened to shape [ensemble_size, len(obs_states) * len(obs_indices)].
    """

    def __init__(self, obs_states: np.ndarray, obs_indices: np.ndarray, state_dim: int):
        """
        Initialize the observation operator.

        Args:
            obs_states: Tuple of state indices to observe (from num_states dimension).
                       Example: (0, 1) to observe states at indices 0 and 1.
            obs_indices: Array of indices to observe (from state_dim dimension).
                        Example: np.arange(0, 50, 5) to observe indices 0, 5, 10, ..., 45.
        """
        self.obs_states = obs_states
        self.obs_indices = obs_indices
        self.state_dim = state_dim
        self.num_states = len(obs_states)
        self.num_obs = len(obs_indices) * len(obs_states)

        self.obs_matrix = get_obs_matrix(
            obs_states, obs_indices, self.num_states, self.state_dim
        )
        self.obs_matrix = sparse.BCOO.fromdense(self.obs_matrix)

    def __call__(self, ensemble: np.ndarray) -> np.ndarray:
        """Apply the observation operator to the ensemble."""
        return ensemble[:, self.obs_states, self.obs_indices]


def get_obs_matrix(
    obs_states: np.ndarray,
    obs_indices: np.ndarray,
    num_states: int,
    state_dim: int,
) -> np.ndarray:
    """
    Create an observation operator matrix H that maps flattened state to observations.

    The matrix H has shape [num_states * state_dim, len(obs_states) * len(obs_indices)].
    When multiplied with a flattened state array of shape [..., num_states * state_dim],
    it produces the same output as the ObservationOperator applied to the unflattened state.

    Args:
        obs_states: Array of state indices to observe (from num_states dimension).
                   Example: np.array([0, 1]) to observe states at indices 0 and 1.
        obs_indices: Array of indices to observe (from state_dim dimension).
                    Example: np.arange(0, 50, 5) to observe indices 0, 5, 10, ..., 45.
        num_states: Total number of states in the state array.
        state_dim: Dimension of each state.

    Returns:
        Observation matrix of shape [num_states * state_dim, len(obs_states) * len(obs_indices)].
        The output observations are ordered as: all obs_indices for state0, then all obs_indices for state1, etc.
    """
    obs_states = np.asarray(obs_states)
    obs_indices = np.asarray(obs_indices)

    num_obs_states = len(obs_states)
    num_obs_indices = len(obs_indices)
    total_obs = num_obs_states * num_obs_indices
    total_state_size = num_states * state_dim

    # Initialize the observation matrix
    H = np.zeros((total_state_size, total_obs))

    # For each observed state and each observed index, set the corresponding entry to 1
    obs_col = 0
    for state_idx in obs_states:
        for dim_idx in obs_indices:
            # Calculate the flattened index in the state array
            # Flattening order: state0_dim0, state0_dim1, ..., state0_dimN, state1_dim0, ...
            flattened_idx = state_idx * state_dim + dim_idx
            H[flattened_idx, obs_col] = 1
            obs_col += 1

    return H.T


def obs_operator(nx: int, obs_vect: np.ndarray) -> np.ndarray:
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
