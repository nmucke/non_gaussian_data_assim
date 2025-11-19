from abc import abstractmethod
from typing import Callable

import numpy as np


class BaseDataAssimilationMethod:
    """Base class for data assimilation methods."""

    def __init__(self, obs_operator: Callable[[np.ndarray], np.ndarray]):
        """Initialize the data assimilation method."""
        self.obs_operator = obs_operator

    @abstractmethod
    def _assimilate_data(
        self, prior_ensemble: np.ndarray, obs_vect: np.ndarray
    ) -> np.ndarray:
        """Assimilate the data."""
        raise NotImplementedError

    def __call__(self, prior_ensemble: np.ndarray, obs_vect: np.ndarray) -> np.ndarray:
        """Run the data assimilation method."""
        return self._assimilate_data(prior_ensemble, obs_vect)
