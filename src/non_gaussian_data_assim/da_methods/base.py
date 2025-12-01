from abc import abstractmethod
from typing import Callable

import numpy as np

from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.observation_operator import ObservationOperator


class BaseDataAssimilationMethod:
    """Base class for data assimilation methods."""

    def __init__(
        self,
        obs_operator: ObservationOperator,
        forward_operator: BaseForwardModel,
    ):
        """Initialize the data assimilation method."""
        self.obs_operator = obs_operator
        self.forward_operator = forward_operator

    def _assimilate_data(
        self, prior_ensemble: np.ndarray, obs_vect: np.ndarray
    ) -> np.ndarray:
        """Assimilate the data."""

        forecast_ensemble = self._forecast_step(prior_ensemble)
        analysis_ensemble = self._analysis_step(forecast_ensemble, obs_vect)
        return analysis_ensemble

    @abstractmethod
    def _analysis_step(
        self, prior_ensemble: np.ndarray, obs_vect: np.ndarray
    ) -> np.ndarray:
        """
        Assimilate the data.

        Args:
            prior_ensemble (np.ndarray): Prior ensemble [Ensemble size, State dimension].
            obs_vect (np.ndarray): Observation vector.

        Returns:
            np.ndarray: Analysis ensemble [Ensemble size, State dimension].
        """
        raise NotImplementedError

    def _forecast_step(self, ensemble: np.ndarray) -> np.ndarray:
        return self.forward_operator(ensemble)

    def __call__(self, prior_ensemble: np.ndarray, obs_vect: np.ndarray) -> np.ndarray:
        """Run the data assimilation method."""
        return self._assimilate_data(prior_ensemble, obs_vect)
