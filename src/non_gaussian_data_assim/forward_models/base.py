from abc import abstractmethod

import numpy as np


class BaseForwardModel:
    """Base class for forward models."""

    def __init__(self, dt: float, num_model_steps: int, state_dim: int):
        """Initialize the forward model."""
        self.num_states = 1
        self.state_dim = state_dim
        self.dt = dt
        self.num_model_steps = num_model_steps

    @abstractmethod
    def _one_model_step(self, x: np.ndarray) -> np.ndarray:
        """One inner step of the forward model."""
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward the model."""

        for _ in range(self.num_model_steps):
            x = self._one_model_step(x)
        return x
