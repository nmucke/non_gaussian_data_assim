from abc import abstractmethod

import numpy as np


class BaseForwardModel:
    """Base class for forward models."""

    def __init__(self, dt: float, inner_steps: int):
        """Initialize the forward model."""
        self.dt = dt
        self.inner_steps = inner_steps

    @abstractmethod
    def _one_inner_step(self, x: np.ndarray) -> np.ndarray:
        """One inner step of the forward model."""
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward the model."""

        for _ in range(self.inner_steps):
            x = self._one_inner_step(x)
        return x
