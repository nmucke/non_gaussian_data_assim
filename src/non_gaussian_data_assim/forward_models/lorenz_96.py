import pdb
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from non_gaussian_data_assim.forward_models.base import BaseForwardModel


@jax.jit  # type: ignore[misc]
def L96_RHS(x: jnp.ndarray, F: float) -> jnp.ndarray:
    """Lorenz 96 right hand side."""
    return (jnp.roll(x, -1) - jnp.roll(x, 2)) * jnp.roll(x, 1) - x + F


class Lorenz96Model(BaseForwardModel):
    """Base class for forward models."""

    def __init__(
        self,
        forcing_term: float,
        state_dim: int,
        dt: float,
        inner_steps: int,
    ) -> None:
        """Initialize the forward model."""
        super().__init__(dt, inner_steps, state_dim)

        self.forcing_term = forcing_term
        self.num_states = 1

    def RHS(self, x: jnp.ndarray) -> jnp.ndarray:
        """Lorenz 96 right hand side."""
        # Ensure x is 1D for the RHS function
        x_flat = x.flatten()
        return L96_RHS(x_flat, self.forcing_term)
