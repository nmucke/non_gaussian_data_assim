import pdb
from abc import abstractmethod
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from non_gaussian_data_assim.time_integrators import RungeKutta4, rollout


class BaseForwardModel:
    """Base class for forward models."""

    def __init__(
        self,
        dt: float,
        inner_steps: int,
        state_dim: int,
        num_states: int = 1,
    ):
        """Initialize the forward model."""
        self.num_states = num_states
        self.state_dim = state_dim
        self.dt = dt
        self.inner_steps = inner_steps

    @abstractmethod
    def one_step(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute one inner step of the forward model."""
        raise NotImplementedError

    def __call__(self, x: jnp.ndarray, _: None = None) -> jnp.ndarray:
        """
        Forward the model.

        Args:
            x: State array of shape [ensemble, num_states, state_dim]

        Returns:
            tuple: (new_time, new_state) where new_state has shape [ensemble, num_states, state_dim]
        """

        rollout_fn = rollout(
            self.one_step, self.inner_steps, output_only_final_state=True
        )
        rollout_fn = jax.jit(rollout_fn)

        return jax.vmap(rollout_fn)(x)

    def rollout(
        self,
        x: jnp.ndarray,
        outer_steps: int,
    ) -> jnp.ndarray:
        """
        Outer rollout the model for the given number of outer steps.

        Args:
            x: State array of shape [ensemble, num_states, state_dim]
            outer_steps: Number of outer steps to rollout

        Returns:
            State array of shape [ensemble, num_states, state_dim]
        """
        rollout_fn = rollout(
            self.__call__,
            outer_steps,
            output_only_final_state=False,
            include_initial_state=True,
        )
        rollout_fn = jax.jit(rollout_fn)
        return jax.vmap(rollout_fn)(x)
