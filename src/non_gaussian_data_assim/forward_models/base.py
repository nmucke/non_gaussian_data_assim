import functools
from abc import ABC, abstractmethod
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from non_gaussian_data_assim.time_integrators import (
    rollout,
    rollout_with_model_integration_steps,
)


class BaseForwardModel(ABC):
    """Base class for forward models."""

    def __init__(
        self,
        dt: float,
        model_integration_steps: int,
        state_dim: int,
        num_states: int = 1,
    ):
        """Initialize the forward model."""
        self.num_states = num_states
        self.state_dim = state_dim
        self.dt = dt
        self.model_integration_steps = model_integration_steps

        # Cache jitted callables so repeated DA steps reuse compiled functions
        # instead of re-jitting a fresh closure on every invocation.
        self._call_cache: dict[tuple[bool, bool], Callable] = {}
        self._rollout_cache: dict[tuple[bool, int], Callable] = {}

    @abstractmethod
    def one_step(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute one inner step of the forward model."""
        raise NotImplementedError

    def _get_call_fn(
        self, return_model_integration_steps: bool, is_ensemble: bool
    ) -> Callable:
        """Return a cached jitted rollout for the given flags."""
        key = (return_model_integration_steps, is_ensemble)
        if key not in self._call_cache:
            # include_initial_state only affects the output when
            # return_model_integration_steps is True (otherwise rollout returns
            # just the last step and ignores the flag). For this inner rollout we
            # never want the initial state prepended to the trajectory, so it is
            # always False. (Previously written as `not return_model_integration_steps`,
            # which was equivalent but confusing: the `True` branch only occurred
            # in the non-trajectory case where the flag is ignored anyway.)
            rollout_fn = rollout(
                self.one_step,
                self.model_integration_steps,
                return_model_integration_steps=return_model_integration_steps,
                include_initial_state=False,
            )
            if is_ensemble:
                rollout_fn = jax.vmap(rollout_fn)
            self._call_cache[key] = jax.jit(rollout_fn)
        return self._call_cache[key]

    def __call__(
        self,
        x: jnp.ndarray,
        _: None = None,
        return_model_integration_steps: bool = False,
        is_ensemble: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward the model.

        Args:
            x: State array of shape [ensemble, num_states, state_dim]

        Returns:
            tuple: (new_time, new_state) where new_state has shape [ensemble, num_states, state_dim]
        """

        rollout_fn = self._get_call_fn(return_model_integration_steps, is_ensemble)
        return rollout_fn(x)

    def _get_rollout_fn(
        self, return_model_integration_steps: bool, data_assimilation_steps: int
    ) -> Callable:
        """Return a cached jitted outer rollout for the given flags/steps."""
        key = (return_model_integration_steps, data_assimilation_steps)
        if key not in self._rollout_cache:
            if return_model_integration_steps:
                inner_rollout_fn = functools.partial(
                    self.__call__,
                    return_model_integration_steps=True,
                )
                rollout_fn = rollout_with_model_integration_steps(
                    inner_rollout_fn,
                    data_assimilation_steps,
                    include_initial_state=False,
                )
            else:
                rollout_fn = rollout(
                    self.__call__,
                    data_assimilation_steps,
                    return_model_integration_steps=True,
                    include_initial_state=True,
                )
            self._rollout_cache[key] = jax.jit(rollout_fn)
        return self._rollout_cache[key]

    def rollout(
        self,
        x: jnp.ndarray,
        data_assimilation_steps: int,
        return_model_integration_steps: bool = False,
    ) -> jnp.ndarray:
        """
        Outer rollout the model for the given number of outer steps.

        Args:
            x: State array of shape [ensemble, num_states, state_dim]
            data_assimilation_steps: Number of outer steps to rollout

        Returns:
            State array of shape [ensemble, num_states, state_dim]
        """

        rollout_fn = self._get_rollout_fn(
            return_model_integration_steps, data_assimilation_steps
        )
        return jax.vmap(rollout_fn)(x)
