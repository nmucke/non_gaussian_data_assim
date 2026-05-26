from abc import ABC, abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp

from non_gaussian_data_assim.forward_models.base import BaseForwardModel


class BasePerturbation(ABC):
    """Base class for ensemble perturbations.

    x0_bg: Best-Guess reference state
        -- if suplied, ensemble-perturbation will be added to it
        -- REQUIRED as starting Conditions for Breeding with shape [num_states, state_dim]
                    or [1, num_states, state_dim].
    """

    def __init__(
        self,
        name: str,
        forward_model: Optional[BaseForwardModel] = None,
        num_states: Optional[int] = None,
        state_dim: Optional[int] = None,
    ) -> None:
        self.name = name
        if forward_model:
            self.num_states = forward_model.num_states
            self.state_dim = forward_model.state_dim
        else:
            self.num_states = num_states
            self.state_dim = state_dim

    def _add_ensemble_to_bestguess_profile(
        self, bg_profile: jnp.ndarray, ensemble: jnp.ndarray
    ) -> jnp.ndarray:
        return bg_profile + ensemble

    @abstractmethod
    def sample(
        self,
        rng_key: jax.Array,
        ensemble_size: int,
        bg_profile: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Return perturbations of shape [ensemble_size, num_states, state_dim].

        Args:
            rng_key: JAX PRNG key.
            ensemble_size: Number of ensemble perturbations.

        Returns:
            Perturbations with shape
            [ensemble_size, num_states, state_dim].
        """
        raise NotImplementedError
