from typing import Optional

import jax
import jax.numpy as jnp

from non_gaussian_data_assim.perturbations.base import BasePerturbation


class WhiteNoise(BasePerturbation):
    """Spatially white Gaussian noise with standard deviation `scale`."""

    def __init__(self, num_states: int, state_dim: int, scale: float) -> None:
        super().__init__(name="white_noise", num_states=num_states, state_dim=state_dim)
        self.scale = scale

    def sample(
        self,
        rng_key: jax.Array,
        ensemble_size: int,
        bg_profile: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # --- Since fields are uncorrelated, we can create for all members, all states and entire state-dim at once
        shape = (ensemble_size, self.num_states, self.state_dim)
        ensemble = jax.random.normal(rng_key, shape) * self.scale

        if bg_profile is not None:
            ensemble = self._add_ensemble_to_bestguess_profile(
                bg_profile=bg_profile, ensemble=ensemble
            )

        return ensemble
