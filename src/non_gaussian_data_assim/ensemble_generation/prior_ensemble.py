from typing import Optional

import jax
import jax.numpy as jnp

from non_gaussian_data_assim.ensemble_generation.ensemble_perturbations import (
    BasePerturbation,
)
from non_gaussian_data_assim.initial_profiles import BaseProfile, ConstantProfile


class PriorEnsemble:
    """Build a prior ensemble by combining a deterministic profile with optional noise.

    Each ensemble member is `profile.sample(...) + noise.sample(...)`. Periodic
    boundary conditions are enforced post-hoc by tying the last spatial index to
    the first.
    """

    def __init__(
        self,
        profile: Optional[BaseProfile] = None,
        noise: Optional[BasePerturbation] = None,
        periodic: bool = False,
    ) -> None:
        """Configure the ensemble generator.

        Args:
            profile: Deterministic base profile shared across the ensemble.
            noise: Stochastic perturbation added on top of the profile. If
                None, the ensemble is fully deterministic and members are
                identical.
            periodic: If True, set `X[..., -1] = X[..., 0]` so the last
                spatial index matches the first.
        """

        if (profile is None) and (noise is None):
            raise ValueError(
                "PriorEnsemble requires either `profile` or `noise` to be set."
            )
        self.profile = profile
        self.noise = noise
        self.periodic = periodic

    def sample(self, rng_key: jax.Array, ensemble_size: int) -> jnp.ndarray:
        """Draw an ensemble of shape [ensemble_size, num_states, state_dim].

        Args:
            rng_key: JAX PRNG key, split internally between profile and noise.
            ensemble_size: Number of ensemble members to draw.

        Returns:
            Array of shape [ensemble_size, num_states, state_dim].
        """
        rng_key, key = jax.random.split(rng_key)
        if self.profile is None:
            self.profile = ConstantProfile(
                num_states=self.noise.num_states,  # type: ignore[union-attr]
                state_dim=self.noise.state_dim,  # type: ignore[union-attr]
                value=0.0,
            )
        prior_ensemble = self.profile.sample(rng_key=key, ensemble_size=ensemble_size)
        if self.noise is not None:
            rng_key, key = jax.random.split(rng_key)
            prior_ensemble = prior_ensemble + self.noise.sample(
                rng_key=key, ensemble_size=ensemble_size
            )

        if self.periodic:
            prior_ensemble = prior_ensemble.at[..., -1].set(prior_ensemble[..., 0])

        return prior_ensemble
