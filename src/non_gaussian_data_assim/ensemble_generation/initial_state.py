from typing import Optional

import jax
import jax.numpy as jnp

from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.initial_profiles import BaseProfile
from non_gaussian_data_assim.perturbations import BasePerturbation
from non_gaussian_data_assim.utils.spinup import spinup_ensemble


class InitialState:
    """Sample a single initial state.

    The state is built as:

        initial_profile  ->  (+ optional noise)  ->  (optional spin-up)  ->  (optional periodicity)

    and returned with shape ``[1, num_states, state_dim]``.

    Used both for the ground-truth initial state (profile + spin-up) and for a
    best-guess (profile + noise + spin-up) that an ensemble can be centered on.
    """

    def __init__(
        self,
        initial_profile: BaseProfile,
        forward_model: Optional[BaseForwardModel] = None,
        num_states: Optional[int] = None,
        state_dim: Optional[int] = None,
        perturbation: Optional[BasePerturbation] = None,
        spinup_steps: int = 0,
        periodic: bool = False,
        return_natural_variability: bool = True,
    ) -> None:
        self.initial_profile = initial_profile
        self.forward_model = forward_model
        self.perturbation = perturbation
        self.spinup_steps = spinup_steps
        self.periodic = periodic
        self.retur_nv = return_natural_variability

        if forward_model:
            self.num_states = forward_model.num_states
            self.state_dim = forward_model.state_dim
        else:
            self.num_states = num_states  # type: ignore
            self.state_dim = state_dim  # type: ignore

    def sample(self, rng_key: jax.Array) -> jnp.ndarray:
        profile_key, noise_key = jax.random.split(rng_key)

        # --- Base profile.
        state = self.initial_profile.sample(rng_key=profile_key)

        # --- Optional periodicity.
        if self.periodic:
            state = state.at[..., -1].set(state[..., 0])

        self.state_before_spinup = state

        # --- Optional noise perturbation around the profile.
        if self.perturbation is not None:
            state = self.perturbation.sample(
                rng_key=noise_key, ensemble_size=1, bg_profile=state
            )

        # --- Natural variability fallback for the no-spin-up case; the spin-up
        # branch below overrides it with the value measured over the trajectory.
        self.natural_variability = jnp.std(state, ddof=1)

        # --- Optional spin-up so the raw state lies on the model attractor.
        if self.spinup_steps:
            if self.forward_model is None:
                raise ValueError("spinup_steps > 0 requires a `forward_model`.")
            state = spinup_ensemble(
                ensemble=state,
                forward_model=self.forward_model,
                spinup_steps=self.spinup_steps,
                return_natural_variability=self.retur_nv,
            )
            if self.retur_nv and isinstance(state, tuple):
                state, self.natural_variability = state

        return state
