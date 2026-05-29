from typing import Optional

import jax
import jax.numpy as jnp

from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.initial_profiles import BaseProfile
from non_gaussian_data_assim.perturbations import BasePerturbation, WhiteNoise
from non_gaussian_data_assim.utils.spinup import spinup_ensemble

best_guess_perturbation_mapping = {
    None: lambda x: 0.0,
    "natural_variability": lambda x: jnp.std(x[0], ddof=1),
}


class InitialEnsembleGenerator:
    """Generate a prior ensemble by perturbing a center field.

    The center is either:

        - a supplied ``best_guess`` state (already spun up), or
        - a fresh sample from ``initial_profile`` (then optionally spun up).

    Periodicity is enforced post-hoc if requested.
    """

    def __init__(
        self,
        perturbation: Optional[BasePerturbation] = None,
        forward_model: Optional[BaseForwardModel] = None,
        initial_profile: Optional[BaseProfile] = None,
        spinup_steps: int = 0,
        periodic: bool = False,
        use_best_guess: bool = False,
        best_guess_perturbation: Optional[str] = None,
    ) -> None:
        self.perturbation = perturbation
        self.forward_model = forward_model
        self.initial_profile = initial_profile
        self.spinup_steps = spinup_steps
        self.periodic = periodic
        self.use_best_guess = use_best_guess
        self.best_guess_perturbation = best_guess_perturbation_mapping[
            best_guess_perturbation
        ]

        self.num_states = forward_model.num_states  # type: ignore
        self.state_dim = forward_model.state_dim  # type: ignore

    def sample(
        self,
        rng_key: jax.Array,
        ensemble_size: int,
        best_guess: Optional[jnp.ndarray] = None,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:

        center_key, pert_key, best_guess_key = jax.random.split(rng_key, 3)

        # --- 1. Center the ensemble on a best-guess or a fresh profile sample.
        if best_guess is not None:
            best_guess_std = self.best_guess_perturbation(best_guess)
            noise = WhiteNoise(
                num_states=self.num_states,
                state_dim=self.state_dim,
                scale=best_guess_std,
            ).sample(best_guess_key, ensemble_size=1)
            center = best_guess + noise
            self.best_guess_profile = center
        else:
            if self.initial_profile is None:
                raise ValueError(
                    "InitialEnsembleGenerator requires either a `best_guess` state "
                    "or an `initial_profile` to center the ensemble on."
                )
            center = self.initial_profile.sample(rng_key=center_key)
            self.best_guess_profile = center

        # --- 2. Perturb around the center (Breeding may also return diagnostics).
        if self.perturbation:
            output = self.perturbation.sample(
                rng_key=pert_key,
                ensemble_size=ensemble_size,
                bg_profile=center,
            )
        else:
            output = jnp.broadcast_to(center, (ensemble_size,) + center.shape[1:])

        if isinstance(output, tuple):
            ensemble, diagnostics = output
        else:
            ensemble, diagnostics = output, None

        # --- 3. Spin up only when built from a profile (a best-guess is already spun up).
        if self.spinup_steps and best_guess is None:
            if self.forward_model is None:
                raise ValueError("spinup_steps > 0 requires a `forward_model`.")
            ensemble = spinup_ensemble(
                ensemble=ensemble,
                forward_model=self.forward_model,
                spinup_steps=self.spinup_steps,
            )

        # --- 4. Enforce periodicity if desired.
        if self.periodic:
            ensemble = ensemble.at[..., -1].set(ensemble[..., 0])

        # if diagnostics is not None:
        #     return ensemble, diagnostics
        return ensemble
