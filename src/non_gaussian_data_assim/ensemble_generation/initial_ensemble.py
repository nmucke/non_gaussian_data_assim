from typing import Optional

import jax
import jax.numpy as jnp
from non_gaussian_data_assim.utils.spinup import spinup_ensemble

from non_gaussian_data_assim.ensemble_generation.ensemble_perturbations import (
    BasePerturbation,
)
from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.initial_profiles import BaseProfile, ConstantProfile


class InitialEnsembleGenerator:

    def __init__(
        self,
        forward_model: BaseForwardModel,
        initial_profile: BaseProfile,
        perturbation: BasePerturbation,
        spinup: bool = False,
        periodic: bool = False,
        use_best_guess: bool = False
    ) -> None:

        self.forward_model = forward_model
        self.perturbation = perturbation
        self.spinup = spinup
        self.periodic = periodic


    def _spin_up(self, state: jnp.array, spinup_steps: int=0):
        spun_up_state = spinup_ensemble(
            ensemble=state,
            forward_model=self.forward_model,
            spinup_steps=spinup_steps,
        )
        return spun_up_state

    def _get_natural_variability(self, state: jnp.array):
        # -- Calculate spread
        return jnp.std(
            state[0],
            ddof=1,
        )  #  axis=0

    def sample():


        return ensemble





class InitialEnsemble:
    """

    Build a initial ensemble in one of two ways:

        1) Best-guess fields is present:
            Best-guess is a field of type: BaseProfile and ensemble perturbations are added to it

        2) Best-guess is NOT present:
            In this case ensemble-perturbations are the 'full' ensemble members

    Optional:
        - Periodic boundary conditions are enforced post-hoc.
    """

    def __init__(
        self,
        centered_around_bestguess: bool,
        ens_perturbation: BasePerturbation,
        periodic: bool,
    ) -> None:

        self.centered_around_bestguess = centered_around_bestguess
        self.ens_perturbation = ens_perturbation
        self.periodic = periodic

    def sample(
        self,
        rng_key: jax.Array,
        ensemble_size: int,
        bg_profile: Optional[jnp.ndarray] = None,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:

        if self.centered_around_bestguess and bg_profile is None:
            err_msga = "If centered_around_bestguess is true, a best-guess profile (bg_profile) must be specified"
            err_msgb = f" and must be a jnp.ndarrray. \nCurrently passed as bg_profile: {bg_profile} \n of type: {type(bg_profile)}.\n"
            raise ValueError(err_msga + err_msgb)
        if not self.centered_around_bestguess:
            # If centered_around_bestguess is passed as false, ensure that bg_profiule is None
            bg_profile = None

        rng_key, key = jax.random.split(rng_key)

        # --- Sample Ensemble Perturbations!
        rng_key, key = jax.random.split(rng_key)
        output = self.ens_perturbation.sample(
            rng_key=key,
            ensemble_size=ensemble_size,
            bg_profile=bg_profile,
        )

        # --- Catch case: Breeding might return additional metrics
        if isinstance(output, tuple):
            ensgen_diagnostics = True
            ensemble, diagnostics = (
                output[0],
                output[1],
            )
        else:
            ensemble = output
            ensgen_diagnostics = False

        # Enforce periodicity if desried
        if self.periodic:
            ensemble = ensemble.at[..., -1].set(ensemble[..., 0])

        if ensgen_diagnostics:
            return ensemble, diagnostics
        else:
            return ensemble
