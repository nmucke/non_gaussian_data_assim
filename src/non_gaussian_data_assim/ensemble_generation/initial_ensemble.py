from typing import Optional

import jax
import jax.numpy as jnp

from non_gaussian_data_assim.ensemble_generation.ensemble_perturbations import (
    BasePerturbation,
)
from non_gaussian_data_assim.initial_profiles import BaseProfile, ConstantProfile


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
        bg_profile: Optional[BaseProfile] = None,
    ) -> None:
        if centered_around_bestguess:
            if bg_profile is None or isinstance(bg_profile, jnp.ndarray):
                err_msga = "If centered_around_bestguess is true, a best-guess profile (bg_profile) must be specified"
                err_msgb = f" and must be a jnp.ndarrray. \nCurrently passed as bg_profile: {bg_profile} \n of type: {type(bg_profile)}.\n"
                raise ValueError(err_msga + err_msgb)
        else:
            # If centered_around_bestguess is passed as false, ensure that bg_profiule is None
            bg_profile = None

        self.centered_around_bestguess = centered_around_bestguess
        self.ens_perturbation = ens_perturbation
        self.periodic = periodic

    def sample(
        self,
        rng_key: jax.Array,
        ensemble_size: int,
    ) -> jax.Array:

        rng_key, key = jax.random.split(rng_key)

        ###########################################################
        # All that stuff is WRONG or DEPRECATED

        # ## BUG? In case no profile is passed, make a constant one
        # if self.profile is None:
        #     print(
        #         "\n\n\nWARNING: No Best-guess Profile was passed. This behaviour might not be desirable\n\n\n"
        #     )
        #     self.profile = ConstantProfile(
        #         num_states=self.ens_perturbation.num_states,  # type: ignore[union-attr]
        #         state_dim=self.ens_perturbation.state_dim,  # type: ignore[union-attr]
        #         value=0.0,
        #     )

        # if profile_bg is None:
        #     example_ensemble = self.profile.sample(
        #         rng_key=key, ensemble_size=ensemble_size
        #     )
        #     x0_bg = jnp.empty(example_ensemble.shape)
        # else:
        #     x0_bg = profile_bg

        # if self.ens_perturbation is None:
        #     # raise ValueError("Missing a ens_perturbation perturbation method")
        #     initial_ensemble = self.profile.sample(
        #         rng_key=key, ensemble_size=ensemble_size
        #     )
        # else:
        ####################################################################

        # --- Sample Ensemble Perturbations!
        rng_key, key = jax.random.split(rng_key)
        output = self.ens_perturbation.sample(rng_key=key, ensemble_size=ensemble_size)

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
