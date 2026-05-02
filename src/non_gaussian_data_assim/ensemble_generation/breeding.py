""" Run breeding vectors in for any given moldel (model step taken as input callable) """

import random
from collections.abc import Callable

import numpy as np

from non_gaussian_data_assim.ensemble_generation.random_noise import RandomNoise


class BreedingVector:
    """Creates ensemble perturbation based on breeding vector"""

    pass
    # def __init__(
    #     self,
    #     forward_model: Callable,
    #     n_windows: int,
    #     dt_window: int,
    #     sigma_level: float,
    #     rescaling: str = "global",
    # ) -> None:
    #     """
    #     n_windows: How many breeding windows (cycles) are run
    #     dt_window: How many model-steps per window
    #         --> Total model-steps = n_windows * dt_window
    #     sigma_level: Level of std.dev that field is rescaled to

    #     NOTE: For now breeding is only implemented for single-variable runs
    #     """

    #     # --- Ensure that forward_model has method 'rollout'absinvert_op = getattr(foo, "rollout", None)
    #     if not Callable(getattr(forward_model, "rollout", None)):
    #         raise ValueError(
    #             "'forward_model' must have method 'rollout, but does not have it!"
    #         )

    #     def _init_bv(self):
    #         pass


# def bred_vector_prior_ensemble(
#     rng_key: jax.Array,
#     ensemble_size: int,
#     num_states: int,
#     state_dim: int,
#     initial_perturbation_scale: float,
#     rescale_amplitude: float,
#     breeding_cycles: int,
#     cycle_steps: int,
#     periodic_boundary: bool = False,
#     base_state: jax.Array | None = None,
#     forward_model: Any | None = None,
# ) -> jnp.ndarray:
#     """Create a prior ensemble using bred-vector perturbations."""

#     if base_state is None:
#         raise ValueError("Breeding requires base_state.")

#     if forward_model is None:
#         raise ValueError("Breeding requires forward_model.")

#     perturbations = random_normal_prior_ensemble(
#         rng_key=rng_key,
#         ensemble_size=ensemble_size,
#         num_states=num_states,
#         state_dim=state_dim,
#         scale=initial_perturbation_scale,
#         periodic_boundary=periodic_boundary,
#     )

#     ensemble = base_state + perturbations
#     control = base_state

#     for _ in range(breeding_cycles):
#         control_rollout = forward_model.rollout(
#             control,
#             cycle_steps,
#             return_model_integration_steps=True,
#         )

#         ensemble_rollout = forward_model.rollout(
#             ensemble,
#             cycle_steps,
#             return_model_integration_steps=True,
#         )

#         control = control_rollout[:, -1]
#         ensemble = ensemble_rollout[:, -1]

#         perturbations = ensemble - control

#         flat = perturbations.reshape(ensemble_size, -1)
#         norms = jnp.std(flat, axis=1).reshape(ensemble_size, 1, 1)

#         perturbations = perturbations / (norms + 1e-12)
#         perturbations = perturbations * rescale_amplitude

#         ensemble = control + perturbations

#         if periodic_boundary:
#             ensemble = ensemble.at[:, :, -1].set(ensemble[:, :, 0])

#     return ensemble
