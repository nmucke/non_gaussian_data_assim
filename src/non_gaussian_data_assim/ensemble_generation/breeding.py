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
