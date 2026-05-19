"""Ensemble spinup helpers."""

import logging

import jax.numpy as jnp

from non_gaussian_data_assim.forward_models.base import BaseForwardModel

logger = logging.getLogger(__name__)


def spinup_ensemble(
    ensemble: jnp.ndarray,
    forward_model: BaseForwardModel,
    spinup_steps: int,
    get_natural_variablity: bool = False,
) -> jnp.ndarray:
    """Roll an ensemble forward for `spinup_steps` and return the final state.

    Returns the ensemble at the end of the spinup window, with the time axis
    dropped (shape matches the input `ensemble`).

    It spins up spinup_steps * model_integration_steps.
    """
    integration_steps = spinup_steps * forward_model.model_integration_steps
    seconds = integration_steps * forward_model.dt
    logger.info(
        f"Model spinup for {integration_steps} model integration steps ({seconds} s)"
    )

    rolled = forward_model.rollout(
        ensemble, spinup_steps, return_model_integration_steps=get_natural_variablity
    )
    # Shape rolled: [1, model_steps/int.steps, nr.state, state-dim]

    if get_natural_variablity:
        # -- Calculate spread not from first input, as it might still be too dependent on inputted I.C.
        istart = jnp.min(jnp.asarray([spinup_steps // 10, 5]))
        nat_variability = jnp.std(
            rolled[0, istart:, :, :],
            ddof=1,
        )  #  axis=0
        return rolled[:, -1], nat_variability
    else:
        return rolled[:, -1]
