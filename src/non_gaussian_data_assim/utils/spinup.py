"""Ensemble spinup helpers."""

import logging

import jax.numpy as jnp

from non_gaussian_data_assim.forward_models.base import BaseForwardModel

logger = logging.getLogger(__name__)


def spinup_ensemble(
    ensemble: jnp.ndarray,
    forward_model: BaseForwardModel,
    spinup_steps: int,
    return_model_integration_steps: bool = False,
) -> jnp.ndarray:
    """Roll an ensemble forward for `spinup_steps`.

    By default returns the ensemble at the end of the spinup window, with the
    time axis dropped (shape matches the input `ensemble`). If
    `return_model_integration_steps` is True, the full trajectory over all model
    integration steps is returned with shape
    [ensemble, time, num_states, state_dim].

    It spins up spinup_steps * model_integration_steps.
    """
    integration_steps = spinup_steps * forward_model.model_integration_steps
    seconds = integration_steps * forward_model.dt
    logger.info(
        f"Model spinup for {integration_steps} model integration steps ({seconds} s)"
    )

    # --- Rollout model
    rolled = forward_model.rollout(
        ensemble,
        spinup_steps,
        return_model_integration_steps=return_model_integration_steps,
    )
    # Shape rolled: [ensemble, time, nr.state, state-dim]

    if return_model_integration_steps:
        return rolled

    return rolled[:, -1]
