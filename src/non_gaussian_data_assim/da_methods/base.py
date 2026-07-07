import abc
import functools
from abc import abstractmethod
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.localization import distance_based_localization
from non_gaussian_data_assim.observations.observation_operator import (
    ObservationOperator,
)


def make_localization_fn(
    localization_distance: Optional[int],
    state_dim: int,
    num_states: int,
    periodic: bool,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build the covariance localization closure shared by all filters.

    When ``localization_distance`` is ``None`` the returned function is the
    identity; otherwise it applies distance-based localization to the prior
    covariance. Kept in one place so EnKF, AGMF and PFF cannot diverge.
    """
    if localization_distance is None:
        return lambda x: x

    return lambda x: distance_based_localization(
        r_influ=localization_distance,  # type: ignore[arg-type]
        state_dim=state_dim,
        cov_prior=x,
        num_states=num_states,
        periodic=periodic,
    )


def da_rollout(
    da_model: Callable,
    observations: jnp.ndarray,
    rng_key: jax.random.PRNGKey,
    include_initial_state: bool = True,
) -> Callable:
    """Rollout the data assimilation system using the given DA model.

    Args:
        da_model: The data assimilation model callable that takes
            (prior_ensemble, obs_vect, rng_key) and returns the posterior ensemble.
        observations: Array of observations with shape (num_steps, obs_dim).
            The scan forecasts the ensemble one step BEFORE assimilating each
            observation, so observations[0] is the first observation to be
            assimilated (at the first forecast time), not an observation of the
            initial state.
        rng_key: JAX random key for random operations.
        include_initial_state: Whether to include the initial state in the output.
            If True, output has shape (ensemble_size, num_steps, ...).
            If False, output has shape (ensemble_size, num_steps - 1, ...).

    Returns:
        A function that takes the initial ensemble and returns the trajectory.
    """
    initial_rng_key = rng_key

    def scan_fn(
        state: tuple[jnp.ndarray, jax.random.PRNGKey], obs_vect: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Scan function for data assimilation."""
        posterior_state, rng_key = state
        rng_key, key = jax.random.split(rng_key)
        next_state = da_model(
            prior_ensemble=posterior_state, obs_vect=obs_vect, rng_key=key
        )
        return (next_state, rng_key), next_state

    def da_rollout_fn(init_ensemble: jnp.ndarray) -> jnp.ndarray:
        """Data assimilation rollout function."""
        initial_state = (init_ensemble, initial_rng_key)
        _, posterior_trajectory = jax.lax.scan(scan_fn, initial_state, xs=observations)

        # posterior_trajectory has shape (num_steps - 1, ensemble_size, ...)
        # Transpose to (ensemble_size, num_steps - 1, ...)
        posterior_trajectory = jnp.transpose(
            posterior_trajectory, (1, 0) + tuple(range(2, posterior_trajectory.ndim))
        )

        if include_initial_state:
            # Add initial state at the beginning
            return jnp.concatenate(
                [init_ensemble[:, None, ...], posterior_trajectory], axis=1
            )

        return posterior_trajectory

    return da_rollout_fn


class BaseDataAssimilationMethod(abc.ABC):
    """Base class for data assimilation methods."""

    # Stateful filters mutate ``self`` inside ``_analysis_step`` (e.g. AGMF's
    # weight recursion) and therefore cannot be safely traced by jax.lax.scan
    # in ``rollout``. Such subclasses override this to True.
    is_stateful: bool = False

    def __init__(
        self,
        name: str,
        obs_operator: ObservationOperator,
        forward_operator: BaseForwardModel,
    ):
        """Initialize the data assimilation method."""
        self.name = name
        self.obs_operator = obs_operator
        self.forward_operator = forward_operator

    def _assimilate_data(
        self,
        prior_ensemble: np.ndarray,
        obs_vect: np.ndarray,
        rng_key: Optional[jax.random.PRNGKey] = None,
        return_model_integration_steps: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Assimilate the data."""

        forecast_ensemble = self._forecast_step(
            prior_ensemble,
            return_model_integration_steps=return_model_integration_steps,
        )
        analysis_ensemble = self._analysis_step(
            forecast_ensemble[:, -1], obs_vect, rng_key=rng_key, **kwargs
        )
        if return_model_integration_steps:
            analysis_ensemble = jnp.concatenate(
                [forecast_ensemble[:, :-1, :, :], analysis_ensemble[:, None, :, :]],
                axis=1,
            )

        return analysis_ensemble

    @abstractmethod
    def _analysis_step(
        self,
        prior_ensemble: np.ndarray,
        obs_vect: np.ndarray,
        rng_key: jax.random.PRNGKey,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Assimilate the data.

        Args:
            prior_ensemble (np.ndarray): Prior ensemble [Ensemble size, State dimension].
            obs_vect (np.ndarray): Observation vector.
            rng_key (Optional[jax.random.PRNGKey]): Optional RNG key for random operations.

        Returns:
            np.ndarray: Analysis ensemble [Ensemble size, State dimension].
        """
        raise NotImplementedError

    def _forecast_step(
        self, ensemble: np.ndarray, return_model_integration_steps: bool = False
    ) -> np.ndarray:
        """Forecast the ensemble."""
        return self.forward_operator(
            ensemble,
            return_model_integration_steps=return_model_integration_steps,
            is_ensemble=True,
        )

    def __call__(
        self,
        prior_ensemble: np.ndarray,
        obs_vect: np.ndarray,
        rng_key: Optional[jax.random.PRNGKey] = None,
        return_model_integration_steps: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Run the data assimilation method."""
        return self._assimilate_data(
            prior_ensemble,
            obs_vect,
            rng_key=rng_key,
            return_model_integration_steps=return_model_integration_steps,
            **kwargs,
        )

    def rollout(
        self,
        prior_ensemble: np.ndarray,
        observations: jnp.ndarray,
        rng_key: Optional[jax.random.PRNGKey] = None,
        return_model_integration_steps: bool = False,
    ) -> jnp.ndarray:
        """Rollout the data assimilation method over a sequence of observations.

        Note: this scans ``_assimilate_data`` with jax.lax.scan and therefore
        only supports stateless filters. Stateful filters (``is_stateful=True``,
        e.g. AGMF) mutate ``self`` during the analysis step and would either
        corrupt their internal recursion or leak a tracer into ``self`` under
        the scan, so they are rejected here.
        """

        if self.is_stateful:
            raise NotImplementedError(
                f"{type(self).__name__} is a stateful filter and cannot be run "
                "via rollout() (jax.lax.scan cannot safely trace the in-place "
                "state update). Call the filter eagerly, one step at a time."
            )

        if return_model_integration_steps:
            raise NotImplementedError(
                "rollout(return_model_integration_steps=True) is not implemented; "
                "da_rollout only returns the per-step analysis ensemble. Set "
                "return_model_integration_steps=False."
            )

        da_rollout_fn = da_rollout(
            self._assimilate_data,
            observations,
            rng_key,
            include_initial_state=True,
        )

        da_rollout_fn = jax.jit(da_rollout_fn)
        return da_rollout_fn(prior_ensemble)
