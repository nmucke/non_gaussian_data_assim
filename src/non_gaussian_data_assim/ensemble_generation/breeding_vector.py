from dataclasses import dataclass
from typing import Optional, Protocol

import jax
import jax.numpy as jnp
import tqdm

from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.time_integrators import rollout


@dataclass
class BreedingDiagnostics:
    norm_during_window: jnp.ndarray
    growth_rate_during_window: jnp.ndarray
    norm_before_rescale: jnp.ndarray
    growth_rate_before_rescale: jnp.ndarray
    norm_after_rescale: jnp.ndarray
    growth_rate_after_rescale: jnp.ndarray
    perturbations: jnp.ndarray


class NormLike(Protocol):
    """Interface for norm implementations"""

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray: ...


class L2Norm:
    """L2 norm"""

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(x, ord=2)


class L1Norm:
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(x, ord=1)


class ChebyshevNorm:
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(x, ord=jnp.inf)


class BreedingVector:
    """Generate bred perturbation vectors around one base state.

    Important:
        'rescaling_interval' is interpreted as the number of raw model
        integration steps, not as the number of 'data_assimilation_steps'.
            Thus I use 'forward_operator.one_step', instead of 'forward_operator(...)'
            (because that advances the state by 'model_integration_steps'
    """

    def __init__(
        self,
        forward_operator: BaseForwardModel,
        number_of_intervals: int,
        rescaling_interval: int,
        perturbation_amplitude: float,
        norm: Optional[NormLike] = None,
        compute_metrics: Optional[bool] = True,
        min_norm: float = 1e-10,
    ) -> None:
        """
        Initialise the breeding-vector

        Inputs:
        -------
        forward_operator: BaseForwardModel
            Model that is used to propagate the states forward

        number_of_intervals: int
            Number of breeding-cycles

        rescaling_interval: int
            Number of integration steps per interval (breeding-cycle).
        --> Total Model Steps = number_of_intervals * rescaling_interval

        perturbation_amplitude: float
            Determines scale of initally random noise as well as scale at every rescaling steps

        norm: NormLike
            Choose Norm of perturbation vector (default: L2-norm)
        """
        self.name = "breeding_vector"
        self.forward_operator = forward_operator
        self.number_of_intervals = number_of_intervals
        self.rescaling_interval = rescaling_interval
        self.perturbation_amplitude = perturbation_amplitude
        self.norm = norm if norm is not None else L2Norm()
        self.min_norm = min_norm

        self.compute_metrics: bool = True
        # --- Define method-maps that compute the norm per member and then for the entire ensemble
        self._norm_ensemble = jax.vmap(self.norm, in_axes=0)
        self._norm_member_trajectory = jax.vmap(self.norm, in_axes=0)
        self._norm_ensemble_trajectory = jax.vmap(
            self._norm_member_trajectory, in_axes=0
        )

        self._integrate_member = rollout(
            self.forward_operator.one_step,
            self.rescaling_interval,
            return_model_integration_steps=self.compute_metrics,
            include_initial_state=False,
        )
        self._integrate_ensemble = jax.vmap(self._integrate_member, in_axes=0)

    def _rescale_member(self, perturbation: jnp.ndarray) -> jnp.ndarray:
        """Rescale one perturbation field to the configured amplitude"""
        # --- 1) Calc. norm
        norm = self.norm(perturbation)
        # --- 2) Return rescaled perturbation
        return self.perturbation_amplitude * perturbation / norm

    def _rescale_ensemble(self, perturbations: jnp.ndarray) -> jnp.ndarray:
        """Rescale perturbations member by member.
        Input:
            perturbations: shape [ensemble_size, num_states, state_dim]
        """
        return jax.vmap(self._rescale_member, in_axes=0)(perturbations)

    def _growth_rate(self, norm_value: jnp.ndarray, elapsed_steps: int) -> jnp.ndarray:
        elapsed_time = elapsed_steps * self.forward_operator.dt
        return jnp.log(norm_value / self.perturbation_amplitude) / elapsed_time

    def initial_perturbations(
        self,
        ensemble_size: int,
        state_shape: tuple[int, ...],
        rng_key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """Create normalized initial perturbation directions."""
        raw = jax.random.normal(rng_key, (ensemble_size, *state_shape))
        return self._rescale_ensemble(raw)

    def __call__(
        self,
        x0_bg: jnp.ndarray,
        ensemble_size: int,
        rng_key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Breed perturbations around one base state.

        Inputs:
        -------
            x0_bg: jnp.ndarray     Base state with shape [num_states, state_dim] or [1, num_states, state_dim]
            ensemble_size: int  Number of bred perturbation vectors to generate.
        Returns:
        --------
            Breeded perturbations (not full state!) with shape [ensemble_size, num_states, state_dim].
        """

        # --- 1) Define control and initial-pertrubation for all ensemble members
        control = x0_bg
        perturbations = self.initial_perturbations(
            rng_key=rng_key, ensemble_size=ensemble_size, state_shape=x0_bg.shape
        )

        if self.compute_metrics:
            norm_during_window = []
            growth_rate_during_window = []
            norm_before_rescale = []
            growth_rate_before_rescale = []
            norm_after_rescale = []
            growth_rate_after_rescale = []
            perturbation_history = []

        # --- 2) Run Breeding Loop
        for _ in tqdm(range(self.number_of_intervals), desc="b"):

            # -- 2.1) Concatenate Control + Ensemble-Members to a State-Array of shape: [1 + ensemble_size, num_states, state_dim]
            states = jnp.concatenate(
                [control[None, ...], control[None, ...] + perturbations], axis=0
            )
            # -- 2.2) Integrate State-Array over breeding interval (n-model)
            advanced = self._integrate_ensemble(states)

            # --- 2.3) Get perturbations and calculate the gorwth rate if desired?
            if self.compute_metrics:
                control_traj = advanced[0]
                perturbed_traj = advanced[1:]

                grown_traj = perturbed_traj - control_traj[None, ...]
                grown_perturbations = grown_traj[
                    :, -1, ...
                ]  # [ensemble_size, num_states, state_dim]

                norm_traj = self._norm_ensemble_trajectory(grown_traj)

                # --- Calcualte growth rate: \delta = \delta_0 * exp[\Lambda * time]   ;  time=dt*steps
                delta0 = self.perturbation_amplitude
                dt_bv = self.forward_operator.dt
                elapsed_steps = jnp.arange(1, self.rescaling_interval + 1)
                growth_rate_traj = (
                    1 / (elapsed_steps[None, :] * dt_bv) * jnp.log(norm_traj / delta0)
                )

                control = control_traj[-1]

                # --- Rescale perturbations
                perturbations = self._rescale_ensemble(grown_perturbations)

                norm_after = self._norm_ensemble(perturbations)
                growth_rate_after = (
                    1 / (self.rescaling_interval * dt_bv) * jnp.log(norm_after / delta0)
                )

                # --- Store metrics
                norm_during_window.append(norm_traj.T)
                growth_rate_during_window.append(growth_rate_traj.T)
                norm_before_rescale.append(norm_traj[:, -1])
                growth_rate_before_rescale.append(growth_rate_traj[:, -1])
                norm_after_rescale.append(norm_after)
                growth_rate_after_rescale.append(growth_rate_after)
                perturbation_history.append(perturbations)

            else:
                control_next = advanced[0]
                perturbed_next = advanced[1:]

                grown_perturbations = perturbed_next - control_next[None, ...]
                control = control_next

                # --- Rescale perturbations
                perturbations = self._rescale_ensemble(grown_perturbations)

        if self.compute_metrics:
            diagnostics = BreedingDiagnostics(
                norm_during_window=jnp.stack(norm_during_window, axis=0),
                growth_rate_during_window=jnp.stack(growth_rate_during_window, axis=0),
                norm_before_rescale=jnp.stack(norm_before_rescale, axis=0),
                growth_rate_before_rescale=jnp.stack(
                    growth_rate_before_rescale, axis=0
                ),
                norm_after_rescale=jnp.stack(norm_after_rescale, axis=0),
                growth_rate_after_rescale=jnp.stack(growth_rate_after_rescale, axis=0),
                perturbations=jnp.stack(perturbation_history, axis=0),
            )

            return perturbations, diagnostics
        else:
            return perturbations
