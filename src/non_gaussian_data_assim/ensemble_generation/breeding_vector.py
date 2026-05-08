from typing import Optional, Protocol

import jax
import jax.numpy as jnp
from tqdm import tqdm

from non_gaussian_data_assim.forward_models.base import BaseForwardModel


class NormLike(Protocol):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray: ...


class L2Norm:
    """L2 norm over the full perturbation field."""

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(x.ravel(), ord=2)


class L1Norm:
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(x, ord=1)


class ChebyshevNorm:
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(x, ord=jnp.inf)


class BreedingVector:
    """Generate bred perturbation vectors around one base state.

    Important:
        'outer_steps_per_cycle' is interpreted as the number of raw model
        integration steps, not as the number of 'data_assimilation_steps'.
            Thus I use 'forward_model.one_step', instead of 'forward_model(...)'
            (because that advances the state by 'model_integration_steps'
    """

    def __init__(
        self,
        forward_model: BaseForwardModel,
        breeding_cycles: int,
        outer_steps_per_cycle: int,
        delta0: float,
        norm_fct: Optional[NormLike] = None,
        min_norm: float = 1e-10,
    ) -> None:
        """
        Initialise the breeding-vector

        Inputs:
        -------
        forward_model: BaseForwardModel
            Model that is used to propagate the states forward

        breeding_cycles: int
            Number of breeding-cycles

        outer_steps_per_cycle: int
            Number of integration steps per interval (breeding-cycle).
        --> Total Model Steps = breeding_cycles * outer_steps_per_cycle

        delta0: float
            Determines scale of initally random noise as well as scale at every rescaling steps

        norm: NormLike
            Choose Norm of perturbation vector (default: L2-norm)
        """
        self.name = "breeding_vector"
        self.forward_model = forward_model
        self.breeding_cycles = breeding_cycles
        self.outer_steps_per_cycle = outer_steps_per_cycle
        self.delta0 = delta0
        self.norm_fct = norm_fct if norm_fct is not None else L2Norm()
        self.min_norm = min_norm

        self.integration_steps = (
            self.outer_steps_per_cycle * self.forward_model.model_integration_steps
        )

    def _singelton_ensemble_axis(self, x0_bg: jnp.ndarray) -> jnp.ndarray:
        """Return base state that makes sure ensemble_member axis is leading!!"""
        if x0_bg.ndim == 2:
            return x0_bg[None, ...]  # [1, num_states, state_dim]

        if x0_bg.ndim == 3 and x0_bg.shape[0] == 1:
            return x0_bg

        err_msg = f"x0_bg should of shape [num_states, state_dim] or [1, num_states, state_dim]. Got {x0_bg.shape}."
        raise ValueError(err_msg)

    def rescale_bv_perturbation(self, perturbation: jnp.ndarray) -> jnp.ndarray:
        """Rescale one perturbation field to specified norm delta0."""
        pert_norm = self.norm_fct(perturbation)
        pert_norm = jnp.maximum(pert_norm, self.min_norm)
        return self.delta0 * perturbation / pert_norm

    def rescale_bv_ensemble(self, perturbations: jnp.ndarray) -> jnp.ndarray:
        """Rescale every ensemble member independently"""
        return jax.vmap(self.rescale_bv_perturbation, in_axes=0, out_axes=0)(
            perturbations
        )  # assumes shape: (ens_size, ....)

    def initial_perturbation_ensemble(
        self,
        rng_key: jax.Array,
        ensemble_size: int,
        state_shape: tuple[int, ...],
    ) -> jnp.ndarray:
        """Create member-wise independent initial perturbations."""
        member_keys = jax.random.split(rng_key, ensemble_size)

        def sample_one_initial_perturbation(key: jax.Array) -> jnp.ndarray:
            return jax.random.normal(key, state_shape)

        raw = jax.vmap(sample_one_initial_perturbation, in_axes=0, out_axes=0)(
            member_keys
        )

        return self.rescale_bv_ensemble(raw)

    def forward_control_and_ensemble(
        self,
        x_ctrl0: jnp.ndarray,
        bvp_t0: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Run 1 control and ALL perturbed states
        --> Fct changes shape of arrays !!!
        Input:
            x_ctrl0: (1, num_states, state_dim)
            bvp_t0:  (ensemble_size, num_states, state_dim)

        Returns:
            x_ctrl_traj: [1, time, num_states, state_dim]
            x_pert_traj: [ensemble_size, time, num_states, state_dim]
        """
        x_pert0 = x_ctrl0 + bvp_t0

        # One batched model integration. First member is control, remaining are ens-nembers!
        x_all0 = jnp.concatenate([x_ctrl0, x_pert0], axis=0)

        x_all_traj = self.forward_model.rollout(
            x_all0, self.outer_steps_per_cycle, return_model_integration_steps=True
        )

        x_ctrl_traj = x_all_traj[:1]
        x_pert_traj = x_all_traj[1:]

        return x_ctrl_traj, x_pert_traj

    def _growth_rate(self, perturb_norm: jnp.ndarray) -> jnp.ndarray:
        """
        Calcualte growth rate:
             ∂ = ∂_0 * exp[Lambda * time]   ;  time=dt*steps
        """
        dt_bv = self.forward_model.dt
        cycle_time = self.integration_steps * dt_bv
        return (1 / cycle_time) * jnp.log(perturb_norm / self.delta0)

    def compute_metrics(self, bvp_traj: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute Metrics based on the BreedingVector evolution over the breeding-cycle"""

        # -- Calculate Norm of perturbations for all steps
        calc_bv_norm = jax.vmap(self.norm_fct, in_axes=1)
        bvp_norm_traj = calc_bv_norm(bvp_traj)  # shape (number_steps,)

        # -- Calculate Growth rate of perturbations over breeding-cycle
        growth_rate_cycle = self._growth_rate(bvp_norm_traj[-1])

        return bvp_norm_traj, growth_rate_cycle

    def compute_ensemble_metrics(
        self, bvp_traj: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute norms and growth rates for all ensemble members.

        Input:
            bvp_traj: [ensemble_size, time, num_states, state_dim]

        Returns:
            bvp_norm_traj:     [ensemble_size, time]
            growth_rate_cycle: [ensemble_size]
        """

        # --- 1) Create map for one member trajectory: [time, num_states, state_dim] -> [time]
        norm_one_member_traj = jax.vmap(self.norm_fct, in_axes=0, out_axes=0)
        # --- For all members: [ensemble_size, time, num_states, state_dim] ->  [ensemble_size, time]
        norm_all_members_traj = jax.vmap(norm_one_member_traj, in_axes=0, out_axes=0)(
            bvp_traj
        )

        # --- 2) Compute growth-rate
        growth_rate_cycle = self._growth_rate(norm_all_members_traj[:, -1])

        return norm_all_members_traj, growth_rate_cycle

    def sample_ensemble(
        self,
        x0_bg: jnp.ndarray,
        rng_key: jax.Array,
        ensemble_size: int,
        return_metrics: bool = True,
    ) -> jnp.ndarray | tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """
        Create bred perturbations for full ensemble.

        Returns:
            bv_pert: [ensemble_size, num_states, state_dim] + Fields + Metric-dictionary
        """

        x_ctrl0 = self._singelton_ensemble_axis(x0_bg)
        state_shape = tuple(x_ctrl0.shape[1:])

        bvp_t0 = self.initial_perturbation_ensemble(
            rng_key=rng_key,
            ensemble_size=ensemble_size,
            state_shape=state_shape,
        )

        # Store per-cycle arrays
        bvp_norm_list = []
        growth_rate_list = []
        bv0_states_list = []
        bv1_states_list = []

        for _ in tqdm(range(self.breeding_cycles), desc="Run Breeding"):

            # --- 1) Integrate Control- and Perturbed- State to over breeding cycle
            x_ctrl_traj, x_pert_traj = self.forward_control_and_ensemble(
                x_ctrl0=x_ctrl0, bvp_t0=bvp_t0
            )

            # --- 2) Compute BV-perturbations over entire cycle
            bvp_traj = x_pert_traj - x_ctrl_traj
            ## BV-perturbation at last step of this breeding interval
            bvp_t1 = bvp_traj[:, -1, ...]

            # --- 3) Optional: Store fields and metrics
            if return_metrics:
                bvp_norm, gr_cycle = self.compute_ensemble_metrics(bvp_traj)
                bvp_norm_list.append(bvp_norm)
                growth_rate_list.append(gr_cycle)
                bv0_states_list.append(bvp_t0)
                bv1_states_list.append(bvp_t1)

            # --- 4) Rescale perturbations and set them as new starting perturbations Also for control
            bvp_t0 = self.rescale_bv_ensemble(bvp_t1)
            x_ctrl0 = x_ctrl_traj[:, -1, ...]

        ### --- Last rescaled BV-perturabtion -> This will be used for ensemble-generation
        bv_pert = bvp_t0

        if not return_metrics:
            return bv_pert

        fields_metrics = {
            "bv0_states": jnp.stack(
                bv0_states_list, axis=0
            ),  # [breeding_cycles, ensemble_size, num_states, state_dim]
            "bv1_states": jnp.stack(
                bv1_states_list, axis=0
            ),  # [breeding_cycles, ensemble_size, num_states, state_dim]
            "bvp_norm": jnp.concatenate(bvp_norm_list, axis=1),  # [ensemble_size, time]
            "growth_rate": jnp.stack(
                growth_rate_list, axis=0
            ),  # [breeding_cycles, ensemble_size]
        }

        return bv_pert, fields_metrics
