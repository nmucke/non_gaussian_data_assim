import pdb
import time
from curses import KEY_BREAK
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from non_gaussian_data_assim.da_methods.base import BaseDataAssimilationMethod
from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.localization import distance_based_localization
from non_gaussian_data_assim.observation_operator import ObservationOperator
from non_gaussian_data_assim.time_integrators import RungeKutta4, rollout

DEFAULT_ALPHA = 0.01 / 10
DEFAULT_STEP_SIZE = 0.01 / 10
DEFAULT_B_D = 1.0


def get_prior_score_fn(
    prior_mean: np.ndarray,
    prior_cov_inv: np.ndarray,
) -> np.ndarray:
    """
    Get the prior score function.
    """

    def prior_score_fn(x_s: np.ndarray) -> np.ndarray:
        return prior_cov_inv @ (x_s - prior_mean)

    return prior_score_fn


def get_likelihood_score_fn(
    obs_vect: np.ndarray,
    obs_matrix: np.ndarray,
    obs_cov_inv: np.ndarray,
) -> np.ndarray:
    """
    Get the likelihood score function.
    """

    def likelihood_score_fn(x_s: np.ndarray) -> np.ndarray:
        return obs_matrix.T @ obs_cov_inv @ (obs_vect - obs_matrix @ x_s)

    return likelihood_score_fn


def get_posterior_score_fn(
    prior_score_fn: Callable[[np.ndarray], np.ndarray],
    likelihood_score_fn: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Get the posterior score function.
    """

    def posterior_score_fn(x_s: np.ndarray) -> np.ndarray:
        return prior_score_fn(x_s) + likelihood_score_fn(x_s)

    return posterior_score_fn


def get_scalar_kernel_fn(
    distance_weight_matrix: jnp.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Calculate the scalar kernel.
    """

    def scalar_kernel_fn(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-0.5 * (x - y).T @ distance_weight_matrix @ (x - y))

    return scalar_kernel_fn


def get_scalar_kernel_matrix_fn(
    scalar_kernel_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Calculate the scalar kernel matrix.
    """

    def scalar_kernel_matrix_fn(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return scalar_kernel_fn(x, y) * jnp.eye(x.shape[0])

    return scalar_kernel_matrix_fn


def get_divergence_scalar_kernel_matrix_fn(
    scalar_kernel_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    distance_weight_matrix: jnp.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Calculate the divergence of the scalar kernel matrix.
    """

    def divergence_scalar_kernel_matrix_fn(
        x: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        return -distance_weight_matrix.T @ (x - y) * scalar_kernel_fn(x, y)

    return divergence_scalar_kernel_matrix_fn


def compute_pairwise_interaction(
    x_s: jnp.ndarray, pair_function: Callable
) -> jnp.ndarray:
    """
    Compute the pairwise interaction of the ensemble.

    Args:
    x_s: State array of shape [dofs, ensemble].
    pair_function: Function f(x, y) taking two [dofs] vectors, returns [dofs].

    Returns:
    Array of shape [dofs, ensemble, ensemble] where
    result[:, i, j] = pair_function(x_s[:, i], x_s[:, j]).
    """
    # Inner vmap over j: for fixed i, compute pair_function(x_s[:, i], x_s[:, j]) for all j
    # in_axes=(None, 1): batch over columns (axis 1) of second arg
    # out_axes=1: stack j along axis 1 -> [dofs, ensemble]
    inner_vmap = jax.vmap(pair_function, in_axes=(None, 1), out_axes=1)

    # Outer vmap over i: for each i, run inner_vmap(x_s[:, i], x_s)
    # in_axes=(1, None): batch over columns (axis 1) of first arg
    # out_axes=1: stack i along axis 1 -> [dofs, ensemble, ensemble]
    vmap_func = jax.vmap(inner_vmap, in_axes=(1, None), out_axes=1)

    return vmap_func(x_s, x_s)


def get_pairwise_interactions_kernel_fn(
    distance_weight_matrix: jnp.ndarray,
) -> Tuple[Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
    """
    Get the pairwise interactions kernel function.
    """

    scalar_kernel_fn = get_scalar_kernel_fn(distance_weight_matrix)

    # return divergence_scalar_kernel_matrix_fn, scalar_kernel_matrix_fn
    def divergence_kernel_fn(x_s: jnp.ndarray) -> jnp.ndarray:
        return compute_pairwise_interaction(
            x_s,
            get_divergence_scalar_kernel_matrix_fn(
                scalar_kernel_fn=scalar_kernel_fn,
                distance_weight_matrix=distance_weight_matrix,
            ),
        )

    def scalar_kernel_matrix_fn(x_s: jnp.ndarray) -> jnp.ndarray:
        return compute_pairwise_interaction(
            x_s, get_scalar_kernel_matrix_fn(scalar_kernel_fn=scalar_kernel_fn)
        )

    return divergence_kernel_fn, scalar_kernel_matrix_fn


def get_rhs_fn(
    posterior_score_fn: Callable[[jnp.ndarray], jnp.ndarray],
    divergence_scalar_kernel_matrix_fn: Callable[[jnp.ndarray], jnp.ndarray],
    scalar_kernel_matrix_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Get the right-hand side function.
    """

    def rhs_fn(x_s: jnp.ndarray) -> jnp.ndarray:
        posterior_score = posterior_score_fn(x_s)

        divergence_kernel = divergence_scalar_kernel_matrix_fn(x_s)

        kernel_mat = scalar_kernel_matrix_fn(x_s)

        divergence_kernel_term = divergence_kernel.sum(axis=1)

        post_term = jnp.einsum("dije,ei->dj", kernel_mat, posterior_score)

        I_f = divergence_kernel_term + post_term

        return I_f / x_s.shape[-1]

    return rhs_fn


class ParticleFlowFilter(BaseDataAssimilationMethod):
    def __init__(
        self,
        ensemble_size: int,
        R: np.ndarray,
        obs_operator: ObservationOperator,
        forward_operator: BaseForwardModel,
        localization_distance: Optional[int] = None,
        num_pseudo_time_steps: int = 100,
        step_size: float = DEFAULT_STEP_SIZE,
        alpha: float = DEFAULT_ALPHA,
        kernel_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda x, y: np.exp(
            -((x - y) ** 2) / (2 * DEFAULT_ALPHA * DEFAULT_B_D)
        ),
    ) -> None:
        """
        Initialize the Particle Flow Filter.
        Args:
        mem (int): Number of ensemble members.
        nx (int): Size of the state vector.
        R (numpy.array): Observation error covariance matrix.
        obs_operator (Callable[[np.ndarray], np.ndarray]): Observation operator.
        """
        super().__init__(obs_operator, forward_operator)
        self.ensemble_size = ensemble_size
        self.R = R
        self.num_states = forward_operator.num_states
        self.state_dim = forward_operator.state_dim
        self.dofs = self.num_states * self.state_dim
        self.num_pseudo_time_steps = num_pseudo_time_steps
        self.localization_distance = localization_distance
        self.step_size = step_size
        self.alpha = alpha

        if self.localization_distance is None:
            self.localization = lambda x: x
        else:
            self.localization = lambda x: distance_based_localization(
                self.localization_distance, self.state_dim, x  # type: ignore[arg-type]
            )

    def _analysis_step(
        self,
        prior_ensemble: np.ndarray,
        obs_vect: np.ndarray,
        rng_key: jax.random.PRNGKey,
    ) -> np.ndarray:

        x_s = prior_ensemble.reshape(self.ensemble_size, -1).T  # [dofs, ensemble]

        prior_cov = jnp.cov(x_s)
        prior_cov = self.localization(prior_cov)
        prior_mean = jnp.mean(x_s, axis=1)
        prior_cov_inv = jnp.linalg.inv(prior_cov)

        distance_weight_matrix = jnp.linalg.inv(self.alpha * prior_cov)

        obs_cov_inv = jnp.linalg.inv(self.R)

        prior_score_fn = get_prior_score_fn(
            prior_mean=prior_mean,
            prior_cov_inv=prior_cov_inv,
        )
        likelihood_score_fn = get_likelihood_score_fn(
            obs_vect=obs_vect,
            obs_matrix=self.obs_operator.obs_matrix,
            obs_cov_inv=obs_cov_inv,
        )
        posterior_score_fn = get_posterior_score_fn(
            prior_score_fn=prior_score_fn,
            likelihood_score_fn=likelihood_score_fn,
        )

        posterior_score_vmap = jax.vmap(posterior_score_fn, in_axes=-1, out_axes=-1)

        divergence_kernel_fn, scalar_kernel_matrix_fn = (
            get_pairwise_interactions_kernel_fn(
                distance_weight_matrix=distance_weight_matrix,
            )
        )

        rhs_fn = get_rhs_fn(
            posterior_score_fn=jax.jit(posterior_score_vmap),
            divergence_scalar_kernel_matrix_fn=jax.jit(divergence_kernel_fn),
            scalar_kernel_matrix_fn=jax.jit(scalar_kernel_matrix_fn),
        )
        rk_stepper = RungeKutta4(self.step_size, rhs_fn)
        rollout_fn = rollout(rk_stepper, self.num_pseudo_time_steps)
        rollout_fn = jax.jit(rollout_fn)

        x_s = rollout_fn(x_s)

        # Pseudo-time for data assimilation
        # for _ in range(self.num_pseudo_time_steps):
        #     x_s = rk_stepper(x_s)

        x_s = x_s.T

        return x_s.reshape(self.ensemble_size, self.num_states, self.state_dim)


# def pff(
#     mem: int,
#     n_states: int,
#     ensemble: np.ndarray,
#     obs_vect: np.ndarray,
#     R: np.ndarray,
# ) -> Dict[str, Any]:
#     """
#     Implement the Particle Flow Filter.

#     Args:
#     mem (int): Number of ensemble members.
#     n_states (int): Number of states.
#     ensemble (numpy.array): Initial ensemble of states.
#     obs_vect (numpy.array): Observation vector.
#     R (numpy.array): Observation error covariance matrix.

#     Returns:
#     dict: Dictionary containing the posterior ensemble, mean, and covariance.
#     """
#     index_obs = np.where(obs_vect > -999)[0]
#     B = np.cov(ensemble)
#     x0_mean = np.mean(ensemble, axis=1)

#     # Pseudo-time flow parameters
#     s = 0
#     max_s = 100
#     ds = 0.05 / 10
#     alpha = 0.05 / 10  # Tuning parameter for the covariance of the kernel

#     x_s = ensemble.copy()
#     python_pseudoflow = np.zeros((n_states, mem, max_s + 1))
#     python_pseudoflow[:, :, 0] = x_s.copy()

#     n_obs = np.sum(obs_vect > -999)
#     R_inv = np.linalg.inv(R)

#     # Pseudo-time for data assimilation
#     while s < max_s:

#         H = np.zeros((n_obs, n_states))
#         Hx = np.zeros((n_obs, mem))
#         dHdx = np.zeros((n_obs, n_states, mem))
#         dpdx = np.zeros((n_states, mem))

#         for i in range(mem):
#             H = h_operator(n_states, obs_vect)
#             Hx[:, :] = x_s[index_obs, :]
#             y = np.ones((n_obs, 1))
#             y[:, 0] = obs_vect[index_obs]
#             y_i = np.ones((n_obs, 1))
#             y_i[:, 0] = Hx[:, i]

#             dpdx[:, i] = grad_log_post(H, R, R_inv, y, y_i, B, x_s[:, i], x0_mean)

#         # Kernel calculation
#         B_d = np.zeros((n_states))
#         for d in range(n_states):
#             B_d[d] = B[d, d]

#         kernel = np.zeros((n_states, mem, mem))
#         dkdx = np.zeros((n_states, mem, mem))
#         I_f = np.zeros((n_states, mem))

#         for i in range(mem):
#             for j in range(i, mem):
#                 kernel[:, i, j] = np.exp(
#                     (-1 / 2) * ((x_s[:, i] - x_s[:, j]) ** 2) / (alpha * B_d[:])
#                 )
#                 dkdx[:, i, j] = ((x_s[:, i] - x_s[:, j]) / alpha) * kernel[:, i, j]
#                 if j != i:
#                     kernel[:, i, j] = kernel[:, j, i]
#                     dkdx[:, i, j] = -dkdx[:, j, i]

#             attractive_term = (1 / mem) * (kernel[:, i, :] * dpdx)
#             repelling_term = (1 / mem) * dkdx[:, i, :]
#             I_f[:, i] = np.sum(attractive_term + repelling_term, axis=1)

#         # Update the state vector for next pseudo time step
#         fs = I_f
#         x_s += ds * fs
#         python_pseudoflow[:, :, s + 1] = x_s

#         s += 1

#     # Gathering final results
#     posterior_vect = python_pseudoflow[:, :, -1]
#     mean_posterior = np.mean(posterior_vect, axis=1)
#     cov_posterior = np.cov(posterior_vect)

#     pff_pseudoflow = {
#         "posterior": posterior_vect,
#         "mean_post": mean_posterior,
#         "cov_post": cov_posterior,
#     }

#     return pff_pseudoflow
