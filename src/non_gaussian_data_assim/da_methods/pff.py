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

DEFAULT_ALPHA = 0.01 / 10
DEFAULT_DS = 0.01 / 10
DEFAULT_B_D = 1.0


def grad_log_post(
    H: np.ndarray,
    R: np.ndarray,
    R_inv: np.ndarray,
    y: np.ndarray,
    y_i: np.ndarray,
    B: np.ndarray,
    x_s_i: np.ndarray,
    x0_mean: np.ndarray,
) -> np.ndarray:
    """
    Calculate the gradient of the log posterior distribution.

    Args:
    H (numpy.array): Observation operator matrix.
    R (numpy.array): Observation error covariance matrix.
    R_inv (numpy.array): Inverse of R.
    y (numpy.array): Observation vector.
    y_i (numpy.array): Individual observation vector for a particle.
    B (numpy.array): Covariance matrix of the ensemble.
    x_s_i (numpy.array): Current state of a particle.
    x0_mean (numpy.array): Mean state of the prior distribution.

    Returns:
    numpy.array: Gradient of the log posterior.
    """
    obs_part = B @ H.T @ R_inv @ (y - y_i)
    prior_part = x_s_i - x0_mean
    grad_log_post_est = obs_part - prior_part

    return grad_log_post_est


def exp_kernel(
    x: jnp.ndarray, y: jnp.ndarray, alpha: float, B_d: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Get the value and derivative of the kernel function.
    Args:
    diff (numpy.array): Difference between two state vectors.

    Returns:
    numpy.array: Value and derivative of the kernel function.
    """
    value = jnp.exp((-1 / 2) * ((x - y) ** 2) / (alpha * B_d))
    derivative = ((x - y) / (alpha * B_d)) * value
    return value, derivative


def compute_pairwise_interaction(
    x_s: jnp.ndarray, pair_function: Callable
) -> jnp.ndarray:
    """
    Compute the pairwise interaction of the ensemble.

    Args:
    x_s: State array of shape [ensemble, dof].
    pair_function: Pairwise function to compute the interaction.
    """
    x_transposed = x_s.T

    # Inner vmap (iterates j): returns [dof, ensemble] (j is dim 1)
    # Outer vmap (iterates i): stacks i at dim 1 -> [dof, ensemble, ensemble]
    vmap_func = jax.vmap(
        jax.vmap(pair_function, in_axes=(None, 0), out_axes=(1, 1)),
        in_axes=(0, None),
        out_axes=(1, 1),
    )

    vmap_func = jax.jit(vmap_func)

    return vmap_func(x_transposed, x_transposed)


class ParticleFlowFilter(BaseDataAssimilationMethod):
    def __init__(
        self,
        ensemble_size: int,
        R: np.ndarray,
        obs_operator: ObservationOperator,
        forward_operator: BaseForwardModel,
        localization_distance: Optional[int] = None,
        num_pseudo_time_steps: int = 100,
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

        x_s = prior_ensemble.reshape(self.ensemble_size, -1).T

        B = jnp.cov(x_s)
        B = self.localization(B)
        x0_mean = jnp.mean(x_s, axis=1)

        # Pseudo-time flow parameters
        ds = 0.01 / 10
        alpha = 0.01 / 10  # Tuning parameter for the covariance of the kernel

        # python_pseudoflow = jnp.zeros((self.dofs, self.ensemble_size, self.num_pseudo_time_steps + 1))
        # python_pseudoflow = python_pseudoflow.at[:, :, 0].set(x_s)

        R_inv = jnp.linalg.inv(self.R)

        # Pseudo-time for data assimilation
        for _ in range(self.num_pseudo_time_steps):

            # Gradient of the log posterior
            grad_log_post_fn = lambda x: grad_log_post(
                H=self.obs_operator.obs_matrix,
                R=self.R,
                R_inv=R_inv,
                y=obs_vect,
                y_i=self.obs_operator.obs_matrix @ x,
                B=B,
                x_s_i=x,
                x0_mean=x0_mean,
            )
            grad_log_post_fn = jax.jit(jax.vmap(grad_log_post_fn))
            dpdx = grad_log_post_fn(x_s.T).T

            # B_d = jnp.diag(B)
            # kk = np.zeros((self.dofs, self.ensemble_size, self.ensemble_size))
            # dkdx = np.zeros((self.dofs, self.ensemble_size, self.ensemble_size))
            # I_f = np.zeros((self.dofs, self.ensemble_size))

            # for i in range(self.ensemble_size):
            #     for j in range(i, self.ensemble_size):
            #         kk[:, i, j] = np.exp((-1 / 2) * ((x_s[:, i] - x_s[:, j]) ** 2) / (alpha * B_d[:]))
            #         dkdx[:, i, j] = ((x_s[:, i] - x_s[:, j]) / alpha) * kk[:, i, j]
            #         if j != i:
            #             kk[:, j, i] = kk[:, i, j]
            #             dkdx[:, j, i] = -dkdx[:, i, j]

            #     attractive_term = (1 / self.ensemble_size) * (kk[:, i, :] * dpdx)
            #     repelling_term = (1 / self.ensemble_size) * dkdx[:, i, :]
            #     I_f[:, i] = np.sum(attractive_term + repelling_term, axis=1)

            # Kernel calculation
            @jax.jit  # type: ignore[misc]
            def kernel_fn(
                x: jnp.ndarray, y: jnp.ndarray
            ) -> Tuple[jnp.ndarray, jnp.ndarray]:
                """Kernel function."""
                return exp_kernel(x, y, alpha, jnp.diag(B))

            kernel, dkernel_dx = compute_pairwise_interaction(x_s, kernel_fn)

            I_f = jnp.zeros((self.dofs, self.ensemble_size))
            for i in range(self.ensemble_size):
                attractive_term = (1 / self.ensemble_size) * (kernel[:, i, :] * dpdx)
                repelling_term = (1 / self.ensemble_size) * dkernel_dx[:, i, :]
                I_f = I_f.at[:, i].set(np.sum(attractive_term + repelling_term, axis=1))

            # Update the state vector for next pseudo time step
            x_s = x_s + (ds * I_f)
            # python_pseudoflow[:, :, s + 1] = x_s

            # s += 1

        # Gathering final results
        # posterior_ensemble = python_pseudoflow[:, :, -1]
        # mean_posterior = np.mean(posterior_vect, axis=1)
        # cov_posterior = np.cov(posterior_vect)

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
