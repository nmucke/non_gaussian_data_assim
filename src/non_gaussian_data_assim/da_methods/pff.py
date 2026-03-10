from typing import Any, Callable, Dict, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from non_gaussian_data_assim.da_methods.base import BaseDataAssimilationMethod
from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.jax_utils import get_pairwise_interaction_fn
from non_gaussian_data_assim.kernels import (
    get_divergence_kernel_fn,
    get_kernel_matrix_fn,
)
from non_gaussian_data_assim.localization import distance_based_localization
from non_gaussian_data_assim.observation_operator import (
    LinearObservationOperator,
    NonlinearObservationOperator,
    ObservationOperator,
)
from non_gaussian_data_assim.time_integrators import get_stepper, rollout

DEFAULT_ALPHA = 0.01 / 10
DEFAULT_STEP_SIZE = 0.01 / 10
DEFAULT_B_D = 1.0
DEFAULT_HELLINGER_COV_REGULARIZATION = 1e-6


# =============================================================================
# Score functions
# =============================================================================


def get_prior_score_fn(
    prior_mean: np.ndarray,
    prior_cov_inv: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Get the prior score function: nabla log p_0(x) = -B^{-1}(x - m_0)."""

    def prior_score_fn(x_s: np.ndarray) -> np.ndarray:
        return -prior_cov_inv @ (x_s - prior_mean)

    return prior_score_fn


def get_likelihood_score_fn_with_linear_obs_operator(
    obs_vect: np.ndarray,
    obs_matrix: np.ndarray,
    obs_cov_inv: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Get the likelihood score function for a linear observation operator."""

    def likelihood_score_fn(x_s: np.ndarray) -> np.ndarray:
        return -obs_matrix.T @ obs_cov_inv @ (obs_matrix @ x_s - obs_vect)

    return likelihood_score_fn


def get_likelihood_score_fn_with_non_linear_obs_operator(
    obs_vect: np.ndarray,
    obs_operator: NonlinearObservationOperator,
    obs_cov_inv: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Get the likelihood score function for a nonlinear observation operator."""

    def likelihood_score_fn(x_s: np.ndarray) -> np.ndarray:
        obs_gradient = obs_operator.grad_obs_operator(x_s)
        return (
            -obs_gradient @ obs_cov_inv @ (obs_operator._obs_operator(x_s) - obs_vect)
        )

    return likelihood_score_fn


def get_posterior_score_fn(
    prior_score_fn: Callable[[np.ndarray], np.ndarray],
    likelihood_score_fn: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """Get the posterior score function: nabla log pi(x) = nabla log p_0(x) + nabla log ell(y|x)."""

    def posterior_score_fn(x_s: np.ndarray) -> np.ndarray:
        return prior_score_fn(x_s) + likelihood_score_fn(x_s)

    return posterior_score_fn


# =============================================================================
# Hellinger weight estimation
# =============================================================================


def get_log_unnorm_posterior_fn_linear(
    prior_mean: jnp.ndarray,
    prior_cov_inv: jnp.ndarray,
    obs_vect: jnp.ndarray,
    obs_matrix: jnp.ndarray,
    obs_cov_inv: jnp.ndarray,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get log pi(x) up to a normalising constant (linear observation operator).

    Returns a function mapping a single state vector x [dofs] to the scalar
    log p_0(x) + log ell(y|x), omitting additive constants.
    """

    def log_unnorm_posterior(x: jnp.ndarray) -> jnp.ndarray:
        dx = x - prior_mean
        log_prior = -0.5 * dx @ prior_cov_inv @ dx
        innovation = obs_matrix @ x - obs_vect
        log_lik = -0.5 * innovation @ obs_cov_inv @ innovation
        return log_prior + log_lik

    return log_unnorm_posterior


def get_log_unnorm_posterior_fn_nonlinear(
    prior_mean: jnp.ndarray,
    prior_cov_inv: jnp.ndarray,
    obs_vect: jnp.ndarray,
    obs_operator: NonlinearObservationOperator,
    obs_cov_inv: jnp.ndarray,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get log pi(x) up to a normalising constant (nonlinear observation operator)."""

    def log_unnorm_posterior(x: jnp.ndarray) -> jnp.ndarray:
        dx = x - prior_mean
        log_prior = -0.5 * dx @ prior_cov_inv @ dx
        innovation = obs_operator._obs_operator(x) - obs_vect
        log_lik = -0.5 * innovation @ obs_cov_inv @ innovation
        return log_prior + log_lik

    return log_unnorm_posterior


def get_hellinger_weights_fn_gaussian(
    log_unnorm_posterior_fn: Callable[[jnp.ndarray], jnp.ndarray],
    regularization: float = DEFAULT_HELLINGER_COV_REGULARIZATION,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute Hellinger importance weights w^i = sqrt(pi(x^i) / p_t(x^i))
    using a Gaussian approximation for p_t.

    The weights are self-normalised so that their mean equals 1, recovering
    the KL scaling when p_t = pi.

    Args:
        log_unnorm_posterior_fn: Maps x [dofs] -> log pi(x) up to a constant.
        regularization: Tikhonov regularisation for the ensemble covariance.

    Returns:
        A function mapping x_s [ensemble, dofs] -> weights [ensemble].
    """

    def weights_fn(x_s: jnp.ndarray) -> jnp.ndarray:
        n_ens, d = x_s.shape

        # Ensemble mean and covariance (current p_t approximation)
        m_t = jnp.mean(x_s, axis=0)
        diff = x_s - m_t[None, :]
        cov_t = (diff.T @ diff) / jnp.maximum(n_ens - 1, 1) + regularization * jnp.eye(
            d
        )
        # cov_t_inv = jnp.linalg.inv(cov_t)

        # log p_t(x_i) under Gaussian approximation (up to additive constant)
        # log_pt = jax.vmap(lambda x: -0.5 * (x - m_t) @ cov_t_inv @ (x - m_t))(x_s)
        log_pt = jax.vmap(
            lambda x: -0.5 * (x - m_t) @ jnp.linalg.solve(cov_t, (x - m_t))
        )(x_s)

        # log pi(x_i) (up to additive constant)
        log_pi = jax.vmap(log_unnorm_posterior_fn)(x_s)

        # log(pi / p_t) with shift for numerical stability
        log_ratios = log_pi - log_pt
        log_ratios = log_ratios - jnp.max(log_ratios)

        # w^i = sqrt(pi / p_t), self-normalised to mean 1
        weights = jnp.exp(0.5 * log_ratios)
        weights = weights * n_ens / jnp.sum(weights)

        return weights

    return weights_fn


def get_hellinger_weights_fn_kde(
    log_unnorm_posterior_fn: Callable[[jnp.ndarray], jnp.ndarray],
    bandwidth: Optional[float] = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute Hellinger importance weights w^i = sqrt(pi(x^i) / p_t(x^i))
    using kernel density estimation for p_t.

    If no bandwidth is provided, Silverman's rule is used:
        h = (4 / (d + 2))^{1/(d+4)} * n^{-1/(d+4)} * sigma_median.

    Args:
        log_unnorm_posterior_fn: Maps x [dofs] -> log pi(x) up to a constant.
        bandwidth: KDE bandwidth. If None, Silverman's rule is applied.

    Returns:
        A function mapping x_s [ensemble, dofs] -> weights [ensemble].
    """

    def weights_fn(x_s: jnp.ndarray) -> jnp.ndarray:
        n_ens, d = x_s.shape

        # Bandwidth selection
        if bandwidth is not None:
            h = bandwidth
        else:
            # Silverman's rule with median absolute deviation
            median_std = jnp.median(jnp.std(x_s, axis=0))
            h = (
                median_std
                * jnp.power(4.0 / (d + 2.0), 1.0 / (d + 4.0))
                * jnp.power(n_ens * 1.0, -1.0 / (d + 4.0))
            )
            h = jnp.maximum(h, 1e-8)

        h_sq = h**2

        # Pairwise squared distances: ||x_i - x_j||^2
        # diff[i, j] = x_i - x_j
        diff = x_s[:, None, :] - x_s[None, :, :]  # [n_ens, n_ens, d]
        sq_dist = jnp.sum(diff**2, axis=-1)  # [n_ens, n_ens]

        # log k_h(x_i - x_j) = -||x_i - x_j||^2 / (2h^2) + const
        log_kernels = -sq_dist / (2.0 * h_sq)

        # log p_hat_t(x_i) = log(1/N * sum_j k_h(x_i - x_j))
        # = logsumexp_j(log_kernels[i,j]) - log(N) + const
        log_pt = jax.scipy.special.logsumexp(log_kernels, axis=1) - jnp.log(n_ens * 1.0)

        # log pi(x_i) (up to additive constant)
        log_pi = jax.vmap(log_unnorm_posterior_fn)(x_s)

        # log(pi / p_t) with shift for numerical stability
        log_ratios = log_pi - log_pt
        log_ratios = log_ratios - jnp.max(log_ratios)

        # w^i = sqrt(pi / p_t), self-normalised to mean 1
        weights = jnp.exp(0.5 * log_ratios)
        weights = weights * n_ens / jnp.sum(weights)

        return weights

    return weights_fn


# =============================================================================
# Right-hand side functions
# =============================================================================


def get_rhs_fn(
    posterior_score_fn: Callable[[jnp.ndarray], jnp.ndarray],
    divergence_kernel_fn: Callable[[jnp.ndarray], jnp.ndarray],
    kernel_matrix_fn: Callable[[jnp.ndarray], jnp.ndarray],
    divergence_type: Literal["kl", "hellinger"] = "kl",
    weights_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get the right-hand side function for the particle flow ODE.

    For divergence_type="kl" (Eq. 33 / KL-RKHS flow):
        f(x_i) = (1/N_p) sum_j [ nabla_j . K(x_j, x_i) + K(x_j, x_i) nabla log pi(x_j) ]

    For divergence_type="hellinger" (Eq. 45 / Hellinger-RKHS flow):
        f(x_i) = (1/(2 N_p)) sum_j w^j [ nabla_j . K(x_j, x_i) + K(x_j, x_i) nabla log pi(x_j) ]

    where w^j = sqrt(pi(x_j) / p_t(x_j)) are adaptive importance weights.

    Args:
        posterior_score_fn: Maps x_s [ensemble, dofs] -> posterior_score [ensemble, dofs].
        divergence_kernel_fn: Maps x_s -> [ensemble, ensemble, dofs].
        kernel_matrix_fn: Maps x_s -> [ensemble, ensemble, 1].
        divergence_type: "kl" for KL-based flow, "hellinger" for Hellinger-based flow.
        weights_fn: Required when divergence_type="hellinger". Maps x_s [ensemble, dofs]
            -> weights [ensemble]. Ignored when divergence_type="kl".
    """
    if divergence_type == "hellinger" and weights_fn is None:
        raise ValueError(
            "weights_fn must be provided when divergence_type='hellinger'."
        )

    def rhs_fn_kl(x_s: jnp.ndarray) -> jnp.ndarray:
        """KL-based RKHS particle flow (Eq. 33)."""
        posterior_score = posterior_score_fn(x_s)
        kernel_matrix = kernel_matrix_fn(x_s)
        divergence_kernel = divergence_kernel_fn(x_s)

        # Stein operator: S[i,j,:] = nabla_{x_j} . K(x_j, x_i) + K(x_j, x_i) nabla log pi(x_j)
        out = divergence_kernel + kernel_matrix[:, :, None] * posterior_score[None]
        out = out.sum(axis=1)

        return out / x_s.shape[0]

    def rhs_fn_hellinger(x_s: jnp.ndarray) -> jnp.ndarray:
        """Hellinger-based RKHS particle flow (Eq. 45).

        Identical Stein operator structure as KL, but weighted by
        w^j = sqrt(pi(x_j) / p_t(x_j)) and with a global factor of 1/2.
        """
        posterior_score = posterior_score_fn(x_s)
        kernel_matrix = kernel_matrix_fn(x_s)
        divergence_kernel = divergence_kernel_fn(x_s)

        # Stein operator (same as KL)
        stein_op = divergence_kernel + kernel_matrix[:, :, None] * posterior_score[None]

        # Apply importance weights along the source-particle axis
        weights = weights_fn(x_s)  # [ensemble]
        stein_op = stein_op * weights[None, :, None]

        out = stein_op.sum(axis=1)

        # Factor of 1/2 from the Hellinger first variation
        return out / (2.0 * x_s.shape[0])

    if divergence_type == "hellinger":
        return rhs_fn_hellinger
    return rhs_fn_kl


# =============================================================================
# Particle Flow Filter class
# =============================================================================


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
        kernel_type: str = "scalar",
        stepper: str = "runge_kutta_4",
        return_pff_trajectory: bool = False,
        divergence_type: Literal["kl", "hellinger"] = "kl",
        weight_estimation: Literal["gaussian", "kde"] = "gaussian",
        kde_bandwidth: Optional[float] = None,
        hellinger_cov_regularization: float = DEFAULT_HELLINGER_COV_REGULARIZATION,
    ) -> None:
        """
        Initialize the Particle Flow Filter.

        Args:
            ensemble_size (int): Number of ensemble members.
            R (numpy.array): Observation error covariance matrix.
            obs_operator (Callable[[np.ndarray], np.ndarray]): Observation operator.
            forward_operator (Callable[[np.ndarray], np.ndarray]): Forward operator.
            localization_distance (int): Localization distance.
            num_pseudo_time_steps (int): Number of pseudo-time steps.
            step_size (float): Step size.
            alpha (float): Alpha parameter.
            kernel_type (str): Type of kernel to use.
            stepper (str): Type of stepper to use.
            return_pff_trajectory (bool): If True, return the full pseudo-time trajectory.
            divergence_type (str): "kl" for the KL-based RKHS flow (Eq. 33),
                "hellinger" for the Hellinger-based RKHS flow (Eq. 45).
            weight_estimation (str): Method for estimating the Hellinger importance
                weights w^i = sqrt(pi(x^i) / p_t(x^i)). Only used when
                divergence_type="hellinger".
                - "gaussian": Approximate p_t as Gaussian from ensemble statistics.
                - "kde": Kernel density estimation for p_t.
            kde_bandwidth (float): KDE bandwidth. If None, Silverman's rule is used.
                Only used when weight_estimation="kde".
            hellinger_cov_regularization (float): Tikhonov regularisation for the
                ensemble covariance inverse in the Gaussian weight estimation.
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
        self.kernel_type = kernel_type
        self.stepper = stepper
        self.return_pff_trajectory = return_pff_trajectory
        self.is_linear_obs_operator = obs_operator.is_linear
        self.divergence_type = divergence_type
        self.weight_estimation = weight_estimation
        self.kde_bandwidth = kde_bandwidth
        self.hellinger_cov_regularization = hellinger_cov_regularization

        if self.localization_distance is None:
            self.localization = lambda x: x
        else:
            self.localization = lambda x: distance_based_localization(
                self.localization_distance, self.state_dim, x  # type: ignore[arg-type]
            )

    def _build_hellinger_weights_fn(
        self,
        prior_mean: jnp.ndarray,
        prior_cov_inv: jnp.ndarray,
        obs_vect: jnp.ndarray,
        obs_cov_inv: jnp.ndarray,
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Construct the weight function for the Hellinger flow.

        Builds log pi(x) from the prior and likelihood, then wraps it in the
        appropriate weight estimation method (Gaussian or KDE).
        """
        # Build log unnormalised posterior
        if self.is_linear_obs_operator:
            log_unnorm_posterior_fn = get_log_unnorm_posterior_fn_linear(
                prior_mean=prior_mean,
                prior_cov_inv=prior_cov_inv,
                obs_vect=obs_vect,
                obs_matrix=self.obs_operator.obs_matrix,  # type: ignore[attr-defined]
                obs_cov_inv=obs_cov_inv,
            )
        else:
            log_unnorm_posterior_fn = get_log_unnorm_posterior_fn_nonlinear(
                prior_mean=prior_mean,
                prior_cov_inv=prior_cov_inv,
                obs_vect=obs_vect,
                obs_operator=self.obs_operator,  # type: ignore[arg-type]
                obs_cov_inv=obs_cov_inv,
            )

        # Select weight estimation strategy
        if self.weight_estimation == "kde":
            return get_hellinger_weights_fn_kde(
                log_unnorm_posterior_fn=log_unnorm_posterior_fn,
                bandwidth=self.kde_bandwidth,
            )
        else:
            return get_hellinger_weights_fn_gaussian(
                log_unnorm_posterior_fn=log_unnorm_posterior_fn,
                regularization=self.hellinger_cov_regularization,
            )

    def _analysis_step(
        self,
        prior_ensemble: np.ndarray,
        obs_vect: np.ndarray,
        rng_key: Optional[jax.random.PRNGKey] = None,
        prior_mean: Optional[np.ndarray] = None,
        prior_cov: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:

        x_s = prior_ensemble.reshape(self.ensemble_size, -1)  # [ensemble, dofs]

        if prior_mean is None:
            prior_mean = jnp.mean(x_s, axis=0)
        if prior_cov is None:
            prior_cov = jnp.cov(x_s.T)
        prior_cov = self.localization(prior_cov)
        prior_cov_inv = jnp.linalg.inv(prior_cov)

        if len(prior_cov.shape) == 0:
            prior_cov = prior_cov.reshape(1, 1)

        # distance_weight_matrix = jnp.linalg.inv(self.alpha * prior_cov)
        distance_weight_matrix = jnp.eye(self.state_dim) * jnp.pi

        obs_cov_inv = jnp.linalg.inv(self.R)

        # Distributions
        prior_score_fn = get_prior_score_fn(
            prior_mean=prior_mean,
            prior_cov_inv=prior_cov_inv,
        )
        if self.is_linear_obs_operator:
            likelihood_score_fn = get_likelihood_score_fn_with_linear_obs_operator(
                obs_vect=obs_vect,
                obs_matrix=self.obs_operator.obs_matrix,  # type: ignore[attr-defined]
                obs_cov_inv=obs_cov_inv,
            )
        else:
            likelihood_score_fn = get_likelihood_score_fn_with_non_linear_obs_operator(
                obs_vect=obs_vect,
                obs_operator=self.obs_operator,  # type: ignore[arg-type]
                obs_cov_inv=obs_cov_inv,
            )
        posterior_score_vmap = jax.vmap(
            get_posterior_score_fn(
                prior_score_fn=prior_score_fn,
                likelihood_score_fn=likelihood_score_fn,
            ),
        )
        # Kernels
        divergence_kernel_fn = get_pairwise_interaction_fn(
            pair_fn=get_divergence_kernel_fn(
                kernel_type=self.kernel_type,
                distance_weight_matrix=distance_weight_matrix,
            ),
        )
        kernel_matrix_fn = get_pairwise_interaction_fn(
            pair_fn=get_kernel_matrix_fn(
                kernel_type=self.kernel_type,
                distance_weight_matrix=distance_weight_matrix,
            ),
        )

        # Hellinger weights (only constructed if needed)
        weights_fn = None
        if self.divergence_type == "hellinger":
            weights_fn = self._build_hellinger_weights_fn(
                prior_mean=prior_mean,
                prior_cov_inv=prior_cov_inv,
                obs_vect=obs_vect,
                obs_cov_inv=obs_cov_inv,
            )

        compile_rhs = True
        if compile_rhs:
            posterior_score_vmap = jax.jit(posterior_score_vmap)
            divergence_kernel_fn = jax.jit(divergence_kernel_fn)
            kernel_matrix_fn = jax.jit(kernel_matrix_fn)

        # RHS
        rhs_fn = get_rhs_fn(
            posterior_score_fn=posterior_score_vmap,
            divergence_kernel_fn=divergence_kernel_fn,
            kernel_matrix_fn=kernel_matrix_fn,
            divergence_type=self.divergence_type,
            weights_fn=weights_fn,
        )
        stepper = get_stepper(self.stepper, self.step_size, rhs_fn)

        # rollout_fn = rollout(
        #     stepper,
        #     self.num_pseudo_time_steps,
        #     return_inner_steps=self.return_pff_trajectory,
        # )
        # rollout_fn = jax.jit(rollout_fn)

        # x_s = rollout_fn(x_s)

        x = [x_s]
        for i in range(self.num_pseudo_time_steps):
            x.append(stepper(x[-1]))

        x_s = jnp.array(x)

        if self.return_pff_trajectory:
            x_s = jnp.transpose(x_s, (1, 0, 2))
            return (
                x_s.reshape(
                    self.ensemble_size,
                    self.num_pseudo_time_steps + 1,
                    self.num_states,
                    self.state_dim,
                ),
                rhs_fn,
            )

        return (x_s.reshape(self.ensemble_size, self.num_states, self.state_dim),)
