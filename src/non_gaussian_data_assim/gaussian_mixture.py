import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular


def gaussian_mixt(
    weight_vect: np.ndarray,
    n_obs: int,
    ens_vect: np.ndarray,
    obs_vect: np.ndarray,
    h_matrix: np.ndarray,
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute the AGMF mixture weights for ensemble members.

    Implements the adaptive Gaussian-mixture weight update

        w_i ~ w_prev_i * N(y ; H x_i^f, S),  S = H Sigma H^T + R,

    evaluated at the actual (unperturbed) observation ``y`` using the FORECAST
    (prior) members ``x_i^f``. The computation is done entirely in the log
    domain and normalized with a log-sum-exp trick so it stays finite for large
    observation dimensions (e.g. p ~ 91).

    Args:
    weight_vect (numpy.array): Current weights of the ensemble members, shape [N].
    n_obs (int): Number of observations (p).
    ens_vect (numpy.array): FORECAST ensemble matrix, shape [dofs, N].
    obs_vect (numpy.array): Unperturbed observation vector y, shape [n_obs].
    h_matrix (numpy.array): Observation operator matrix, shape [n_obs, dofs].
    cov_matrix (numpy.array): Innovation covariance S = H Sigma H^T + R,
        shape [n_obs, n_obs].

    Returns:
    numpy.array: Updated, normalized weights for the ensemble members, shape [N].
    """
    # Cholesky factor of the (shared) innovation covariance S = L L^T. Factor
    # once and reuse for both the quadratic form and the log-determinant.
    chol = jnp.linalg.cholesky(cov_matrix)

    # innovations[:, i] = y - H @ ens_vect[:, i]  -> shape (n_obs, N)
    innovations = obs_vect.reshape(-1, 1) - h_matrix @ ens_vect

    # Per-member quadratic form r_i^T S^-1 r_i = ||L^-1 r_i||^2 via a triangular
    # solve (avoids forming S^-1 and is numerically stable).
    z = solve_triangular(chol, innovations, lower=True)
    quad_form = jnp.sum(z**2, axis=0)

    # log|S| = 2 * sum(log(diag(L))).
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))

    # Log mixture weights. The 0.5*logdet and 0.5*p*log(2pi) terms are shared
    # across members (S is shared) and cancel in the normalization, but are kept
    # for correctness/clarity; log-sum-exp keeps everything finite regardless.
    log_w = (
        jnp.log(weight_vect)
        - 0.5 * quad_form
        - 0.5 * logdet
        - 0.5 * n_obs * jnp.log(2 * jnp.pi)
    )

    # Numerically stable normalization (log-sum-exp).
    log_w = log_w - jnp.max(log_w)
    weight_mixt = jnp.exp(log_w)
    weight_final = weight_mixt / jnp.sum(weight_mixt)

    return weight_final
