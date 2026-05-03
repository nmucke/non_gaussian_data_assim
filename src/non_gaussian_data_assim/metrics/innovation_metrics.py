import jax
import jax.numpy as jnp
import numpy as np


def _cholesky_S(U: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Build S = R + U@U.T with diagonal R = diag(sigma_vec**2),
    then return the lower diag L s.t. S = L@L.T

    U has shape (N, K), where U@U.T is the sample forecast covariance
    in observation space.
    sigma_vec has shape (N,) and contains observation-error std devs.
    """
    S = (U @ U.T) + R
    # Cholesky to find lower triangular matrix
    L = np.linalg.cholesky(S)
    return L


def _as_matrix_R(sigma_obs: float | np.ndarray, p: int) -> np.ndarray:
    """
    Return the observation-error covariance matrix R of shape (p, p).

    If sigma_obs is a scalar, return diag(sigma_obs_1**2,...,sigma_obs_p**2) of dim (p,p)
    If sigma_obs is an array, it must have shape (p,) and retrun diag(sigma_obs**2)

    """
    if np.isscalar(sigma_obs):
        sigma_vec = np.full(p, float(sigma_obs), dtype=float)
    else:
        sigma_vec = np.asarray(sigma_obs, dtype=float)
        if sigma_vec.shape != (p,):
            raise ValueError(
                f"sigma_obs must be a scalar or have shape ({p},), got {sigma_vec.shape}"
            )

    if np.any(sigma_vec < 0):
        raise ValueError("sigma_obs must be nonnegative.")

    return np.diag(sigma_vec**2)


def normalized_innovations(
    HXf: np.ndarray,
    yo: np.ndarray,
    sigma_obs: float | np.ndarray,
) -> np.ndarray:
    """
    Compute whitened / normalized innovations
        z = S^{-1/2} d = L^{-1} d,

    where
        d = yo - mean(HXf),
        S = HPfH^T + R ≈ U U^T + R,
        U = Yp / sqrt(K-1),
        S = L L^T.

    Inputs:
        - HXf: np.ndarray (N,K) Prior ensemble, in observation space (N:#obs  &  K:ens-size)
        - yo: np.ndarray (N)    Observations at current analysis step

    Output:
        z : ndarray (N,)        Normalized innovations
    """

    N, K = HXf.shape
    if K < 2:
        raise ValueError("Need at least 2 ensemble members.")

    # --- Construct Observation Error covariance matrix
    R = _as_matrix_R(sigma_obs, N)

    # --- 1) Get Mean + Forecast-Anomalies (Yp) in Obs-Space
    HXf_bar = HXf.mean(axis=1)  # shape (N,)
    Yp = HXf - HXf_bar[:, None]  # shape (N, K)

    # --- 2) Sample covariance factor in observation space:               # U @ U.T = (1/(K-1)) * Yp@Yp.T ≈ H P^f H.T
    U = Yp / np.sqrt(K - 1)  # shape (N, K)

    # --- 3) Innovation: observed minus forecast mean
    d = yo - HXf_bar  # shape (N,)

    # --- 4) Calc np.inv(S) using Cholesky  S = R + H @ P^f @ H.T = U @ U.T
    L = _cholesky_S(U, R)

    # --- 5) Whitened innovation
    z = np.linalg.solve(L, d)  # z = L^{-1} d
    return z


def chi_squared(
    HXf: np.ndarray | jnp.ndarray,
    yo: np.ndarray | jnp.ndarray,
    sigma_obs: float | np.ndarray,
) -> np.ndarray:
    """
    Compute normalized chi^2 test for a given assimilation step.
    Source: https://doi.org/10.1175/1520-0493(2000)128<2654:AOSCTO>2.0.CO;2

    Output:
        - chi2_norm: float          Normalized chi^2 value --> If uncertainty is estimated accurately, this value is 1
    """
    z = normalized_innovations(HXf, yo, sigma_obs)
    return (z @ z) / z.size


def normalized_innovation_histogram(
    HXf: np.ndarray | jnp.ndarray,
    yo: np.ndarray | jnp.ndarray,
    sigma_obs: float | np.ndarray,
    bins: int | np.ndarray,
    density: bool = True,
    hist_range: tuple[float, float] | None = (-5.0, 5.0),
) -> np.ndarray:
    """
    Compute normalized innovation histogram.
    Source: Reichle et al. 2002: https://doi.org/10.1175/1525-7541(2002)003<0728:EVEKFF>2.0.CO;2
    """
    z = normalized_innovations(HXf, yo, sigma_obs)
    hist, bin_edges = np.histogram(z, bins=bins, range=hist_range, density=density)
    return hist, bin_edges, z
