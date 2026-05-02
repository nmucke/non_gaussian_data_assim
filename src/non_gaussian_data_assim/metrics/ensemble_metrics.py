from abc import abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

AGGREGATION_METHODS = {
    "none": lambda x, axis: x,
    "mean": lambda x, axis: jnp.mean(x, axis=axis),
    "sum": lambda x, axis: jnp.sum(x, axis=axis),
    "median": lambda x, axis: jnp.median(x, axis=axis),
    "max": lambda x, axis: jnp.max(x, axis=axis),
    "min": lambda x, axis: jnp.min(x, axis=axis),
}


class EnsembleMetric:
    """Score an ensemble forecast against a truth trajectory.

    Subclasses implement ``compute`` for a single time step; ``__call__``
    vmaps that over the time axis and applies an optional time aggregation.
    """

    def __init__(self, name: str, time_aggregation: Optional[str] = None) -> None:
        self.name = name
        self.time_aggregation = (
            time_aggregation if time_aggregation is not None else "none"
        )

    @abstractmethod
    def compute(self, ensemble: jnp.ndarray, truth: jnp.ndarray) -> jnp.ndarray:
        """Compute metric for a single time step.

        Args:
            ensemble: shape (n_ensemble, *state_shape)
            truth:    shape (*state_shape,)

        Returns:
            Scalar score for this time step.
        """
        raise NotImplementedError

    def __call__(self, ensemble: jnp.ndarray, truth: jnp.ndarray) -> jnp.ndarray:
        """Compute metric over all time steps, then aggregate.

        Args:
            ensemble: shape (n_ensemble, n_time, *state_shape)
            truth:    shape (n_time, *state_shape)

        Returns:
            Aggregated metric. Shape depends on time_aggregation:
            - None:      (n_time,)
            - otherwise: scalar
        """
        compute_over_time = jax.vmap(self.compute, in_axes=(1, 0))
        result = compute_over_time(ensemble, truth)  # (n_time,)
        return AGGREGATION_METHODS[self.time_aggregation](result, axis=0)


def _crps_pointwise(ensemble_vals: jnp.ndarray, truth_val: jnp.ndarray) -> jnp.ndarray:
    """Fair CRPS for one scalar truth and its E ensemble samples.

    Uses the sort identity
        sum_{i,j} |x_i - x_j| = 2 * sum_k (2k - E - 1) * x_(k)
    (1-indexed, sorted ascending) so the spread term is O(E log E) instead of
    the O(E^2) pairwise form.
    """
    E = ensemble_vals.shape[0]
    mae_term = jnp.mean(jnp.abs(ensemble_vals - truth_val))
    sorted_ens = jnp.sort(ensemble_vals)
    i = jnp.arange(E)
    spread_term = jnp.sum((2 * i + 1 - E) * sorted_ens) / (E * (E - 1))
    return mae_term - spread_term


class CRPS(EnsembleMetric):
    """Continuous Ranked Probability Score, averaged over all state dimensions.

    For each entry of ``truth`` at a given time step this evaluates
        CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|
    using the sort-based fair estimator, then averages over the flattened
    state to return a scalar per time step.
    """

    def __init__(self, time_aggregation: Optional[str] = None) -> None:
        super().__init__("crps", time_aggregation)

    def compute(self, ensemble: jnp.ndarray, truth: jnp.ndarray) -> jnp.ndarray:
        truth_flat = truth.reshape(-1)
        ensemble_flat = ensemble.reshape(ensemble.shape[0], -1)
        pointwise = jax.vmap(_crps_pointwise, in_axes=(1, 0))(ensemble_flat, truth_flat)
        return jnp.mean(pointwise)


## Max: My method of computing crps,
def crps_ensemble_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute CRPS for an ensemble forecast at many points (scalar variable).

    Ensemble CRPS formula:
      CRPS = (1/K) sum_i |x_i - y| - 1/(2 K^2) * sum_{k}sum{k'} |x_k - x_k'|

    Parameters
    ----------
    x : (N, K) array ---> Ensemble
        N predictions wiht ensemble of isze K
    y : (N,) array ---> Truth
        N verifying observatios points.

    Returns
    -------
    crps : (N,) array
        CRPS per observation point.

    """

    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 2:
        raise ValueError(f"x must be 2D (N,K). Got shape {x.shape}")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError(
            f"y must be 1D (N,) with same number of obs for x. Got y {y.shape}, x {x.shape}"
        )

    N, K = x.shape

    # --- 1) MAE part of CRPS: (1/K) * sum_k |x_k - y|
    mae = (1 / K) * np.sum(np.abs(x - y[:, None]), axis=1)  # (N,)

    # --- 2) MeanAbsDiff part of CRPS:  1/(2 K^2) * sum_{k} sum{k'} |x_k - x_k'|
    mad = np.zeros(N)
    for n in range(N):
        s = 0.0
        for k in range(K):
            for kp in range(K):
                s += abs(x[n, k] - x[n, kp])
        mad[n] = s / (2.0 * K * K)

    return mae - mad


## Max: Innovation metrics (Chi-sq + Normalized-Innov-Ranks)


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
