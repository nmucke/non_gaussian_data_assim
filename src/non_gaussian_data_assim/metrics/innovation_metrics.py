from abc import ABC, abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

AGGREGATION_METHODS = {
    "none": lambda x, axis: x,
    "mean": lambda x, axis: jnp.mean(x, axis=axis),
    "median": lambda x, axis: jnp.median(x, axis=axis),
    "max": lambda x, axis: jnp.max(x, axis=axis),
    "min": lambda x, axis: jnp.min(x, axis=axis),
}


class InnovationMetric(ABC):
    """
    Base class for innovation-space diagnostics.

    Expected input shapes:
        predicted_obs: (n_ensemble, n_time, n_obs)
        obs:           (n_time, n_obs)
        R:             (n_obs, n_obs)

    These metrics use the ensemble dimension to estimate S = H P^f H^T + R.
    Therefore, there is no ensemble aggregation.
    """

    def __init__(
        self,
        name: str,
        time_aggregation: Optional[str] = None,
    ) -> None:
        self.name = name
        self.time_aggregation = time_aggregation or "none"

        if self.time_aggregation not in AGGREGATION_METHODS:
            raise ValueError(self.time_aggregation)

    @abstractmethod
    def compute(
        self, predicted_obs: jnp.ndarray, obs: jnp.ndarray, R: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the metric for ONE assimilation time step.
        Inputs:
            predicted_obs: shape (n_ensemble, n_obs)
            obs:           shape (n_obs,)
            R:             shape (n_obs, n_obs)
        Returns:
            Innovation-metric for a given time step.
        """
        raise NotImplementedError

    def __call__(
        self, predicted_obs: jnp.ndarray, obs: jnp.ndarray, R: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute metric over ALL assimilation time steps.
        Inputs:
            predicted_obs: shape (n_ensemble, n_obs)
            obs:           shape (n_obs,)
            R:             shape (n_obs, n_obs)
        Returns:
            Metirc per time step (If time_aggregation="none"), else time-mean (if time_aggregation="mean")
        """

        # --- 1) Some checks to verify dims! Especially since innovation has different time-steps and locations than state!
        if predicted_obs.ndim != 3:
            raise ValueError(
                f"predicted_obs must have shape (n_ensemble, n_time, n_obs). Got shape {predicted_obs.shape}."
            )
        if obs.ndim != 2:
            raise ValueError(
                f"obs must have shape (n_time, n_obs). Got shape {obs.shape}."
            )
        if predicted_obs.shape[1] != obs.shape[0]:
            raise ValueError(
                f"Time dimension mismatch between predicted_obs: {predicted_obs.shape[0]} and obs has: {obs.shape[0]}"
            )
        if predicted_obs.shape[2] != obs.shape[1]:
            raise ValueError(
                f"Observation dimension mismatch betweem predicted_obs: {predicted_obs.shape[1]} and obs: {obs.shape[1]}"
            )

        # --- 2) Map compute function over axes in argument
        compute_over_time = jax.vmap(self.compute, in_axes=(1, 0, None))

        # --- 3) Compute over time
        # NOTE: Shape of metric depends on the metric's compute():
        #           NormalizedInnovations -> (n_time, n_obs)
        #           ChiSquared            -> (n_time,)
        result = compute_over_time(predicted_obs, obs, R)
        result = AGGREGATION_METHODS[self.time_aggregation](result, axis=0)

        return result


class NormalizedInnovations(InnovationMetric):
    """
    Compute whitened (normalized) innovations:      z = S^{-1/2} d
    where:
        d = y_o - mean(H(X_f))
        S = H P_f H^T + R

    The ensemble estimate is:
        H P_f H^T = Y_f Y_f^T / (K - 1)
        with Y_f being the ensemble anomalies in observation space.

    Source: Reichle et al. 2002: https://doi.org/10.1175/1525-7541(2002)003<0728:EVEKFF>2.0.CO;2
    """

    def __init__(
        self,
        time_aggregation: Optional[str] = None,
    ) -> None:
        super().__init__("normalized_innovations", time_aggregation)

    def _cholesky_S(self, U: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
        """
        Build S = U @ U.T + R, then retunr lower-triangular L such that
            S = L @ L.T

        U has shape (N, K).
        R is observation-error covariance matrix.
        """

        S = U @ U.T + R
        # Apply Cholesky to find lower triangular matrix
        L = jnp.linalg.cholesky(S)
        return L

    def normalized_innovations(
        self,
        predicted_obs: jnp.ndarray,
        obs: jnp.ndarray,
        R: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute normalized innovations for one assimilation step.

        Args:
            predicted_obs: shape (n_ensemble, n_obs)
            obs:           shape (n_obs,)
            R:             shape (n_obs, n_obs)

        Returns:
            z: shape (n_obs,)

        NOTE Why this function is not called directly compute()?
            --> Because child-class (ChiSquared) has to call it, but already has its own compute()
                It would get tricky to call the avoid self-calling?
        """

        n_ensemble, n_obs = predicted_obs.shape

        if n_ensemble < 2:
            raise ValueError(
                "Need at least two ensemble members to calc normalized innovations."
            )

        # --- 1) Forecast mean and anomalies in observation space
        predicted_obs_mean = jnp.mean(predicted_obs, axis=0)
        # Ensemble anomalies in observation space.
        anomalies = predicted_obs - predicted_obs_mean[None, :]

        # --- 2) Sample covariance factor  H P_f H^T = U U^T
        # U = anomalies / jnp.sqrt(n_ensemble - 1)
        U = anomalies.T / jnp.sqrt(n_ensemble - 1)

        # --- 3) Innovation
        d = obs - predicted_obs_mean

        # --- 4) Cholesky factorization of S ---> S = H P_f H^T + R = L L^T
        L = self._cholesky_S(U, R)

        # --- 5) Whitened innovatiom --->  z = L^{-1} d
        z = solve_triangular(L, d, lower=True)

        return z

    def compute(
        self, predicted_obs: jnp.ndarray, obs: jnp.ndarray, R: jnp.ndarray
    ) -> jnp.ndarray:
        return self.normalized_innovations(predicted_obs, obs, R)


class ChiSquared(NormalizedInnovations):
    """
    Compute normalized chi^2 test for a given assimilation step.

    Calcualtion:
    ------------
    For one assimilation step:
        chi2_norm = d^T S^{-1} d / n_obs

    Since z = S^{-1/2} d, this is equivalent to:
        chi2_norm = mean(z^2)

    Interpretation:
    ---------------
    If the ensemble forecast uncertainty and observation-error covariance are
    statistically consistent, the expected value is 1.

    Source: Menard et al. 2000 --> https://doi.org/10.1175/1520-0493(2000)128<2654:AOSCTO>2.0.CO;2
    """

    def __init__(self, time_aggregation: Optional[str] = None) -> None:
        super().__init__(time_aggregation=time_aggregation)
        self.name = "chi_squared"

    def compute(
        self, predicted_obs: jnp.ndarray, obs: jnp.ndarray, R: jnp.ndarray
    ) -> jnp.ndarray:

        z = self.normalized_innovations(predicted_obs, obs, R)
        chi2_norm = jnp.mean(z**2)
        return chi2_norm
