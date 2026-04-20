from abc import abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp

AGGREGATION_METHODS = {
    "none": lambda x: x,
    "mean": lambda x: jnp.mean(x),
    "sum": lambda x: jnp.sum(x),
    "median": lambda x: jnp.median(x),
    "max": lambda x: jnp.max(x),
    "min": lambda x: jnp.min(x),
}


class EnsembleMetric:
    """Score an ensemble forecast as a whole against a truth trajectory.

    Unlike ``TrajectoryMetric`` — which scores each member against truth and
    then aggregates — an ``EnsembleMetric`` consumes the ensemble axis inside
    ``compute`` and returns a per-point score field shaped like ``truth``.
    ``aggregation_method`` then reduces that field to a scalar, or leaves it
    pointwise with ``"none"``.
    """

    def __init__(self, name: str, aggregation_method: Optional[str] = None):
        self.name = name
        self.aggregation_method = (
            aggregation_method if aggregation_method is not None else "none"
        )
        self.aggregation_method_fn = AGGREGATION_METHODS[self.aggregation_method]

    @abstractmethod
    def compute(self, ensemble: jnp.ndarray, truth: jnp.ndarray) -> jnp.ndarray:
        """Score the ensemble against truth.

        Args:
            ensemble: array of shape ``[E, *truth.shape]`` — the forecast ensemble.
            truth:    array of shape ``truth.shape`` — the ground-truth trajectory.

        Returns:
            Per-point score array with shape equal to ``truth.shape``.
        """
        raise NotImplementedError

    def __call__(self, ensemble: jnp.ndarray, truth: jnp.ndarray) -> jnp.ndarray:
        if ensemble.shape[1:] != truth.shape:
            raise ValueError(
                f"ensemble shape {ensemble.shape} incompatible with truth shape "
                f"{truth.shape}; expected ensemble.shape[1:] == truth.shape."
            )
        return self.aggregation_method_fn(self.compute(ensemble, truth))


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
    # fair (unbiased) estimator of 0.5 * E|X - X'|
    spread_term = jnp.sum((2 * i + 1 - E) * sorted_ens) / (E * (E - 1))
    return mae_term - spread_term


class CRPS(EnsembleMetric):
    """Continuous Ranked Probability Score, computed pointwise then aggregated.

    For each entry of ``truth`` this evaluates
        CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|
    using the sort-based fair estimator over the E ensemble samples. The
    returned field has the same shape as ``truth``; the instance-level
    ``aggregation_method`` then reduces it.
    """

    def __init__(self, aggregation_method: Optional[str] = None):
        super().__init__("crps", aggregation_method)

    def compute(self, ensemble: jnp.ndarray, truth: jnp.ndarray) -> jnp.ndarray:
        truth_flat = truth.reshape(-1)
        ensemble_flat = ensemble.reshape(ensemble.shape[0], -1)
        pointwise = jax.vmap(_crps_pointwise, in_axes=(1, 0))(ensemble_flat, truth_flat)
        return pointwise.reshape(truth.shape)
