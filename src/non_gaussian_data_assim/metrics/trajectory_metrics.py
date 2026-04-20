from abc import abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp

AGGREGATION_METHODS = {
    "none": lambda x: x,
    "mean": lambda x: jnp.mean(x, axis=0),
    "median": lambda x: jnp.median(x, axis=0),
    "max": lambda x: jnp.max(x, axis=0),
    "min": lambda x: jnp.min(x, axis=0),
    "std": lambda x: jnp.std(x, axis=0),
    "var": lambda x: jnp.var(x, axis=0),
}


class TrajectoryMetric:
    def __init__(self, name: str, aggregation_method: Optional[str] = None):
        self.name = name
        self.aggregation_method = (
            aggregation_method if aggregation_method is not None else "none"
        )
        self.aggregation_method_fn = AGGREGATION_METHODS[self.aggregation_method]

    @abstractmethod
    def compute(self, preds: jnp.ndarray, truth: jnp.ndarray) -> float:
        raise NotImplementedError

    def compute_ensemble(self, preds: jnp.ndarray, truth: jnp.ndarray) -> float:
        compute_fun = lambda pred: self.compute(pred, truth)
        return jax.vmap(compute_fun)(preds)

    def __call__(self, preds: jnp.ndarray, truth: jnp.ndarray) -> float:
        return self.aggregation_method_fn(self.compute_ensemble(preds, truth))


class RMSE(TrajectoryMetric):
    def __init__(
        self,
        aggregation_method: Optional[str] = None,
    ):
        super().__init__("rmse", aggregation_method)

    def compute(self, preds: jnp.ndarray, truth: jnp.ndarray) -> float:
        return jnp.sqrt(jnp.mean((preds - truth) ** 2))


class MAE(TrajectoryMetric):
    def __init__(
        self,
        aggregation_method: Optional[str] = None,
    ):
        super().__init__("mae", aggregation_method)

    def compute(self, preds: jnp.ndarray, truth: jnp.ndarray) -> float:
        return jnp.mean(jnp.abs(preds - truth))


def print_metrics_table(
    prior_errors: dict, posterior_errors: dict, title: str = ""
) -> None:
    col_w = 14
    header = f"{'Metric':<10} | {'Prior':>{col_w}} | {'Posterior':>{col_w}}"
    sep = "-" * len(header)
    if title:
        print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for metric in prior_errors:
        prior_val = float(prior_errors[metric])
        post_val = float(posterior_errors[metric])
        print(f"{metric.upper():<10} | {prior_val:{col_w}.6f} | {post_val:{col_w}.6f}")
    print(sep)


class MAPE(TrajectoryMetric):
    def __init__(
        self,
        aggregation_method: Optional[str] = None,
    ):
        super().__init__("mape", aggregation_method)

    def compute(self, preds: jnp.ndarray, truth: jnp.ndarray) -> float:
        return jnp.mean(jnp.abs(preds - truth) / (jnp.abs(truth) + 1e-6))
