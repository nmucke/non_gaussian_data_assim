"""
Information metrics for DA according to paper of Samantha Kim and Nearing et al.

Notation:
    X  : open-loop / prior model prediction
    Y  : assimilated data / retrieval
    Z  : independent evaluation data
    Xp : analysis / posterior model prediction

Implementation details and maybe points I was unsure about:
    - probabilities are normalized histogram counts, not probability densities <-- Is this a valid approximation?
    - entropies use natural logarithms instead of base 2
    - Validation observation are created like the normal observation, just with a different sample of random noise
"""

from typing import Any, Optional

import jax.numpy as jnp


def _as_1d_samples(values: jnp.ndarray, name: str) -> jnp.ndarray:
    """Convert input to a finite 1D sample vector."""
    samples = jnp.asarray(values).reshape(-1)
    if samples.size == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    return samples


def _align_samples(*named_samples: tuple[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    """Flatten arrays, check equal length, and remove non-finite sample pairs."""

    # --- Save Arrays as dict, where key is variable (e.g. Y or Z) and value is flattened array
    arrays = {name: _as_1d_samples(values, name=name) for name, values in named_samples}

    # --- Check if all the arrays have the same length
    sizes = {name: values.size for name, values in arrays.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"All sample arrays must have the same length. Got {sizes}.")

    # # --- Some checks that all values are finite
    # finite_mask = jnp.ones(next(iter(arrays.values())).size, dtype=bool)
    # for values in arrays.values():
    #     finite_mask &= jnp.isfinite(values)
    # if not jnp.any(finite_mask):
    #     raise ValueError("No finite paired samples remain after filtering.")
    # return {name: values[finite_mask] for name, values in arrays.items()}

    return {name: values for name, values in arrays.items()}


def _check_n_bins(n_bins: int) -> None:
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}.")


def _normalize_counts(counts: jnp.ndarray) -> jnp.ndarray:
    total = jnp.sum(counts)
    if total <= 0:
        raise ValueError("Cannot normalize an empty histogram.")
    return counts / total


def _entropy_from_probability(probability: jnp.ndarray, n_bins: int) -> float | Any:
    """Compute normalized discrete entropy from probabilities."""
    p = probability[probability > 0.0]
    if p.size == 0:
        return 0.0
    return -jnp.sum(p * jnp.log(p)) / jnp.log(n_bins)


def probability_1d(
    values: jnp.ndarray,
    n_bins: int,
    value_range: tuple[float, float] | None = None,
) -> jnp.ndarray:
    """Return normalized 1D histogram probabilities."""
    _check_n_bins(n_bins)
    samples = _as_1d_samples(values, name="values")
    samples = samples[jnp.isfinite(samples)]
    counts, _ = jnp.histogram(samples, bins=n_bins, range=value_range)
    return _normalize_counts(counts)


def probability_nd(
    columns: list[jnp.ndarray] | tuple[jnp.ndarray, ...],
    n_bins: int,
    value_range: tuple[float, float] | None = None,
) -> jnp.ndarray:
    """Return normalized joint histogram probabilities.

    Each entry in ``columns`` is one random variable. All variables are flattened
    and paired sample-by-sample.
    """
    _check_n_bins(n_bins)
    named = [(f"var_{i}", values) for i, values in enumerate(columns)]
    aligned = _align_samples(*named)
    matrix = jnp.column_stack([aligned[f"var_{i}"] for i in range(len(columns))])

    histogram_range = None
    if value_range is not None:
        histogram_range = [value_range for _ in columns]

    counts, _ = jnp.histogramdd(matrix, bins=n_bins, range=histogram_range)
    return _normalize_counts(counts)


def entropy(
    values: jnp.ndarray,
    n_bins: int,
    value_range: tuple[float, float] | None = None,
) -> float:
    """Compute normalized empirical entropy H(X)."""
    probability = probability_1d(values, n_bins=n_bins, value_range=value_range)
    return _entropy_from_probability(probability, n_bins=n_bins)


def joint_entropy(
    columns: list[jnp.ndarray] | tuple[jnp.ndarray, ...],
    n_bins: int,
    value_range: tuple[float, float] | None = None,
) -> float:
    """Compute normalized empirical joint entropy H(X, Y, ...)."""
    probability = probability_nd(columns, n_bins=n_bins, value_range=value_range)
    return _entropy_from_probability(probability, n_bins=n_bins)


def _safe_divide(numerator: float, denominator: float) -> float:
    _EPS = 1e-10
    if abs(denominator) < _EPS:
        return float("nan")
    return float(numerator / denominator)


def mutual_information_from_entropies(
    h_a: float,
    h_b: float,
    h_ab: float,
    normalize_by: float | None = None,
) -> float:
    """Compute I(A;B) = H(A) + H(B) - H(A,B)"""
    information = h_a + h_b - h_ab
    if normalize_by is None:
        return float(information)
    return _safe_divide(information, normalize_by)


def interaction_information(
    h_x: float,
    h_y: float,
    h_z: float,
    h_xy: float,
    h_xz: float,
    h_yz: float,
    h_xyz: float,
) -> float:
    """Compute I(X;Y;Z) from normalized entropy terms"""
    return (h_x + h_y + h_z) - (h_xy + h_xz + h_yz) + h_xyz


def compute_information_metrics(
    x_model: jnp.ndarray,
    y_data: jnp.ndarray,
    z_eval: jnp.ndarray,
    x_analysis: jnp.ndarray | None = None,
    n_bins: int = 20,
    value_range: tuple[float, float] | None = None,
) -> dict:
    """
    Compute empirical information metrics from aligned and paired samples

    Args:
        x_model (X): Open-loop --> reference ensembles (NO data-assimilation was ever done)
        y_data (Y): Assimilated data
        z_eval (Z): Independent evaluation samples, NOT necarsilly the same as validation observations!
        x_analysis (X+): Posterior model
        n_bins: Nr. hist bins per variable
        value_range: Optional histogram range, e.g. (-10, 10)

    Returns:
        Dict with all the quantities as in the entropy_metrics.py script (see teams-chat of Femke, Nikolaj and Max or ask Femke)
    """
    _check_n_bins(n_bins)

    if x_analysis is None:
        aligned = _align_samples(
            ("x_model", x_model),
            ("y_data", y_data),
            ("z_eval", z_eval),
        )
    else:
        aligned = _align_samples(
            ("x_model", x_model),
            ("y_data", y_data),
            ("z_eval", z_eval),
            ("x_analysis", x_analysis),
        )

    x = aligned["x_model"]
    y = aligned["y_data"]
    z = aligned["z_eval"]

    h_x = entropy(x, n_bins=n_bins, value_range=value_range)
    h_y = entropy(y, n_bins=n_bins, value_range=value_range)
    h_z = entropy(z, n_bins=n_bins, value_range=value_range)

    h_xy = joint_entropy((x, y), n_bins=n_bins, value_range=value_range)
    h_xz = joint_entropy((x, z), n_bins=n_bins, value_range=value_range)
    h_yz = joint_entropy((y, z), n_bins=n_bins, value_range=value_range)
    h_xyz = joint_entropy((x, y, z), n_bins=n_bins, value_range=value_range)

    # Same normalization convention as the reference script:
    # I(Z;Y) and I(Z;X) are normalized by H(Z).
    i_z_y = mutual_information_from_entropies(h_y, h_z, h_yz, normalize_by=h_z)
    i_z_x = mutual_information_from_entropies(h_x, h_z, h_xz, normalize_by=h_z)
    i_xyz = interaction_information(
        h_x=h_x,
        h_y=h_y,
        h_z=h_z,
        h_xy=h_xy,
        h_xz=h_xz,
        h_yz=h_yz,
        h_xyz=h_xyz,
    )

    # Same formula as in the reference script.
    i_z_y_given_x = i_z_y * h_z - i_xyz
    i_z_xy = i_z_x * h_z + i_z_y_given_x
    total_information = _safe_divide(i_z_xy, h_z)

    h_x_analysis = None
    h_x_analysis_z = None
    i_z_x_analysis = None
    data_assimilation_efficiency = None
    data_efficiency = None

    if x_analysis is not None:
        xp = aligned["x_analysis"]
        h_x_analysis = entropy(xp, n_bins=n_bins, value_range=value_range)
        h_x_analysis_z = joint_entropy((xp, z), n_bins=n_bins, value_range=value_range)
        i_z_x_analysis = mutual_information_from_entropies(
            h_x_analysis, h_z, h_x_analysis_z, normalize_by=h_z
        )
        # Use unnormalized information for efficiency ratios.
        info_z_x_analysis = i_z_x_analysis * h_z
        info_z_x = i_z_x * h_z
        data_assimilation_efficiency = _safe_divide(info_z_x_analysis, i_z_xy)
        data_efficiency = _safe_divide(info_z_x_analysis - info_z_x, i_z_y_given_x)

    return dict(
        h_x=h_x,
        h_y=h_y,
        h_z=h_z,
        h_xy=h_xy,
        h_xz=h_xz,
        h_yz=h_yz,
        h_xyz=h_xyz,
        i_z_y=i_z_y,
        i_z_x=i_z_x,
        i_xyz=i_xyz,
        i_z_y_given_x=i_z_y_given_x,
        i_z_xy=i_z_xy,
        total_information=total_information,
        h_x_analysis=h_x_analysis,
        h_x_analysis_z=h_x_analysis_z,
        i_z_x_analysis=i_z_x_analysis,
        data_assimilation_efficiency=data_assimilation_efficiency,
        data_efficiency=data_efficiency,
        n_samples=int(x.size),
        n_bins=int(n_bins),
    )
