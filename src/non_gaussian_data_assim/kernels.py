import functools
from typing import Any, Callable

import jax.numpy as jnp

from non_gaussian_data_assim.jax_utils import compute_pairwise_interaction


def exp_scalar_kernel_fn(
    x: jnp.ndarray,
    y: jnp.ndarray,
    distance_weight_matrix: jnp.ndarray,
) -> jnp.ndarray:
    """Get the scalar kernel function."""
    return jnp.exp(-0.5 * (x - y).T @ distance_weight_matrix @ (x - y))


def get_kernel_fn(
    kernel_type: str,
    **kwargs: Any,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Get the kernel function."""

    if kernel_type == "scalar":
        return functools.partial(exp_scalar_kernel_fn, **kwargs)
    else:
        raise ValueError(
            f"Invalid kernel type: {kernel_type}. We only support 'scalar' kernel type."
        )


def get_kernel_matrix_fn(
    kernel_type: str,
    **kwargs: Any,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Get the kernel matrix function."""

    if kernel_type == "scalar":
        scalar_kernel_fn = get_kernel_fn(kernel_type, **kwargs)

        def scalar_kernel_matrix_fn(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            """Calculate the scalar kernel matrix."""
            return scalar_kernel_fn(x, y) * jnp.eye(x.shape[0])

        return scalar_kernel_matrix_fn
    else:
        raise ValueError(
            f"Invalid kernel type: {kernel_type}. We only support 'scalar' kernel type."
        )


def get_divergence_kernel_fn(
    kernel_type: str,
    **kwargs: Any,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Get the divergence of the kernel function."""

    if kernel_type == "scalar":
        scalar_kernel_fn = get_kernel_fn(kernel_type, **kwargs)
        distance_weight_matrix = kwargs.get("distance_weight_matrix")

        def divergence_scalar_kernel_matrix_fn(
            x: jnp.ndarray, y: jnp.ndarray
        ) -> jnp.ndarray:
            """Calculate the divergence of the scalar kernel matrix."""
            return -distance_weight_matrix.T @ (x - y) * scalar_kernel_fn(x, y)  # type: ignore[union-attr]

        return divergence_scalar_kernel_matrix_fn
    else:
        raise ValueError(
            f"Invalid kernel type: {kernel_type}. We only support 'scalar' kernel type."
        )


def get_pairwise_interactions_fn(
    kernel_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get the pairwise interactions function."""

    def pairwise_interactions_fn(x: jnp.ndarray) -> jnp.ndarray:
        """Compute the pairwise interactions."""
        return compute_pairwise_interaction(x, kernel_fn)

    return pairwise_interactions_fn
