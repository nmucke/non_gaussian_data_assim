from typing import Callable

import jax
import jax.numpy as jnp


def compute_pairwise_interaction(
    x_s: jnp.ndarray, pair_function: Callable
) -> jnp.ndarray:
    """
    Compute the pairwise interaction of the ensemble.

    Args:
    x_s: State array of shape [dofs, ensemble].
    pair_function: Function f(x, y) taking two [dofs] vectors, returns [dofs].

    Returns:
    Array of shape [dofs, ensemble, ensemble] where
    result[:, i, j] = pair_function(x_s[:, i], x_s[:, j]).
    """
    # Inner vmap over j: for fixed i, compute pair_function(x_s[:, i], x_s[:, j]) for all j
    # in_axes=(None, 1): batch over columns (axis 1) of second arg
    # out_axes=1: stack j along axis 1 -> [dofs, ensemble]
    inner_vmap = jax.vmap(pair_function, in_axes=(None, 1), out_axes=1)

    # Outer vmap over i: for each i, run inner_vmap(x_s[:, i], x_s)
    # in_axes=(1, None): batch over columns (axis 1) of first arg
    # out_axes=1: stack i along axis 1 -> [dofs, ensemble, ensemble]
    vmap_func = jax.vmap(inner_vmap, in_axes=(1, None), out_axes=1)

    return vmap_func(x_s, x_s)
