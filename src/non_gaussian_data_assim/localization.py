import jax.numpy as jnp


def distance_based_localization(
    r_influ: int, state_dim: int, cov_prior: jnp.ndarray, num_states: int | None = None
) -> jnp.ndarray:
    """
    Apply localization to the covariance matrix.

    Args:
    r_influ (int): The radius of influence for localization -- grid cells.
    state_dim (int): The dimension of the state vector.
    cov_prior (jax.numpy.array): The prior covariance matrix.
            with shape: [num_states * state_dim, num_states * state_dim]

    Returns:
    jax.numpy.array: Localized covariance matrix.
    """

    # -- Check that if no Number of State is passed, that assumption that there is 1 state is correct
    if num_states is None:
        if cov_prior.shape[0] % state_dim != 0:
            raise ValueError(
                "Num_state was None (assumed to be 1), however "
                f"cov_prior shape {cov_prior.shape} is incompatible with state_dim={state_dim}"
            )
        num_states = cov_prior.shape[0] // state_dim
    # -- Final check covariance matrix has expected shape
    expected_dim = num_states * state_dim
    if cov_prior.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"Expected covariance shape {(expected_dim, expected_dim)}, "
            f"got {cov_prior.shape}. num_states={num_states}, state_dim={state_dim}."
        )

    # --- Periodic mask # TODO: Is this mask periodic
    idx = jnp.arange(state_dim)
    dist = jnp.abs(idx[:, None] - idx[None, :])
    dist = jnp.minimum(dist, state_dim - dist)  # periodic distance

    mask = jnp.exp(-(dist**2) / r_influ**2)
    mask = jnp.where(dist <= 3 * r_influ, mask, 0.0)

    # Apply the localization mask to the prior covariance matrix
    cov_prior_loc = jnp.zeros(cov_prior.shape)
    for i in range(1, num_states + 1):
        for j in range(1, num_states + 1):
            # -- Set indices of current block
            row_slice = slice((i - 1) * state_dim, i * state_dim)
            col_slice = slice((j - 1) * state_dim, j * state_dim)
            # -- Localise by multiplying with mask
            localised_block = jnp.multiply(cov_prior[row_slice, col_slice], mask)
            # -- Assign localised block to new covariance matrix
            cov_prior_loc = cov_prior_loc.at[row_slice, col_slice].set(localised_block)

    return cov_prior_loc
