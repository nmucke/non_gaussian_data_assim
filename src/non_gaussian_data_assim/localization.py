import jax.numpy as jnp


def distance_based_localization(
    r_influ: int,
    state_dim: int,
    cov_prior: jnp.ndarray,
    num_states: int | None = None,
    periodic: bool = False,
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

    Notes:
    The taper is the Gaspari-Cohn (1999, eq. 4.10) 5th-order piecewise-rational
    correlation function, which is compactly supported AND positive
    semi-definite by construction. This guarantees the Schur (elementwise)
    product with the sample covariance stays PSD (unlike a hard-truncated
    Gaussian, which is not a valid correlation function).

    ``r_influ`` is interpreted as the CUTOFF radius: the taper is exactly 0
    for distances > r_influ. Internally the Gaspari-Cohn half-width is
    ``c = r_influ / 2`` so that the support ``d <= 2c`` equals ``r_influ``.
    """

    # -- Check assumption of num_state=1 is correct if that if no num_state is passed!!
    if num_states is None:
        if cov_prior.shape[0] % state_dim != 0:
            raise ValueError(
                "Num_state was None (assumed to be 1), however "
                f"cov_prior shape {cov_prior.shape} is incompatible with state_dim={state_dim}"
            )
        num_states = cov_prior.shape[0] // state_dim
    # -- Final check covariance matrix has expected shape || expected square atrix with dim: num_states * state_dim
    expected_dim = num_states * state_dim
    if cov_prior.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"Expected covariance shape {(expected_dim, expected_dim)}, "
            f"got {cov_prior.shape}. num_states={num_states}, state_dim={state_dim}."
        )

    # --- Make a matrix that shows distance from diagonal
    idx = jnp.arange(state_dim)
    dist = jnp.abs(idx[:, None] - idx[None, :])

    # --- If periodic, wrap around corners of matrix
    if periodic:
        dist = jnp.minimum(dist, state_dim - dist)

    # --- Create mask with the Gaspari-Cohn 5th-order correlation function.
    # r_influ is the cutoff radius; support is d <= r_influ (== 2c), so c=r_influ/2.
    dist = dist.astype(jnp.float32)
    c = r_influ / 2.0
    z = dist / c

    # Safe z for the 1<z<=2 branch to avoid nan/inf at z=0 poisoning the where.
    z_safe = jnp.where(dist > 0, z, 1.0)

    gc_near = -0.25 * z**5 + 0.5 * z**4 + (5.0 / 8.0) * z**3 - (5.0 / 3.0) * z**2 + 1.0
    gc_far = (
        (1.0 / 12.0) * z**5
        - 0.5 * z**4
        + (5.0 / 8.0) * z**3
        + (5.0 / 3.0) * z**2
        - 5.0 * z
        + 4.0
        - 2.0 / (3.0 * z_safe)
    )

    mask = jnp.where(z <= 1.0, gc_near, jnp.where(z <= 2.0, gc_far, 0.0))
    # Clamp tiny negative round-off to keep entries in [0, 1].
    mask = jnp.clip(mask, 0.0, 1.0)

    # Apply the localization mask to the prior covariance matrix.
    # Each (state_dim x state_dim) block is multiplied by the same mask, so
    # tiling the mask across the full matrix lets us do it in one operation.
    full_mask = jnp.tile(mask, (num_states, num_states))
    cov_prior_loc = cov_prior * full_mask

    return cov_prior_loc
