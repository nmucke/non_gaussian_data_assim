import jax
import jax.numpy as jnp
import pdb

# def compute_pairwise_interaction(x_s, pair_function):
#     """
#     Args:
#         x_s: Shape [dof, ensemble]
#         pair_function: Function taking (vec_a, vec_b) -> (k, dk)
#                        where vec_a/b are shape [dof]
#     Returns:
#         kernel: Shape [dof, ensemble, ensemble]
#         dkdx: Shape [dof, ensemble, ensemble]
#     """
    
#     # 1. Transpose input to [ensemble, dof] for easier vmapping
#     # JAX vmap defaults to iterating over dimension 0
#     X = x_s.T  # Shape: [N, dof]

#     # 2. Create the broadcasted function using double vmap
#     # First vmap: Iterate over the FIRST argument (rows of A)
#     # pair_function(a, b) -> (k, dk)
#     # vmapped(A, b) -> (K_col, dK_col)
#     # We use out_axes=1 to stack the results along the second dimension 
#     # to preserve the (dof, N) structure inside intermediate steps.
#     vmap_func = jax.vmap(
#         jax.vmap(pair_function, in_axes=(None, 0), out_axes=(1, 1)), 
#         in_axes=(0, None), 
#         out_axes=(2, 2)
#     )

#     # 3. Compute everything in one shot
#     # This computes the function for all N*N pairs
#     kernel, dkdx = vmap_func(X, X)
    
#     # The output shapes from vmap setup above are:
#     # kernel: [dof, N, N]
#     # dkdx:   [dof, N, N]
    
#     # 4. Enforce symmetry/anti-symmetry (Optional but recommended)
#     # Even if the function is mathematically symmetric, floating point errors 
#     # might make K_ij != K_ji slightly.
    
#     # Symmetrize Kernel: K_ij = K_ji
#     kernel = (kernel + jnp.swapaxes(kernel, 1, 2)) / 2.0
    
#     # Anti-Symmetrize Gradients: D_ij = -D_ji (if that is your requirement)
#     # Based on your code: dkdx[j, i] = -dkdx[i, j]
#     dkdx = (dkdx - jnp.swapaxes(dkdx, 1, 2)) / 2.0

#     return kernel, dkdx


def compute_pairwise_interaction(x_s, pair_function):
    dofs, ensemble_size = x_s.shape
    X = x_s.T

    # Inner vmap (iterates j): returns [dof, ensemble] (j is dim 1)
    # Outer vmap (iterates i): stacks i at dim 1 -> [dof, ensemble, ensemble]
    vmap_func = jax.vmap(
        jax.vmap(pair_function, in_axes=(None, 0), out_axes=(1, 1)), 
        in_axes=(0, None), 
        out_axes=(1, 1) 
    )

    # Compute full grid
    # Shape is now correctly: [dof, i, j]
    k_raw, dk_raw = vmap_func(X, X)

    # We need to replicate: "Compute Upper, Copy to Lower"
    
    # Mask for strictly upper triangle (j > i)
    mask_strict_upper = jnp.triu(jnp.ones((ensemble_size, ensemble_size), dtype=bool), k=1)[None, :, :]

    # Mask for diagonal (j == i)
    mask_diag = jnp.eye(ensemble_size, dtype=bool)[None, :, :]

    # CONSTRUCT KERNEL
    # Take strictly upper part
    k_upper = jnp.where(mask_strict_upper, k_raw, 0.0)
    # Take diagonal part (from raw computation, matching your loop's 'if j!=i' check)
    k_diag = jnp.where(mask_diag, k_raw, 0.0)
    # Create lower part by transposing upper
    k_lower = jnp.swapaxes(k_upper, 1, 2)
    
    kernel = k_upper + k_lower + k_diag

    # 4. CONSTRUCT GRADIENT (Antisymmetric)
    # Take strictly upper part
    dk_upper = jnp.where(mask_strict_upper, dk_raw, 0.0)
    # Take diagonal part
    dk_diag = jnp.where(mask_diag, dk_raw, 0.0)
    # Create lower part: Transpose AND Negate (matching dkdx[j,i] = -dkdx[i,j])
    dk_lower = -jnp.swapaxes(dk_upper, 1, 2)
    
    dkdx = dk_upper + dk_lower + dk_diag

    return kernel, dkdx
# --- Example usage ---
# Define a dummy function representing your logic
def my_function(x, y):
    # Example: Gaussian RBF kernel logic
    diff = x - y
    dist_sq = jnp.sum(diff**2)
    k = jnp.exp(-dist_sq) * x # Dummy return shape [dof]
    dk = diff                 # Dummy return shape [dof]
    return k, dk

# Setup data
dofs = 3
ensemble = 10
x = jax.random.normal(jax.random.PRNGKey(0), (dofs, ensemble))

# Run
kernel, dkdx = compute_pairwise_interaction(x, my_function)

_kernel = jnp.zeros((dofs, ensemble, ensemble))
_dkdx = jnp.zeros((dofs, ensemble, ensemble))
for i in range(ensemble):
    for j in range(i, ensemble):
        kk, dk = my_function(x[:, i], x[:, j])
        _kernel = _kernel.at[:, i, j].set(kk)
        _dkdx = _dkdx.at[:, i, j].set(dk)
        if j != i:
            _kernel = _kernel.at[:, j, i].set(_kernel[:, i, j])
            _dkdx = _dkdx.at[:, j, i].set(-_dkdx[:, i, j])

print("Kernel shape:", kernel.shape)  # (3, 10, 10)
print("dkdx shape:  ", dkdx.shape)    # (3, 10, 10)

vectorized_error = jnp.abs(kernel - _kernel).max()
print(f"Vectorized error: {vectorized_error}")

# Compute the pair (i, j, d) where the absolute difference between kernel and _kernel is maximized
abs_diff = jnp.abs(kernel - _kernel)
max_idx_flat = jnp.argmax(abs_diff)
dof_idx, i_idx, j_idx = jnp.unravel_index(max_idx_flat, abs_diff.shape)
max_value = abs_diff[dof_idx, i_idx, j_idx]
print(f"Max diff at (dof={dof_idx}, i={i_idx}, j={j_idx}) = {max_value}")


pdb.set_trace()