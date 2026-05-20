import jax.numpy as jnp
import numpy as np

from non_gaussian_data_assim.forward_models.base import BaseForwardModel


class LinearModel(BaseForwardModel):
    """Deterministic linear forward model: ``x_{n+1} = M @ x_n``.

    The transition matrix ``M`` is fully user-specified, which makes this model
    convenient for analytical Kalman-filter comparisons: the exact same operator
    can be handed to the filter equations. Model (process) error is *not* part of
    the forward model itself -- callers that want a stochastic system should add
    process noise around the deterministic propagation.
    """

    def __init__(
        self,
        transition_matrix: np.ndarray | list,
        model_integration_steps: int = 1,
        dt: float = 1.0,
    ) -> None:
        """Initialize the linear forward model.

        Args:
            transition_matrix: Square matrix ``M`` of shape ``[state_dim, state_dim]``
                applied once per inner integration step.
            model_integration_steps: Number of inner steps per outer (assimilation)
                step. The effective per-outer-step transition is ``M ** model_integration_steps``.
            dt: Nominal time step (unused by the dynamics, kept for API parity).
        """
        M = jnp.asarray(transition_matrix) * 1.0  # promote int -> float without forcing x64
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(
                f"transition_matrix must be square, got shape {tuple(M.shape)}"
            )
        super().__init__(
            dt=dt,
            model_integration_steps=model_integration_steps,
            state_dim=int(M.shape[0]),
            num_states=1,
        )
        self.transition_matrix = M

    def one_step(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply one linear step ``M @ x`` (state is flattened then restored)."""
        return (self.transition_matrix @ x.reshape(-1)).reshape(x.shape)
