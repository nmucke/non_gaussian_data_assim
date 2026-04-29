import jax.numpy as jnp

from non_gaussian_data_assim.forward_models.base import BaseForwardModel


class CoupledKuramotoSivashinsky(BaseForwardModel):
    """Coupled Kuramoto Sivashinsky forward model."""

    def __init__(
        self,
        dt: float,
        inner_steps: int,
        state_dim: int,
        domain_length: float,
        alpha: float = 0.003,
        lambda_a: float = 1.0,
        lambda_o: float = 1.0,
        tau_a: float = 1.0,
        tau_o: float = 1.0
    ):
        """Initialize the Kuramoto Sivashinsky forward model.

        Args:
            dt: Time step for integration
            num_model_steps: Number of inner time steps per call
            state_dim: Number of spatial grid points
            domain_length: Length of the periodic domain
        """
        super().__init__(dt, inner_steps, state_dim, num_states=2)

        self.alpha = alpha
        self.lambda_a = lambda_a
        self.lambda_o = lambda_o
        self.tau_a = tau_a
        self.tau_o = tau_o

        self.L = domain_length
        self.state_dim = state_dim
        self.domain_length = domain_length

        wavenumbers = jnp.fft.rfftfreq(
            state_dim, d=domain_length / (state_dim * 2 * jnp.pi)
        )
        self.derivative_operator = 1j * wavenumbers

        linear_operator = -self.derivative_operator**2 * self.lambda_a**2 - self.derivative_operator**4  * self.lambda_a**4
        self.exp_term = jnp.exp(dt * linear_operator)
        self.coef = jnp.where(
            linear_operator == 0.0,
            dt,
            (self.exp_term - 1.0) / linear_operator,
        )

        self.alias_mask = wavenumbers < 2 / 3 * jnp.max(wavenumbers)

        self.nonlin_operator = self.coef * self.derivative_operator * self.alias_mask * self.lambda_a

    def _apply_linear_operator(self, state_hat: jnp.ndarray) -> jnp.ndarray:
        """linear operator: (- d_dx^2 - d_dx^4) state"""
        return self.exp_term * state_hat

    def _apply_nonlinear_operator(self, state: jnp.ndarray) -> jnp.ndarray:
        state_nonlin_hat = jnp.fft.rfft(-0.5 * state * state, axis=-1)
        return self.nonlin_operator * state_nonlin_hat

    def one_step(self, x: jnp.ndarray) -> jnp.ndarray:
        """Coupled Kuramoto Sivashinsky one step."""

        x_hat = jnp.fft.rfft(x, axis=-1)
        x_next_hat = self._apply_linear_operator(x_hat) + self._apply_nonlinear_operator(x)

        delta = self.alpha * (x_hat[1] - x_hat[0])
        x_next_hat = x_next_hat + jnp.stack([delta, -delta])

        return jnp.fft.irfft(x_next_hat, n=self.state_dim, axis=-1)
