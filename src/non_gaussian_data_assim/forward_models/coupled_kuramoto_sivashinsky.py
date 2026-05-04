import jax.numpy as jnp

from non_gaussian_data_assim.forward_models.base import BaseForwardModel


class CoupledKuramotoSivashinsky(BaseForwardModel):
    """Coupled multiscale Kuramoto-Sivashinsky model from Vossepoel et al. (2025).

    Solves
        tau_a dA/dt = -lambda_a/2 d(A^2)/dx - lambda_a^2 d^2A/dx^2
                      - lambda_a^4/2 d^4A/dx^4 + alpha_oa (O - A)
        tau_o dO/dt = -lambda_o/2 d(O^2)/dx - lambda_o^2 d^2O/dx^2
                      - lambda_o^4 d^4O/dx^4 + omega_ao (A - O)
    using ETDRK1 in Fourier space with 2/3 dealiasing.
    The Atmos biharmonic damping is halved per Eq. (1) of the paper.
    """

    def __init__(
        self,
        dt: float,
        model_integration_steps: int,
        state_dim: int,
        domain_length: float,
        alpha_oa: float = 0.003,
        omega_ao: float = 0.003,
        lambda_a: float = 32.0,
        lambda_o: float = 4.0,
        tau_a: float = 1.0,
        tau_o: float = 4.0,
    ):
        super().__init__(dt, model_integration_steps, state_dim, num_states=2)

        self.alpha_oa = alpha_oa
        self.omega_ao = omega_ao
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
        deriv = 1j * wavenumbers
        self.derivative_operator = deriv
        self.alias_mask = wavenumbers < 2 / 3 * jnp.max(wavenumbers)

        # Per-state linear operator (already divided by tau).
        # Atmos has the halved biharmonic; Ocean does not.
        linear_a = (-(deriv**2) * lambda_a**2 - 0.5 * (deriv**4) * lambda_a**4) / tau_a
        linear_o = (-(deriv**2) * lambda_o**2 - (deriv**4) * lambda_o**4) / tau_o
        linear_op = jnp.stack([linear_a, linear_o])

        self.exp_term = jnp.exp(dt * linear_op)
        self.coef = jnp.where(
            linear_op == 0.0,
            dt,
            (self.exp_term - 1.0) / linear_op,
        )

        # Per-state Burgers coefficient. Multiplies fft(-0.5 * state^2) to give the
        # Fourier representation of -lambda/(2 tau) * d(state^2)/dx, dealiased.
        nonlin_a = (lambda_a / tau_a) * deriv * self.alias_mask
        nonlin_o = (lambda_o / tau_o) * deriv * self.alias_mask
        self.nonlin_coef = jnp.stack([nonlin_a, nonlin_o])

        # Coupling forcing scalars: alpha_oa/tau_a applied to (O - A) for A and
        # omega_ao/tau_o applied to (A - O) for O.
        self.coupling_scale = jnp.array([alpha_oa / tau_a, -omega_ao / tau_o])

    def one_step(self, x: jnp.ndarray) -> jnp.ndarray:
        """One ETDRK1 step of the coupled KS system. `x` has shape (2, state_dim)."""
        x_hat = jnp.fft.rfft(x, axis=-1)

        nonlin_hat = self.nonlin_coef * jnp.fft.rfft(-0.5 * x * x, axis=-1)

        # Coupling enters as a forcing; both states share the (O - A) Fourier term
        # up to a scalar (alpha/tau_a for A, -alpha/tau_o for O).
        delta = x_hat[1] - x_hat[0]
        coupling_hat = self.coupling_scale[:, None] * delta

        x_next_hat = self.exp_term * x_hat + self.coef * (nonlin_hat + coupling_hat)
        return jnp.fft.irfft(x_next_hat, n=self.state_dim, axis=-1)
