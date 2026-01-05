import numpy as np

from non_gaussian_data_assim.forward_models.base import BaseForwardModel

class KuramotoSivashinsky(BaseForwardModel):
    """Kuramoto Sivashinsky forward model."""

    def __init__(
        self, 
        dt: float, 
        num_model_steps: int, 
        state_dim: int, 
        domain_length: float,
        nu: float = 1.0,
    ):
        """Initialize the Kuramoto Sivashinsky forward model.
        
        Args:
            dt: Time step for integration
            num_model_steps: Number of inner time steps per call
            state_dim: Number of spatial grid points
            domain_length: Length of the periodic domain
        """
        super().__init__(dt, num_model_steps, state_dim)

        self.nu = nu
        self.L = domain_length
        self.state_dim = state_dim
        self.domain_length = domain_length
        
        # Set up wave numbers for spectral method
        # For periodic domain of length L with N points, wave numbers are k = 2π*n/L
        # where n goes from -N/2 to N/2-1 (handled by fftfreq)
        # fftfreq(n) returns frequencies in units of 1/n, so we need to scale by n/L
        self.k = 2 * np.pi * np.fft.fftfreq(state_dim) * state_dim / domain_length
        
        # Precompute linear operator in Fourier space
        # KS equation: u_t + u*u_x + u_xx + nu*u_xxxx = 0
        # Rearranged: u_t = -u*u_x - u_xx - nu*u_xxxx
        # In Fourier space: u_xx -> -k^2*u_hat, u_xxxx -> k^4*u_hat
        # So: -u_xx -> k^2*u_hat, -nu*u_xxxx -> -nu*k^4*u_hat
        # Linear operator: k^2 - nu*k^4
        self.linear_operator = self.k**2 - self.nu * self.k**4

    def _one_model_step(self, u: np.ndarray) -> np.ndarray:
        """One inner step of the forward model using ETDRK4 scheme.
        
        The Kuramoto-Sivashinsky equation is:
        ∂u/∂t + u*∂u/∂x + ∂²u/∂x² + ν*∂⁴u/∂x⁴ = 0
        
        We solve this using a spectral method with ETDRK4 time stepping.
        
        Args:
            u: Current state vector (1D array of size state_dim)
            
        Returns:
            Updated state vector after one time step
        """
        # Handle ensemble input format: (ensemble_size, num_states, state_dim)
        if u.ndim == 3:
            ensemble_size, num_states, state_dim = u.shape
            u_flat = u.reshape(ensemble_size, state_dim)
            result = np.zeros_like(u_flat)
            
            for i in range(ensemble_size):
                result[i] = self._etdrk4_step(u_flat[i])
            
            return result.reshape(ensemble_size, num_states, state_dim)
        else:
            # Single state vector
            return self._etdrk4_step(u)
    
    def _etdrk4_step(self, u: np.ndarray) -> np.ndarray:
        """Perform one ETDRK4 (Exponential Time Differencing RK4) step.
        
        Standard ETDRK4 scheme for u_t = L*u + N(u):
        a = E2*u_hat + dt*phi_1(L*dt/2)*N1_hat
        b = E2*u_hat + dt*phi_1(L*dt/2)*N2_hat  
        c = E2*a + dt*phi_1(L*dt/2)*(2*N3_hat - N1_hat)
        u_new = E*u_hat + dt*phi_1(L*dt)*N1_hat + 2*dt*(phi_2*(N2_hat+N3_hat) - phi_3*(N1_hat+N4_hat))
        
        Args:
            u: Current state vector
            
        Returns:
            Updated state vector
        """
        dt = self.dt
        L = self.linear_operator
        
        # Transform to Fourier space
        u_hat = np.fft.fft(u)
        
        # ETDRK4 coefficients
        E = np.exp(dt * L)
        E2 = np.exp(dt * L / 2)
        
        # Compute phi functions for ETDRK4
        # phi_1(z) = (exp(z) - 1) / z
        # phi_2(z) = (exp(z) - 1 - z) / z^2
        # phi_3(z) = (exp(z) - 1 - z - z^2/2) / z^3
        
        z = dt * L
        z_half = dt * L / 2
        
        # Avoid division by zero for k=0 mode
        z_safe = np.where(np.abs(z) > 1e-10, z, 1.0)
        z_half_safe = np.where(np.abs(z_half) > 1e-10, z_half, 1.0)
        
        # phi_1 for full step
        phi_1_full = np.where(np.abs(z) > 1e-10, (E - 1) / z_safe, 1.0)
        # phi_1 for half step
        phi_1_half = np.where(np.abs(z_half) > 1e-10, (E2 - 1) / z_half_safe, 1.0)
        # phi_2 for full step
        phi_2_full = np.where(np.abs(z) > 1e-10, (E - 1 - z) / (z_safe**2), 0.5)
        # phi_3 for full step
        phi_3_full = np.where(
            np.abs(z) > 1e-10, (E - 1 - z - z**2 / 2) / (z_safe**3), 1.0 / 6
        )
        
        # Stage 1: N1 = N(u(t))
        N1 = self._nonlinear_term(u)
        N1_hat = np.fft.fft(N1)
        
        # Stage 2: a = u(t + dt/2) using N1
        a_hat = E2 * u_hat + dt * phi_1_half * N1_hat
        a = np.fft.ifft(a_hat).real
        N2 = self._nonlinear_term(a)
        N2_hat = np.fft.fft(N2)
        
        # Stage 3: b = u(t + dt/2) using N2
        b_hat = E2 * u_hat + dt * phi_1_half * N2_hat
        b = np.fft.ifft(b_hat).real
        N3 = self._nonlinear_term(b)
        N3_hat = np.fft.fft(N3)
        
        # Stage 4: c = E2*a + correction using N3
        c_hat = E2 * a_hat + dt * phi_1_half * (2 * N3_hat - N1_hat)
        c = np.fft.ifft(c_hat).real
        N4 = self._nonlinear_term(c)
        N4_hat = np.fft.fft(N4)
        
        # Combine stages using ETDRK4 formula
        u_new_hat = (
            E * u_hat
            + dt * phi_1_full * N1_hat
            + 2 * dt * (phi_2_full * (N2_hat + N3_hat) - phi_3_full * (N1_hat + N4_hat))
        )
        
        # Transform back to physical space
        u_new = np.fft.ifft(u_new_hat).real
        
        return u_new
    
    def _nonlinear_term(self, u: np.ndarray) -> np.ndarray:
        """Compute the nonlinear term -u * ∂u/∂x.
        
        Args:
            u: State vector in physical space
            
        Returns:
            Nonlinear term in physical space
        """
        # Compute derivative in Fourier space
        u_hat = np.fft.fft(u)
        ux_hat = 1j * self.k * u_hat
        ux = np.fft.ifft(ux_hat).real
        
        # Nonlinear term: -u * ux
        return -u * ux


