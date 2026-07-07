from typing import Tuple

import numpy as np


def rk4_bk_1d_step(
    N: int, theta_in: np.ndarray, u_in: np.ndarray, v_in: np.ndarray, dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a single RK4 step for a one-dimensional system.

    Args:
    N (int): Number of grid points.
    theta_in, u_in, v_in (numpy.array): Input state vectors for theta, u, and v.
    dt (float): Time step for the RK4 update.

    Returns:
    tuple: Updated state vectors (theta_out, u_out, v_out) after one RK4 step.
    """
    # Defining parameters for the equations
    eps = 0.3
    xi = 0.5
    gamma_lambda = np.sqrt(0.2)
    gamma_mu = 0.5
    f = 3.2

    def _slopes(
        theta: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the RHS slopes at the given stage state."""
        # Periodic neighbor shifts recomputed from the current stage state.
        u_p1 = np.zeros(N)
        u_p1[0:-1] = u[1:]
        u_p1[-1] = u[0]

        u_n1 = np.zeros(N)
        u_n1[0] = u[1]
        u_n1[1:] = u[0:-1]

        dthetadt = -(v + 1) * (theta + (1 + eps) * np.log(v + 1))
        dudt = v
        dvdt = (
            (gamma_mu**2) * (u_n1 - 2 * u + u_p1)
            - (gamma_lambda**2) * u
            - ((gamma_mu**2) / (xi)) * (f + theta + np.log(v + 1))
        )
        return dthetadt, dudt, dvdt

    # RK4 stages: each stage is evaluated at the ORIGINAL input state plus the
    # appropriate single previous slope (not accumulated across stages).
    # K1 at x
    k11, k12, k13 = _slopes(theta_in, u_in, v_in)
    # K2 at x + 0.5*dt*k1
    k21, k22, k23 = _slopes(
        theta_in + 0.5 * dt * k11,
        u_in + 0.5 * dt * k12,
        v_in + 0.5 * dt * k13,
    )
    # K3 at x + 0.5*dt*k2
    k31, k32, k33 = _slopes(
        theta_in + 0.5 * dt * k21,
        u_in + 0.5 * dt * k22,
        v_in + 0.5 * dt * k23,
    )
    # K4 at x + dt*k3
    k41, k42, k43 = _slopes(
        theta_in + dt * k31,
        u_in + dt * k32,
        v_in + dt * k33,
    )

    theta_out = theta_in + (dt / 6) * (k11 + 2 * k21 + 2 * k31 + k41)
    u_out = u_in + (dt / 6) * (k12 + 2 * k22 + 2 * k32 + k42)
    v_out = v_in + (dt / 6) * (k13 + 2 * k23 + 2 * k33 + k43)

    return theta_out, u_out, v_out


def rk4_bk_1d_ensemble(
    N: int,
    n_mem: int,
    theta_in: np.ndarray,
    u_in: np.ndarray,
    v_in: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a single RK4 step for an ensemble of one-dimensional systems.

    Args:
    N (int): Number of grid points.
    n_mem (int): Number of ensemble members.
    theta_in, u_in, v_in (numpy.array): Input state matrices for theta, u, and v, each with shape (N, n_mem).
    dt (float): Time step for the RK4 update.

    Returns:
    tuple: Updated state matrices (theta_out, u_out, v_out) after one RK4 step.
    """
    # Defining parameters for the equations
    eps = 0.3
    xi = 0.5
    gamma_lambda = np.sqrt(0.2)
    gamma_mu = 0.5
    f = 3.2

    # Initial conditions

    def _slopes(
        theta: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the RHS slopes at the given ensemble stage state."""
        # Periodic neighbor shifts recomputed from the current stage state.
        u_p1 = np.zeros((N, n_mem))
        u_p1[0:-1, :] = u[1:, :]
        u_p1[-1, :] = u[0, :]

        u_n1 = np.zeros((N, n_mem))
        u_n1[0, :] = u[1, :]
        u_n1[1:, :] = u[0:-1, :]

        dthetadt = -(v + 1) * (theta + (1 + eps) * np.log(v + 1))
        dudt = v
        dvdt = (
            (gamma_mu**2) * (u_n1 - 2 * u + u_p1)
            - (gamma_lambda**2) * u
            - ((gamma_mu**2) / (xi)) * (f + theta + np.log(v + 1))
        )
        return dthetadt, dudt, dvdt

    # RK4 stages: each stage is evaluated at the ORIGINAL input state plus the
    # appropriate single previous slope (not accumulated across stages).
    # K1 at x
    k11, k12, k13 = _slopes(theta_in, u_in, v_in)
    # K2 at x + 0.5*dt*k1
    k21, k22, k23 = _slopes(
        theta_in + 0.5 * dt * k11,
        u_in + 0.5 * dt * k12,
        v_in + 0.5 * dt * k13,
    )
    # K3 at x + 0.5*dt*k2
    k31, k32, k33 = _slopes(
        theta_in + 0.5 * dt * k21,
        u_in + 0.5 * dt * k22,
        v_in + 0.5 * dt * k23,
    )
    # K4 at x + dt*k3
    k41, k42, k43 = _slopes(
        theta_in + dt * k31,
        u_in + dt * k32,
        v_in + dt * k33,
    )

    theta_out = theta_in + (dt / 6) * (k11 + 2 * k21 + 2 * k31 + k41)
    u_out = u_in + (dt / 6) * (k12 + 2 * k22 + 2 * k32 + k42)
    v_out = v_in + (dt / 6) * (k13 + 2 * k23 + 2 * k33 + k43)

    return theta_out, u_out, v_out
