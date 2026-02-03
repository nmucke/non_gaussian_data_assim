from typing import Callable

import jax
import jax.numpy as jnp


def rollout(
    stepper: Callable,
    num_steps: int,
    return_inner_steps: bool = False,
    include_initial_state: bool = False,
) -> jnp.ndarray:
    """Rollout the system using the given stepper."""

    def scan_fn(x: jnp.ndarray, _: None = None) -> jnp.ndarray:
        """Scan function for the rollout."""
        next_x = stepper(x)
        return next_x, next_x

    def rollout_fn(init: jnp.ndarray) -> jnp.ndarray:
        """Rollout function for the rollout."""
        last_step, trajectory = jax.lax.scan(scan_fn, init, xs=None, length=num_steps)

        if return_inner_steps:
            out = trajectory
        else:
            return last_step

        if include_initial_state:
            out = jnp.concatenate([init[None, ...], trajectory], axis=0)

        return out

    return rollout_fn


class RungeKutta4:
    """Runge-Kutta 4th order time integrator."""

    def __init__(self, dt: float, rhs: Callable):
        """Initialize the Runge-Kutta 4th order time integrator."""
        self.dt = dt
        self.rhs = rhs

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Runge-Kutta 4th order time integration."""
        k1 = self.rhs(x)
        k2 = self.rhs(x + 0.5 * self.dt * k1)
        k3 = self.rhs(x + 0.5 * self.dt * k2)
        k4 = self.rhs(x + self.dt * k3)
        return x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
