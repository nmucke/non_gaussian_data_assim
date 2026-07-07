"""Minimal data-assimilation demo.

This is the smallest possible example of running the assimilation pipeline. The
*only* library class used is the DA method itself (``EnsembleKalmanFilter``).
The forward model, the observation operator, the truth, the observations and the
initial ensemble are all defined right here in the script.

The point is to show exactly what interface a forward model and an observation
operator must expose so that the EnKF can use them:

    forward model  ->  .num_states, .state_dim
                       __call__(x, return_model_integration_steps, is_ensemble)
                           x:   [ensemble, num_states, state_dim]
                           out: [ensemble, time, num_states, state_dim]

    obs operator   ->  .num_obs      (number of scalar observations)
                       .obs_matrix   [num_obs, num_states * state_dim]

Run with:
    python scripts/main_simple.py
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from non_gaussian_data_assim.da_methods.enkf import EnsembleKalmanFilter
from non_gaussian_data_assim.forward_models.base import BaseForwardModel

# ============================= configuration =============================
SEED = 0
STATE_DIM = 40  # number of grid points
NUM_STATES = 1
ENSEMBLE_SIZE = 50
DA_STEPS = 60  # number of assimilation cycles
MODEL_STEPS = 5  # inner integration steps per assimilation cycle
DT = 0.05  # inner time step
SPEED = 1.0  # advection speed
DX = 1.0  # grid spacing
OBS_EVERY = 2  # observe every OBS_EVERY-th grid point
OBS_NOISE_STD = 0.1  # observation noise standard deviation
INFLATION = 1.02  # multiplicative covariance inflation
# =========================================================================


class SimpleAdvectionModel(BaseForwardModel):
    """An extremely simple forward model: 1-D linear advection on a periodic
    grid, solved with first-order upwind differencing and explicit Euler in
    time (``u_t + c u_x = 0``).

    A field defined on the grid is simply transported to the right at speed
    ``SPEED``. This is enough to exercise the whole DA machinery.
    """

    def __init__(self, state_dim, num_states, dt, speed, dx, model_integration_steps):
        super().__init__(dt, model_integration_steps, state_dim, num_states=1)
        self.state_dim = state_dim
        self.num_states = num_states
        self.dt = dt
        self.speed = speed
        self.dx = dx
        self.model_integration_steps = model_integration_steps

    def one_step(self, u):
        """Advance the field ``u`` ([num_states, state_dim]) by one inner step."""
        u_upwind = jnp.roll(u, 1, axis=-1)  # u[i-1], periodic boundary
        return u - self.speed * self.dt / self.dx * (u - u_upwind)


class SimpleObservationOperator:
    """An extremely simple (linear) observation operator: it samples the field
    at every ``obs_every``-th grid point. The EnKF only needs the resulting
    observation matrix and the number of observations.
    """

    def __init__(self, state_dim, num_states, obs_every):
        self.state_dim = state_dim
        self.num_states = num_states

        dofs = num_states * state_dim
        obs_indices = jnp.arange(0, state_dim, obs_every)
        self.num_obs = int(obs_indices.shape[0])

        # obs_matrix[k, obs_indices[k]] = 1  -> picks out the observed cells.
        obs_matrix = jnp.zeros((self.num_obs, dofs))
        obs_matrix = obs_matrix.at[jnp.arange(self.num_obs), obs_indices].set(1.0)
        self.obs_matrix = obs_matrix

    def __call__(self, ensemble):
        """Map states to observations. ensemble: [ensemble, num_states, state_dim]."""
        flat = ensemble.reshape(ensemble.shape[0], -1)
        return flat @ self.obs_matrix.T  # [ensemble, num_obs]


def rollout_truth(model, initial_state, da_steps):
    """Generate the ground-truth trajectory, one state per assimilation cycle."""
    states = [initial_state]
    u = initial_state[None]  # add ensemble axis of size 1
    for _ in range(da_steps):
        trajectory = model(u, return_model_integration_steps=True, is_ensemble=True)
        u = trajectory[:, -1]
        states.append(u[0])
    return jnp.stack(states)  # [da_steps + 1, num_states, state_dim]


def main():
    rng_key = jax.random.PRNGKey(SEED)
    rng_key, ensemble_key, obs_key = jax.random.split(rng_key, 3)

    # ----- forward model and observation operator (defined above) -----
    forward_model = SimpleAdvectionModel(
        state_dim=STATE_DIM,
        num_states=NUM_STATES,
        dt=DT,
        speed=SPEED,
        dx=DX,
        model_integration_steps=MODEL_STEPS,
    )
    obs_operator = SimpleObservationOperator(
        state_dim=STATE_DIM,
        num_states=NUM_STATES,
        obs_every=OBS_EVERY,
    )
    R = jnp.eye(obs_operator.num_obs) * OBS_NOISE_STD**2

    # ----- truth: a Gaussian bump that gets advected around the grid -----
    grid = jnp.arange(STATE_DIM) * DX
    true_initial_state = jnp.exp(
        -((grid - STATE_DIM * DX / 2.0) ** 2) / (2.0 * 4.0**2)
    ).reshape(NUM_STATES, STATE_DIM)
    true_states = rollout_truth(forward_model, true_initial_state, DA_STEPS)

    # ----- observations: observe the truth, add noise -----
    # observations[i] is the (noisy) observation of the truth at cycle i + 1,
    # i.e. the time the forecast launched at cycle i lands on.
    clean_obs = (true_states[1:].reshape(DA_STEPS, -1)) @ obs_operator.obs_matrix.T
    noise = OBS_NOISE_STD * jax.random.normal(obs_key, clean_obs.shape)
    observations = clean_obs + noise  # [DA_STEPS, num_obs]

    # ----- initial ensemble: a wrong guess (shifted bump) plus noise -----
    wrong_guess = jnp.roll(true_initial_state, 6, axis=-1)  # offset start
    perturbations = 0.3 * jax.random.normal(
        ensemble_key, (ENSEMBLE_SIZE, NUM_STATES, STATE_DIM)
    )
    initial_ensemble = wrong_guess[None] + perturbations

    # ----- the one library class we use: the EnKF -----
    da_model = EnsembleKalmanFilter(
        ensemble_size=ENSEMBLE_SIZE,
        R=R,
        obs_operator=obs_operator,
        forward_operator=forward_model,
        inflation_factor=INFLATION,
    )

    # ----- run the assimilation loop -----
    posterior = initial_ensemble.reshape(ENSEMBLE_SIZE, 1, NUM_STATES, STATE_DIM)
    for i in range(DA_STEPS):
        rng_key, key = jax.random.split(rng_key)
        prior_current = posterior[:, -1]
        posterior_next = da_model(
            prior_ensemble=prior_current,
            obs_vect=observations[i],
            rng_key=key,
            return_model_integration_steps=True,
        )
        posterior = jnp.concatenate([posterior, posterior_next], axis=1)

    # ----- compare posterior (at cycle boundaries) against the truth -----
    # posterior built as: initial state + DA_STEPS blocks of MODEL_STEPS each,
    # so the analysis states sit at indices 0, MODEL_STEPS, 2*MODEL_STEPS, ...
    analysis = posterior[
        :, ::MODEL_STEPS
    ]  # [ensemble, DA_STEPS + 1, num_states, state_dim]
    analysis_mean = analysis.mean(axis=0)  # [DA_STEPS + 1, num_states, state_dim]

    rmse_per_step = jnp.sqrt(jnp.mean((analysis_mean - true_states) ** 2, axis=(1, 2)))
    print(f"Initial RMSE: {rmse_per_step[0]:.4f}")
    print(f"Final RMSE:   {rmse_per_step[-1]:.4f}")

    # ----- plots -----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(rmse_per_step)
    axes[0].set_title("Ensemble-mean RMSE")
    axes[0].set_xlabel("Assimilation cycle")
    axes[0].set_ylabel("RMSE")

    im1 = axes[1].imshow(true_states[:, 0, :].T, aspect="auto", origin="lower")
    axes[1].set_title("Truth")
    axes[1].set_xlabel("Assimilation cycle")
    axes[1].set_ylabel("Grid point")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(analysis_mean[:, 0, :].T, aspect="auto", origin="lower")
    axes[2].set_title("Posterior mean")
    axes[2].set_xlabel("Assimilation cycle")
    axes[2].set_ylabel("Grid point")
    fig.colorbar(im2, ax=axes[2])

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
