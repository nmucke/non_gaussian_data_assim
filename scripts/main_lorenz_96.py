import pdb
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from non_gaussian_data_assim.da_methods.agmf import AdaptiveGaussianMixtureFilter
from non_gaussian_data_assim.da_methods.base import da_rollout
from non_gaussian_data_assim.da_methods.enkf import EnsembleKalmanFilter
from non_gaussian_data_assim.da_methods.pff import ParticleFlowFilter
from non_gaussian_data_assim.forward_models.lorenz_96 import Lorenz96Model
from non_gaussian_data_assim.observation_operator import ObservationOperator

SEED = 42

# Constants and parameters
DT = 0.01
F = 8.0
T0 = 0.0

OUTER_STEPS = 150
INNER_STEPS = 4


NUM_STATES = 1
STATE_DIM = 25
NUM_SKIP = 3
ENSEMBLE_SIZE = 1000

# Observation ids
OBS_IDS = np.arange(0, STATE_DIM, NUM_SKIP)
OBS_STATES = (0,)

# Observation error covariance matrix
R = jnp.eye(len(OBS_IDS)) * 0.1


def main() -> None:
    """Main function."""
    rng_key = jax.random.PRNGKey(SEED)

    # Initial state - shape: [1, num_states, state_dim] for single ensemble member
    rng_key, key = jax.random.split(rng_key)
    X_0 = jax.random.normal(key, (1, NUM_STATES, STATE_DIM)) * 10
    X_0 = X_0.at[0, 0, -1].set(X_0[0, 0, 0])  # Periodic boundary condition

    # Define the forward model
    forward_model = Lorenz96Model(
        forcing_term=F, state_dim=STATE_DIM, dt=DT, inner_steps=INNER_STEPS
    )

    # Define the observation operator
    obs_operator = ObservationOperator(
        obs_states=OBS_STATES, obs_indices=OBS_IDS, state_dim=STATE_DIM
    )

    true_sol = forward_model.rollout(X_0, OUTER_STEPS - 1)

    # Generate observations
    observations = jnp.zeros((OUTER_STEPS, len(OBS_IDS)))
    for i in range(OUTER_STEPS):
        obs_at_t = obs_operator(true_sol[:, i])  # [1, num_obs]
        observations = observations.at[i].set(obs_at_t.flatten())  # [num_obs]

    # Instantiate the data assimilation model
    da_model = AdaptiveGaussianMixtureFilter(
        ensemble_size=ENSEMBLE_SIZE,
        R=R,
        obs_operator=obs_operator,
        forward_operator=forward_model,
        inflation_factor=1.0,
        localization_distance=5,
        w_prev=np.ones(ENSEMBLE_SIZE) / ENSEMBLE_SIZE,
        nc_threshold=0.5,
    )
    # da_model = ParticleFlowFilter(
    #     ensemble_size=ENSEMBLE_SIZE,
    #     R=R,
    #     obs_operator=obs_operator,
    #     forward_operator=forward_model,
    #     localization_distance=5,
    # )

    # Initialize the prior ensemble
    rng_key, key = jax.random.split(rng_key)
    prior_ensemble = jax.random.normal(key, (ENSEMBLE_SIZE, NUM_STATES, STATE_DIM)) * 10
    prior_ensemble = prior_ensemble.at[:, :, -1].set(
        prior_ensemble[:, :, 0]
    )  # Periodic boundary condition

    # Initialize the posterior ensemble
    posterior_ensemble = prior_ensemble.copy()
    posterior_ensemble = posterior_ensemble.reshape(
        ENSEMBLE_SIZE, NUM_STATES, STATE_DIM
    )

    # Rollout the prior ensemble
    prior_ensemble = forward_model.rollout(prior_ensemble, OUTER_STEPS - 1)

    # Perform the data assimilation
    rng_key, key = jax.random.split(rng_key)
    posterior_ensemble = da_model.rollout(posterior_ensemble, observations[1:], rng_key)

    # Calculate the prior and posterior errors
    true_sol = true_sol.reshape(OUTER_STEPS, STATE_DIM)

    prior_error = true_sol - prior_ensemble.mean(axis=(0, 2))
    prior_error = np.sqrt(np.sum(prior_error**2))
    posterior_error = true_sol - posterior_ensemble.mean(axis=(0, 2))
    posterior_error = np.sqrt(np.sum(posterior_error**2))

    print(f"Prior error: {prior_error}")
    print(f"Posterior error: {posterior_error}")

    idx_to_plot = 17

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(true_sol)
    plt.colorbar()
    plt.title("True Solution")
    plt.subplot(2, 3, 2)
    plt.imshow(prior_ensemble.mean(axis=(0, 2)))
    plt.colorbar()
    plt.title("Prior Ensemble Mean")
    plt.subplot(2, 3, 3)
    plt.imshow(posterior_ensemble.mean(axis=(0, 2)))
    plt.colorbar()
    plt.title("Posterior Ensemble Mean")
    plt.subplot(2, 3, 4)
    plt.imshow(true_sol - prior_ensemble.mean(axis=(0, 2)))
    plt.colorbar()
    plt.title("|True - Prior| difference")
    plt.subplot(2, 3, 5)
    plt.imshow(true_sol - posterior_ensemble.mean(axis=(0, 2)))
    plt.colorbar()
    plt.title("|True - Posterior| difference")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.subplot(2, 3, 6)
    plt.plot(
        prior_ensemble.mean(axis=(0, 2))[:, idx_to_plot],
        label="Prior",
        color="tab:red",
        linewidth=3,
    )
    plt.plot(
        posterior_ensemble.mean(axis=(0, 2))[:, idx_to_plot],
        label="Posterior",
        color="tab:blue",
        linewidth=3,
    )
    # Shade the standard deviation of the posterior on the time series plot
    mean_post = posterior_ensemble.mean(axis=(0, 2))[:, idx_to_plot]
    std_post = posterior_ensemble.std(axis=(0, 2))[:, idx_to_plot]
    time_axis = np.arange(posterior_ensemble.shape[1])
    plt.fill_between(
        time_axis,
        mean_post - std_post,
        mean_post + std_post,
        color="tab:blue",
        alpha=0.2,
        label="Posterior ± Std",
    )

    plt.plot(
        true_sol[:, idx_to_plot],
        label="True",
        color="black",
        linewidth=3,
        linestyle="--",
    )
    plt.xlabel("Time")
    plt.ylabel("State 25")
    plt.ylim(-10, 10)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
