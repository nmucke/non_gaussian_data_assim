import pdb

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from non_gaussian_data_assim.da_methods.agmf import AdaptiveGaussianMixtureFilter
from non_gaussian_data_assim.da_methods.enkf import EnsembleKalmanFilter
from non_gaussian_data_assim.da_methods.pff import ParticleFlowFilter
from non_gaussian_data_assim.forward_models.lorenz_96 import Lorenz96Model
from non_gaussian_data_assim.observation_operator import ObservationOperator

np.random.seed(42)

# Constants and parameters
DT = 0.01
F = 8.0
NUM_TIME_STEPS = 1000
NUM_STATES = 1
STATE_DIM = 25
NUM_SKIP = 3
ENSEMBLE_SIZE = 100

# Initial state
X_0 = np.random.randn(NUM_STATES, STATE_DIM) * 10
X_0[-1] = X_0[0]  # Periodic boundary condition

# Observation ids
OBS_IDS = np.arange(0, STATE_DIM, NUM_SKIP)
OBS_STATES = (0,)

# Observation error covariance matrix
R = np.eye(len(OBS_IDS)) * 0.1


def main() -> None:
    """Main function."""

    # Define the forward model
    forward_model = Lorenz96Model(
        forcing_term=F, state_dim=STATE_DIM, dt=DT, num_model_steps=1
    )

    # Define the observation operator
    obs_operator = ObservationOperator(
        obs_states=OBS_STATES, obs_indices=OBS_IDS, state_dim=STATE_DIM
    )

    # Initialize the true solution and observations
    observations = np.zeros((len(OBS_IDS), NUM_TIME_STEPS))
    true_sol = np.zeros((1, NUM_STATES, STATE_DIM, NUM_TIME_STEPS))
    true_sol[0, ..., 0] = X_0

    # Perform the forward model
    for i in range(1, NUM_TIME_STEPS):
        true_sol[..., i] = forward_model(true_sol[..., i - 1])
        observations[:, i] = obs_operator(true_sol[..., i])

    # Define the data assimilation model
    # da_model = EnsembleKalmanFilter(
    #     ensemble_size=ENSEMBLE_SIZE,
    #     R=R,
    #     obs_operator=obs_operator,
    #     forward_operator=forward_model,
    #     inflation_factor=1.0,
    #     localization_distance=5,
    #     # w_prev=np.ones(ENSEMBLE_SIZE) / ENSEMBLE_SIZE,
    #     # nc_threshold=0.5,
    # )
    da_model = ParticleFlowFilter(
        ensemble_size=ENSEMBLE_SIZE,
        R=R,
        obs_operator=obs_operator,
        forward_operator=forward_model,
        localization_distance=5,
    )

    # Initialize the prior ensemble
    prior_ensemble = np.zeros((ENSEMBLE_SIZE, NUM_STATES, STATE_DIM, NUM_TIME_STEPS))
    prior_ensemble[..., 0] = np.random.randn(ENSEMBLE_SIZE, NUM_STATES, STATE_DIM) * 10
    prior_ensemble[:, :, -1] = prior_ensemble[:, :, 0]

    # Initialize the posterior ensemble
    posterior_ensemble = prior_ensemble.copy()

    # Perform the data assimilation
    for i in tqdm(range(1, NUM_TIME_STEPS)):
        prior_ensemble[..., i] = forward_model(prior_ensemble[..., i - 1])
        posterior_ensemble[..., i] = da_model(
            prior_ensemble=posterior_ensemble[..., i - 1], obs_vect=observations[:, i]
        )

    # Calculate the prior and posterior errors
    prior_error = true_sol - prior_ensemble.mean(axis=(0, 1))
    prior_error = np.sqrt(np.sum(prior_error**2, axis=(0, 1)))
    posterior_error = true_sol - posterior_ensemble.mean(axis=(0, 1))
    posterior_error = np.sqrt(np.sum(posterior_error**2, axis=(0, 1)))

    print(f"Prior error: {prior_error.mean()}")
    print(f"Posterior error: {posterior_error.mean()}")

    true_sol = true_sol[0, 0]

    idx_to_plot = 17

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(true_sol)
    plt.colorbar()
    plt.title("True Solution")
    plt.subplot(2, 3, 2)
    plt.imshow(prior_ensemble.mean(axis=(0, 1)))
    plt.colorbar()
    plt.title("Prior Ensemble Mean")
    plt.subplot(2, 3, 3)
    plt.imshow(posterior_ensemble.mean(axis=(0, 1)))
    plt.colorbar()
    plt.title("Posterior Ensemble Mean")
    plt.subplot(2, 3, 4)
    plt.imshow(true_sol - prior_ensemble.mean(axis=(0, 1)))
    plt.colorbar()
    plt.title("|True - Prior| difference")
    plt.subplot(2, 3, 5)
    plt.imshow(true_sol - posterior_ensemble.mean(axis=(0, 1)))
    plt.colorbar()
    plt.title("|True - Posterior| difference")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.subplot(2, 3, 6)
    plt.plot(
        prior_ensemble[:, :, idx_to_plot, :].mean(axis=(0, 1)),
        label="Prior",
        color="tab:red",
        linewidth=3,
    )
    plt.plot(
        posterior_ensemble[:, :, idx_to_plot, :].mean(axis=(0, 1)),
        label="Posterior",
        color="tab:blue",
        linewidth=3,
    )
    # Shade the standard deviation of the posterior on the time series plot
    mean_post = posterior_ensemble[:, :, idx_to_plot, :].mean(axis=(0, 1))
    std_post = posterior_ensemble[:, :, idx_to_plot, :].std(axis=(0, 1))
    time_axis = np.arange(posterior_ensemble.shape[-1])
    plt.fill_between(
        time_axis,
        mean_post - std_post,
        mean_post + std_post,
        color="tab:blue",
        alpha=0.2,
        label="Posterior ± Std",
    )

    plt.plot(
        true_sol[idx_to_plot, :],
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
