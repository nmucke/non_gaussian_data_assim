import pdb

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from non_gaussian_data_assim.da_methods.enkf import EnsembleKalmanFilter
from non_gaussian_data_assim.da_methods.enkf_loc import EnsembleKalmanFilterLocalization
from non_gaussian_data_assim.da_methods.particle_filter import ParticleFilter
from non_gaussian_data_assim.da_methods.pff import ParticleFlowFilter
from non_gaussian_data_assim.da_methods.pff_loc import ParticleFlowFilterLocalization
from non_gaussian_data_assim.forward_models.lorenz_96 import L96_RK4, L96_RK4_ensemble

np.random.seed(42)

# Constants and parameters
DT = 0.01
F = 8.0
NUM_TIME_STEPS = 100
STATE_DIM = 50
MEM_SIZE = 1000

# Initial state
X_0 = np.random.randn(STATE_DIM) * 10
X_0[-1] = X_0[0]  # Periodic boundary condition

# Observation ids
OBS_IDS = np.arange(0, STATE_DIM, 5)
NO_OBS_IDS = np.setdiff1d(np.arange(0, STATE_DIM), OBS_IDS)

# Observation error covariance matrix
R = np.eye(len(OBS_IDS)) * 0.01


def main() -> None:

    true_sol = np.zeros((NUM_TIME_STEPS, STATE_DIM))
    true_sol[0, :] = X_0
    for i in range(1, NUM_TIME_STEPS):
        true_sol[i, :] = L96_RK4(true_sol[i - 1, :], DT, F)

    obs_vect = true_sol.copy()
    obs_vect[:, NO_OBS_IDS] = -9999  # -999 indicates no observation

    da_model = ParticleFilter(
        mem=MEM_SIZE,
        nx=STATE_DIM,
        R=R,
        # N=STATE_DIM,
        # r_influ=10,
        obs_operator=lambda x: x,
    )

    prior_ensemble = np.zeros((NUM_TIME_STEPS, STATE_DIM, MEM_SIZE))
    prior_ensemble[0, :, :] = np.random.randn(STATE_DIM, MEM_SIZE) * 10
    posterior_ensemble = prior_ensemble.copy()
    for i in tqdm(range(1, NUM_TIME_STEPS)):
        prior_ensemble[i, :] = L96_RK4_ensemble(prior_ensemble[i - 1, :], DT, F)
        posterior_ensemble[i, :] = da_model(
            prior_ensemble=prior_ensemble[i, :], obs_vect=obs_vect[i, :]
        )

    prior_error = true_sol - prior_ensemble.mean(axis=-1)
    prior_error = np.sqrt(np.sum(prior_error**2, axis=-1))
    posterior_error = true_sol - posterior_ensemble.mean(axis=-1)
    posterior_error = np.sqrt(np.sum(posterior_error**2, axis=-1))

    print(f"Prior error: {prior_error.mean()}")
    print(f"Posterior error: {posterior_error.mean()}")

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(true_sol)
    plt.colorbar()
    plt.title("True Solution")
    plt.subplot(2, 3, 2)
    plt.imshow(prior_ensemble.mean(axis=-1))
    plt.colorbar()
    plt.title("Prior Ensemble Mean")
    plt.subplot(2, 3, 3)
    plt.imshow(posterior_ensemble.mean(axis=-1))
    plt.colorbar()
    plt.title("Posterior Ensemble Mean")
    plt.subplot(2, 3, 4)
    plt.imshow(true_sol - prior_ensemble.mean(axis=-1))
    plt.colorbar()
    plt.title("|True - Prior| difference")
    plt.subplot(2, 3, 5)
    plt.imshow(true_sol - posterior_ensemble.mean(axis=-1))
    plt.colorbar()
    plt.title("|True - Posterior| difference")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.subplot(2, 3, 6)
    plt.plot(
        prior_ensemble[:, 25, :].mean(axis=-1),
        label="Prior",
        color="tab:red",
        linewidth=3,
    )
    plt.plot(
        posterior_ensemble[:, 25, :].mean(axis=-1),
        label="Posterior",
        color="tab:blue",
        linewidth=3,
    )
    # Shade the standard deviation of the posterior on the time series plot
    mean_post = posterior_ensemble[:, 25, :].mean(axis=-1)
    std_post = posterior_ensemble[:, 25, :].std(axis=-1)
    time_axis = np.arange(posterior_ensemble.shape[0])
    plt.fill_between(
        time_axis,
        mean_post - 3 * std_post,
        mean_post + 3 * std_post,
        color="tab:blue",
        alpha=0.2,
        label="Posterior ±3 Std",
    )

    plt.plot(true_sol[:, 25], label="True", color="black", linewidth=3, linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("State 25")
    plt.ylim(-5, 5)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
