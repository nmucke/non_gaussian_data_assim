import pdb

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from non_gaussian_data_assim.da_methods.agmf import AdaptiveGaussianMixtureFilter
from non_gaussian_data_assim.da_methods.enkf import EnsembleKalmanFilter

# from non_gaussian_data_assim.da_methods.enkf_loc import EnsembleKalmanFilterLocalization
# from non_gaussian_data_assim.da_methods.particle_filter import ParticleFilter
# from non_gaussian_data_assim.da_methods.pff import ParticleFlowFilter
# from non_gaussian_data_assim.da_methods.pff_loc import ParticleFlowFilterLocalization
from non_gaussian_data_assim.da_methods.pff import ParticleFlowFilter
from non_gaussian_data_assim.forward_models.kuramoto_sivashinsky import KuramotoSivashinsky
from non_gaussian_data_assim.observation_operator import ObservationOperator

np.random.seed(42)

# Constants and parameters
DT = 0.05
NU = 1.0
NUM_TIME_STEPS = 500
NUM_STATES = 1
STATE_DIM = 1024
NUM_MODEL_STEPS = 4
NUM_SKIP_OBS = 4
ENSEMBLE_SIZE = 100
DOMAIN_LENGTH = 100

# Observation ids
OBS_IDS = np.arange(0, STATE_DIM, NUM_SKIP_OBS)
OBS_STATES = (0,)

# Observation error covariance matrix
R = np.eye(len(OBS_IDS)) * 0.25

x = np.linspace(start=0, stop=DOMAIN_LENGTH, num=STATE_DIM)

# initial condition 
U0 = lambda magnitude: magnitude * np.cos((2 * np.pi * x) / DOMAIN_LENGTH) + magnitude * np.cos((4 * np.pi * x) / DOMAIN_LENGTH)

U0_PRIOR = lambda magnitude: magnitude * np.cos((2 * np.pi * x) / DOMAIN_LENGTH) + magnitude * np.cos((4 * np.pi * x) / DOMAIN_LENGTH)



def main() -> None:
    """Main function."""

    # Define the forward model
    forward_model = KuramotoSivashinsky(
        dt=DT,
        num_model_steps=NUM_MODEL_STEPS, 
        state_dim=STATE_DIM, 
        domain_length=DOMAIN_LENGTH,
        nu=NU
    )

    # Define the observation operator
    obs_operator = ObservationOperator(
        obs_states=OBS_STATES, obs_indices=OBS_IDS, state_dim=STATE_DIM
    )

    # Initialize the true solution and observations
    observations = np.zeros((len(OBS_IDS), NUM_TIME_STEPS))
    true_sol = np.zeros((1, NUM_STATES, STATE_DIM, NUM_TIME_STEPS))
    true_sol[0, ..., 0] = U0(0.1)

    # Perform the forward model
    for i in range(1, NUM_TIME_STEPS):
        true_sol[..., i] = forward_model(true_sol[..., i - 1])
        observations[:, i] = obs_operator(true_sol[..., i])

    # plt.figure()
    # plt.imshow(true_sol[0, 0, :, -STATE_DIM*2:])
    # plt.colorbar()
    # plt.show()
    # pdb.set_trace()

    # Define the data assimilation model
    da_model = EnsembleKalmanFilter(
        ensemble_size=ENSEMBLE_SIZE,
        R=R,
        obs_operator=obs_operator,
        forward_operator=forward_model,
        inflation_factor=8.0,
        localization_distance=10,
        # w_prev=np.ones(ENSEMBLE_SIZE) / ENSEMBLE_SIZE,
        # nc_threshold=0.5,
    )

    # Initialize the prior ensemble
    prior_ensemble = np.zeros((ENSEMBLE_SIZE, NUM_STATES, STATE_DIM, NUM_TIME_STEPS))
    magnitude_samples = np.random.uniform(low=0.0, high=1.5, size=ENSEMBLE_SIZE)
    prior_ensemble[..., 0] = np.array([U0_PRIOR(magnitude) for magnitude in magnitude_samples]).reshape(ENSEMBLE_SIZE, 1, STATE_DIM)
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

    idx_to_plot = STATE_DIM//5 + 51

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(true_sol[..., -STATE_DIM*2:].T, origin="lower")
    plt.colorbar()
    plt.title("True Solution")
    plt.subplot(2, 3, 2)
    plt.imshow(prior_ensemble.mean(axis=(0, 1))[..., -STATE_DIM*2:].T, origin="lower")
    plt.colorbar()
    plt.title("Prior Ensemble Mean")
    plt.subplot(2, 3, 3)
    plt.imshow(posterior_ensemble.mean(axis=(0, 1))[..., -STATE_DIM*2:].T, origin="lower")
    plt.colorbar()
    plt.title("Posterior Ensemble Mean")
    plt.subplot(2, 3, 4)
    plt.imshow(prior_error[..., -STATE_DIM*2:].T, origin="lower")
    plt.colorbar()
    plt.title("|True - Prior| difference")
    plt.subplot(2, 3, 5)
    plt.imshow(posterior_error[..., -STATE_DIM*2:].T, origin="lower")
    plt.colorbar()
    plt.title("|True - Posterior| difference")
    plt.xlabel("Space")
    plt.ylabel("Time")
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
    plt.ylabel(f"State {idx_to_plot}")
    plt.ylim(-3, 3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
