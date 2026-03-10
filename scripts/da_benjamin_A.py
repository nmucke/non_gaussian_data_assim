import pdb
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.stats import gaussian_kde
from scipy.io import loadmat

from non_gaussian_data_assim.da_methods.agmf import AdaptiveGaussianMixtureFilter
from non_gaussian_data_assim.da_methods.enkf import EnsembleKalmanFilter
from non_gaussian_data_assim.da_methods.pff import ParticleFlowFilter
from non_gaussian_data_assim.forward_models.identity import IdentityModel
from non_gaussian_data_assim.observation_operator import SineObservationOperatorNoError

jax.config.update("jax_disable_jit", False)
jax.config.update("jax_enable_x64", True)


SEED = 42

ENSEMBLE_SIZE = 200

DA_METHOD = "pff"
DA_METHODS = {
    "enkf": EnsembleKalmanFilter,
    "agmf": AdaptiveGaussianMixtureFilter,
    "pff": ParticleFlowFilter,
}
SPECIFIC_DA_ARGS = {
    "enkf": {
        "inflation_factor": 5.0,
    },
    "agmf": {
        "inflation_factor": 1.0,
        "nc_threshold": 0.5,
        "w_prev": np.ones(ENSEMBLE_SIZE) / ENSEMBLE_SIZE,
    },
    "pff": {
        "return_pff_trajectory": True,
        "num_pseudo_time_steps": 10000,
        "step_size": 0.01,
        # "stepper": "runge_kutta_4",
        "stepper": "forward_euler",
        # "stepper": "backward_euler",
        # "divergence_type": "hellinger",
        # "weight_estimation": "kde",
        # "kde_bandwidth": 0.1,
        # "hellinger_cov_regularization": 1e-6,
    },
}

NUM_STATES = 1
STATE_DIM = 1

# Observation ids
OBS_IDS = (0,)
OBS_STATES = (0,)

# Observation error covariance matrix
R = jnp.eye(len(OBS_IDS)) * 1.0
B = 1.0


def main() -> None:
    """Main function."""
    rng_key = jax.random.PRNGKey(SEED)

    # Define the forward model
    forward_model = IdentityModel(state_dim=STATE_DIM)

    # Define the observation operator
    obs_operator = SineObservationOperatorNoError(
        obs_states=OBS_STATES,
        obs_indices=OBS_IDS,
        state_dim=STATE_DIM,
    )

    # Generate observations
    observations = jnp.array([0.0])

    da_model = DA_METHODS[DA_METHOD](
        ensemble_size=ENSEMBLE_SIZE,
        R=R,
        obs_operator=obs_operator,
        forward_operator=forward_model,
        **SPECIFIC_DA_ARGS[DA_METHOD],
    )

    # Initialize the prior ensemble
    # prior_ensemble = jnp.linspace(-1.0, 3.0, ENSEMBLE_SIZE).reshape(ENSEMBLE_SIZE, 1, 1)
    prior_ensemble = loadmat("benjamin_case/testcaseA_det.mat")["X_PFF"][
        :, 0, 0
    ].reshape(ENSEMBLE_SIZE, 1, 1)
    prior_kde = gaussian_kde(prior_ensemble[:, 0, 0])

    # Initialize the posterior ensemble
    posterior_ensemble = prior_ensemble.copy()

    # Perform the data assimilation
    posterior_ensemble = posterior_ensemble.reshape(
        ENSEMBLE_SIZE, 1, NUM_STATES, STATE_DIM
    )
    t0 = time.time()
    posterior_ensemble, nikolaj_rhs_fn = da_model(
        prior_ensemble=posterior_ensemble[:, -1],
        obs_vect=observations,
        return_inner_steps=False,
        prior_mean=jnp.ones(1),
        prior_cov=jnp.eye(1),
    )
    t1 = time.time()
    print(f"Time taken: {t1 - t0} seconds")

    posterior_ensemble = posterior_ensemble[:, -1, 0, 0]
    kde = gaussian_kde(posterior_ensemble)

    x = np.linspace(-2.5, 5.0, 100)

    # # Load reference data for verification
    # benjamin_init_rhs = loadmat("benjamin_case/testcaseA_IC.mat")
    # benjamin_pff_trajectory = loadmat("benjamin_case/testcaseA_det.mat")
    # x0_PFF = benjamin_init_rhs["x0_PFF"]  # [ensemble, state]
    # res_initial = benjamin_init_rhs["res_initial"]  # [ensemble, state]
    # X_PFF = benjamin_pff_trajectory["X_PFF"]  # [ensemble, state, time]

    # nikolaj_rhs_eval = nikolaj_rhs_fn(x0_PFF)

    # # Verification report
    # sep = "=" * 60
    # print(f"\n{sep}")
    # print("  BENJAMIN vs NIKOLAJ PFF VERIFICATION")
    # print(f"{sep}\n")

    # print(f"  x0_PFF comes from testcaseA_IC.mat")
    # print(f"  res_initial comes from testcaseA_IC.mat")
    # print(f"  X_PFF comes from testcaseA_det.mat")
    # print()

    # # 1. RHS comparison at initial condition
    # rhs_diff = nikolaj_rhs_eval - res_initial
    # rhs_norm = float(jnp.linalg.norm(rhs_diff, ord=jnp.inf))
    # print("  1. RHS at x0_PFF (testcaseA_IC.mat)")
    # print("     Compare: Nikolaj_RHS(x0_PFF) vs Benjamin res_initial")
    # print(f"     Inf norm: {rhs_norm:.6e}")
    # print()

    # # 2. Euler step from x0_PFF
    # nikolaj_step = x0_PFF + 0.01 * nikolaj_rhs_eval
    # step_diff = nikolaj_step - X_PFF[:, :, 1]
    # step_norm = float(jnp.linalg.norm(step_diff, ord=jnp.inf))
    # print("  2. Euler step from x0_PFF (IC)")
    # print("     Compare: x0_PFF + 0.01*Nikolaj_RHS(x0_PFF) vs X_PFF[:,:,1]")
    # print(f"     Inf norm: {step_norm:.6e}")
    # print()

    # # 3. Euler step from X_PFF[:,:,0]
    # step_benjamin = X_PFF[:, :, 0] + 0.01 * nikolaj_rhs_fn(X_PFF[:, :, 0])
    # step_benjamin_diff = step_benjamin - X_PFF[:, :, 1]
    # step_benjamin_norm = float(jnp.linalg.norm(step_benjamin_diff, ord=jnp.inf))
    # print("  3. Euler step from X_PFF[:,:,0] (first det step)")
    # print("     Compare: X_PFF[:,:,0] + 0.01*Nikolaj_RHS(X_PFF[:,:,0]) vs X_PFF[:,:,1]")
    # print(f"     Inf norm: {step_benjamin_norm:.6e}")
    # print(f"{sep}\n")

    benjamin_samples = loadmat("benjamin_case/testcaseA_det.mat")

    X_PFF = benjamin_samples["X_PFF"][:, 0, -1]

    kde_benjamin = gaussian_kde(X_PFF)

    n_bins = 100

    plt.figure()
    plt.plot(x, prior_kde.pdf(x), color="tab:orange", linewidth=3, label="Prior")
    plt.hist(X_PFF, bins=n_bins, density=True, color="tab:red", alpha=0.2)
    plt.plot(x, kde_benjamin.pdf(x), color="tab:red", linewidth=3, label="Benjamin PFF")
    # plt.plot(x_pdf_SDE, pdf_SDE, color="tab:blue", linewidth=3, label="Benjamin SDE")
    plt.hist(
        posterior_ensemble, bins=n_bins, density=True, color="tab:green", alpha=0.2
    )
    plt.plot(
        x,
        kde.pdf(x),
        color="tab:green",
        linewidth=3,
        label="Nikolaj PFF",
        linestyle="--",
    )
    plt.grid(True)
    plt.xlim(x.min(), x.max())
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("p(x|y)")
    plt.title("Posterior Distribution")
    plt.show()


if __name__ == "__main__":
    main()
