"""Hydra-free entrypoint for the Kuramoto-Sivashinsky data-assimilation experiment.

This mirrors ``scripts/main.py`` exactly, but every object is constructed
directly in Python instead of being instantiated from Hydra configs. Its only
purpose is to demonstrate that the pipeline runs without Hydra.

Configuration is hardcoded to match ``configs/case/kuramoto.yaml`` with the
EnKF DA method (``configs/da_method/enkf.yaml``).

Run with:
    python scripts/main_manual.py
"""

import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from non_gaussian_data_assim.da_methods.enkf import EnsembleKalmanFilter
from non_gaussian_data_assim.ensemble_generation.initial_ensemble import (
    InitialEnsembleGenerator,
)
from non_gaussian_data_assim.ensemble_generation.initial_state import InitialState
from non_gaussian_data_assim.forward_models.kuramoto_sivashinsky import (
    KuramotoSivashinsky,
)
from non_gaussian_data_assim.initial_profiles import (
    CosineProfile,
    CoupledKuramotoPseudo1DProfile,
)
from non_gaussian_data_assim.metrics.ensemble_metrics import CRPS
from non_gaussian_data_assim.metrics.innovation_metrics import (
    ChiSquared,
    NormalizedInnovations,
)
from non_gaussian_data_assim.metrics.trajectory_metrics import (
    MAE,
    NRMSE,
    RMSE,
    ensemble_spread,
    print_metrics_table,
)
from non_gaussian_data_assim.observations.observation_operator import (
    LinearObservationOperator,
)
from non_gaussian_data_assim.observations.observation_utils import generate_observations
from non_gaussian_data_assim.perturbations.white_noise import WhiteNoise
from non_gaussian_data_assim.plotting.plot_fields import plot_high_dim_field
from non_gaussian_data_assim.plotting.plot_initial_ensemble import plot_initial_fields
from non_gaussian_data_assim.plotting.plot_innov_stats import plot_innov
from non_gaussian_data_assim.plotting.plot_metrics import plot_metric_timeseries

logger = logging.getLogger(__name__)


# ===================== Hardcoded configuration (Kuramoto + EnKF) =====================
# Common, case-agnostic settings (configs/config.yaml + configs/case/kuramoto.yaml).
SEED = 42
DATA_ASSIMILATION_STEPS = 40
MODEL_INTEGRATION_STEPS = 10
ENSEMBLE_SIZE = 50
SPINUP_STEPS = 30  # spinup run for SPINUP_STEPS * MODEL_INTEGRATION_STEPS

INFLATION_FACTOR = 1.0
LOCALIZATION_DISTANCE = 100

# Case: Kuramoto-Sivashinsky.
TITLE = "Kuramoto-Sivashinsky"
NUM_STATES = 1
STATE_DIM = 512
OBS_NOISE_VARIANCE = 0.1
DOMAIN_LENGTH = 100

# Whether to center the prior ensemble on a best-guess (else on the profile).
USE_BEST_GUESS = True

# DA method.
DA_METHOD_NAME = "enkf"
# =====================================================================================


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    rng_key = jax.random.PRNGKey(SEED)
    rng_key, initial_state_key, initial_ensemble_key, obs_key = jax.random.split(
        rng_key, 4
    )

    ########## Forward model ##########
    forward_model = KuramotoSivashinsky(
        dt=0.05,
        model_integration_steps=MODEL_INTEGRATION_STEPS,
        state_dim=STATE_DIM,
        domain_length=DOMAIN_LENGTH,
    )
    logger.info(f"Forward model: {forward_model}")

    ########## True state ##########
    # Initial state
    initial_state_generator = InitialState(
        initial_profile=CosineProfile(
            num_states=NUM_STATES,
            state_dim=STATE_DIM,
            domain_length=DOMAIN_LENGTH,
            periodic=False,
            magnitude=1,
        ),
        forward_model=forward_model,
        spinup_steps=SPINUP_STEPS,
        periodic=False,
    )
    true_initial_state = initial_state_generator.sample(rng_key=initial_state_key)
    logger.info(f"Initial state generator: {initial_state_generator}")

    # Rollout true solution
    true_sol = forward_model.rollout(
        true_initial_state,
        DATA_ASSIMILATION_STEPS,
        return_model_integration_steps=True,
    )

    ########## Initial ensemble ##########
    initial_ensemble_generator = InitialEnsembleGenerator(
        perturbation=WhiteNoise(
            num_states=NUM_STATES,
            state_dim=STATE_DIM,
            scale=1.0,
        ),
        forward_model=forward_model,
        initial_profile=CoupledKuramotoPseudo1DProfile(
            num_states=NUM_STATES,
            state_dim=STATE_DIM,
            domain_length=DOMAIN_LENGTH,
            decorrelation_length=10.0,
            scale=1.0,
        ),
        spinup_steps=SPINUP_STEPS,
        periodic=False,
        use_best_guess=True,  # This overwrites the initial_profile
        best_guess_perturbation="natural_variability",
    )
    initial_ensemble = initial_ensemble_generator.sample(
        rng_key=initial_ensemble_key,
        ensemble_size=ENSEMBLE_SIZE,
        best_guess=true_initial_state if USE_BEST_GUESS else None,
    )
    logger.info(f"Initial ensemble generator: {initial_ensemble_generator}")

    ########## Observation operator ##########
    obs_operator = LinearObservationOperator(
        state_dim=STATE_DIM,
        obs_states=[0],
        obs_indices=np.arange(0, STATE_DIM, 4),
    )
    R = jnp.eye(obs_operator.num_obs) * OBS_NOISE_VARIANCE
    logger.info(f"Observation operator: {obs_operator}")

    # Observe true solution
    observations = generate_observations(
        rng_key=obs_key,
        true_sol=true_sol,
        obs_operator=obs_operator,
        R=R,
        data_assimilation_steps=DATA_ASSIMILATION_STEPS,
        model_integration_steps=MODEL_INTEGRATION_STEPS,
    )

    ########## DA model ##########
    da_model = EnsembleKalmanFilter(
        ensemble_size=ENSEMBLE_SIZE,
        R=R,
        obs_operator=obs_operator,
        forward_operator=forward_model,
        inflation_factor=INFLATION_FACTOR,
        localization_distance=LOCALIZATION_DISTANCE,
    )
    logger.info(f"DA model: {da_model}")

    ########## Prepare ensembles ##########
    posterior_ensemble = initial_ensemble.copy().reshape(
        ENSEMBLE_SIZE, 1, NUM_STATES, STATE_DIM
    )
    prior_ensemble_da = initial_ensemble.copy().reshape(
        ENSEMBLE_SIZE, 1, NUM_STATES, STATE_DIM
    )

    ########## Rollout the initial ensemble for comparison ##########
    reference_ensemble = forward_model.rollout(
        initial_ensemble,
        DATA_ASSIMILATION_STEPS,
        return_model_integration_steps=True,
    )

    ########## Run the DA loop. ##########
    # Initialize empty list to track predicted observations (prior in obs space).
    predicted_obs = []
    logger.info(f"Running DA loop for {DATA_ASSIMILATION_STEPS} steps")
    for i in tqdm(range(DATA_ASSIMILATION_STEPS)):
        rng_key, key = jax.random.split(rng_key)

        prior_current = posterior_ensemble[:, -1]

        posterior_next = da_model(
            prior_ensemble=prior_current,
            obs_vect=observations[i],
            rng_key=key,
            return_model_integration_steps=True,
        )
        if jnp.isnan(posterior_next).any():
            print(f"NaN in posterior_next at time {i}")
            break

        # ------------Track:  Prior ensemble (in obs space)    -----------
        # Get model-state in obs-space
        HXf = da_model.obs_operator(prior_current)  # shape: (EnsSize, N_obs)
        predicted_obs.append(HXf)
        # -----------------------------------------------------------------

        # Concatenate prior (1 time-step) and posterior (2 time-steps).
        prior_ensemble_da = jnp.concatenate(
            [prior_ensemble_da, prior_current[:, None, :, :]], axis=1
        )
        posterior_ensemble = jnp.concatenate(
            [posterior_ensemble, posterior_next], axis=1
        )

    logger.info(f"Finished DA loop")

    # ======================== metrics and plotting ======================================
    # Metrics.
    rmse = RMSE(ensemble_aggregation="mean", time_aggregation="mean")
    mae = MAE(ensemble_aggregation="mean", time_aggregation="mean")
    nrmse = NRMSE(ensemble_aggregation="mean", time_aggregation="mean")
    crps = CRPS(time_aggregation="mean")
    rmse_time = RMSE(ensemble_aggregation="mean", time_aggregation="none")
    crps_time = CRPS(time_aggregation="none")
    innov_white = NormalizedInnovations()
    chi2_mean = ChiSquared(time_aggregation="mean")
    chi2_time = ChiSquared(time_aggregation="none")

    def get_metric_dict(
        ensemble: jnp.ndarray, true_sol: jnp.ndarray, state_dim: Optional[int] = None
    ) -> dict:

        metrics = {
            "rmse": rmse(ensemble, true_sol),
            "mae": mae(ensemble, true_sol),
            "nrmse": nrmse(ensemble, true_sol),
            "crps": crps(ensemble, true_sol),
            "rmse_time": rmse_time(ensemble, true_sol),
            "spread_time": ensemble_spread(ensemble, state_dim=state_dim),
            "crps_time": crps_time(ensemble, true_sol),
        }
        return metrics

    reference_metrics = get_metric_dict(reference_ensemble, true_sol[0])
    posterior_metrics = get_metric_dict(posterior_ensemble, true_sol[0])

    # -- If multiple states are present, save metrics for each individual state
    post_metric_states = []
    if NUM_STATES > 1:
        for i in range(NUM_STATES):
            metric_dict = get_metric_dict(
                posterior_ensemble[:, :, i, :], true_sol[0, :, i, :]
            )
            post_metric_states.append(metric_dict)

    print_metrics_table(reference_metrics, posterior_metrics, title=f"{TITLE} Metrics")

    # --- Plot Hovmöller diagrams and assim-time-series
    logger.info(f"Plotting...")
    plotter = plot_high_dim_field
    plotter(
        true_sol=true_sol,
        reference_ensemble=reference_ensemble,
        posterior_ensemble=posterior_ensemble,
        title=TITLE,
        da_method_name=DA_METHOD_NAME,
        ensemble_size=ENSEMBLE_SIZE,
        reference_metrics=reference_metrics,
        posterior_metrics=posterior_metrics,
        state_dim=STATE_DIM,
        data_assimilation_steps=DATA_ASSIMILATION_STEPS,
        model_integration_steps=MODEL_INTEGRATION_STEPS,
    )

    # --- Plot time-series of errors
    post_metrics_all = [posterior_metrics] + post_metric_states
    _ = plot_metric_timeseries(post_metrics_all)
    # ====================== For EnKF application plot innovation statistics ======================
    if DA_METHOD_NAME == "enkf":

        # -- Store Predicted-Obs and Split-up per state
        pred_obs = jnp.stack(
            predicted_obs, axis=0
        )  # shape: [EnsSize, Assim-Step, N_obs (comined for all states)]

        idx = 0
        predobs_states, obs_states, R_states = [], [], []
        for obs_state in da_model.obs_operator.obs_indices_per_state:
            i = obs_state.shape[0]
            predobs_states.append(pred_obs[:, :, idx : idx + i])
            obs_states.append(observations[:, idx : idx + i])
            R_states.append(R[idx : idx + i, idx : idx + i])
            idx += i

        innov_metric_list = []
        for p_obs, obs, R_state in zip(predobs_states, obs_states, R_states):
            innovation_metrics = {
                "chi_sq_mean": chi2_mean(predicted_obs=p_obs, obs=obs, R=R_state),
                "chi_sq_time": chi2_time(predicted_obs=p_obs, obs=obs, R=R_state),
                "z": innov_white(predicted_obs=p_obs, obs=obs, R=R_state),
            }
            innov_metric_list.append(innovation_metrics)

        # --- Plot Innovations Diagnostics and Skill-Scores
        _ = plot_innov(
            innov_stats_list=innov_metric_list, bins=74, hist_range=None, show_fig=True
        )
    # ==============================================================================================================

    # --- Plot Initial-Conditions (I.C, Initial-Ensemble) + Breeding statistics if available
    plot_initial_fields(
        ensemble=posterior_ensemble,
        truth_t0=true_sol,
        x_before_spinup=None,
        best_guess_profile=(true_initial_state[0] if USE_BEST_GUESS else None),
        bv_dict=None,
    )


if __name__ == "__main__":
    main()
