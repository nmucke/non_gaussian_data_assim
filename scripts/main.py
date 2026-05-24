"""Unified entrypoint for data-assimilation experiments, configured via Hydra.

Examples:
    python scripts/main.py case=lorenz_63 da_method=enkf
    python scripts/main.py case=lorenz_96 da_method=pff
    python scripts/main.py case=kuramoto da_method=agmf data_assimilation_steps=100
"""

import logging
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from hydra import main as hydra_main
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import non_gaussian_data_assim.utils.saving as saving
from non_gaussian_data_assim.metrics.ensemble_metrics import CRPS
from non_gaussian_data_assim.metrics.innovation_metrics import (
    ChiSquared,
    NormalizedInnovations,
)
from non_gaussian_data_assim.metrics.trajectory_metrics import (
    MAE,
    MAPE,
    RMSE,
    ensemble_spread,
    print_metrics_table,
)
from non_gaussian_data_assim.observations.observation_utils import generate_observations
from non_gaussian_data_assim.plotting.plot_initial_ensemble import plot_initial_fields
from non_gaussian_data_assim.plotting.plot_innov_stats import plot_innov
from non_gaussian_data_assim.plotting.plot_metrics import plot_metric_timeseries
from non_gaussian_data_assim.utils.spinup import spinup_ensemble

logger = logging.getLogger(__name__)


@hydra_main(config_path="../configs", config_name="config", version_base=None)  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    forward_model_cfg = cfg.case.forward_model
    true_initial_state_cfg = cfg.case.true_initial_state
    obs_operator_cfg = cfg.case.obs_operator
    initial_ensemble_cfg = cfg.case.initial_ensemble
    plotter_cfg = cfg.case.plotter
    da_method_cfg = OmegaConf.merge(
        cfg.da_method, cfg.case.da_method_overrides[cfg.da_method.name]
    )

    # ----------------------------------------------------------------------
    # --- Save Config
    save_name = str(cfg.save_name).strip()
    base_path = Path("../experiments")
    if bool(save_name) and isinstance(save_name, str):
        # --- Check if folder-nmae is still free, if so, create folder
        save_path_exp = saving.create_experiment_folder(save_name, root=base_path)
        saving.save_config(cfg, save_path_exp)
    # ---------------------------

    rng_key = jax.random.PRNGKey(cfg.seed)

    # --- Instantiate Forward model and observation operator
    forward_model = instantiate(forward_model_cfg)
    logger.info(f"Forward model: {forward_model}")

    # --- Instantiate Observation operator
    obs_operator = instantiate(obs_operator_cfg)
    logger.info(f"Observation operator: {obs_operator}")

    # --- Define an initial GROUND-TRUTH state.
    true_initial_state_profile = instantiate(true_initial_state_cfg)
    logger.info(f"Initial state: {true_initial_state_profile}")
    rng_key, key = jax.random.split(rng_key)

    # --- BEST-GUESS: Reference state for ensemble members --> determines how difficulat assimilation task will be
    BEST_GUESS_FLAG = initial_ensemble_cfg.centered_around_bestguess

    ic_ref = true_initial_state_profile.sample(rng_key=key)

    # --- Spin-Up truth Run (optional)
    if cfg.spinup_steps:
        logger.info(
            f"Run Spin-up of {cfg.spinup_steps * cfg.model_integration_steps} model-steps"
        )

        get_std = True if BEST_GUESS_FLAG else False
        true_sol = spinup_ensemble(
            ensemble=ic_ref,
            forward_model=forward_model,
            spinup_steps=cfg.spinup_steps,
            get_natural_variablity=get_std,
        )
        if get_std:
            true_sol, natural_variability = true_sol

    else:
        logger.info("No SPINUP")
        true_sol = ic_ref

        try:
            natural_variability = cfg.case.true_initial_state.magnitude
        except:
            natural_variability = cfg.case.true_initial_state.scale

    # --- Rollout the truth.
    x0_truth = true_sol
    true_sol = forward_model.rollout(
        x0_truth, cfg.data_assimilation_steps, return_model_integration_steps=True
    )

    if BEST_GUESS_FLAG:
        """
        Best-Guess: Add white noise (certain scale) to ic_ref and spin-up with same aound of setps as x0_truht
            ic_ref              --- spin-up ---> x0_truth
            ic_ref + white noie --- spin-up ---> best_guess

        My suggestion for what scale to use is to look at the standard deviation over the spin-up period of x0_truh and add that
        """
        from non_gaussian_data_assim.ensemble_generation.ensemble_perturbations import (
            WhiteNoise,
        )

        rng_key, bg_key = jax.random.split(rng_key)
        whitenoise = WhiteNoise(
            num_states=cfg.case.num_states,
            state_dim=cfg.case.state_dim,
            scale=natural_variability / 3,
        )
        perturbs_best_guess = whitenoise.sample(rng_key=bg_key, ensemble_size=1)
        best_guess_profile0 = ic_ref + perturbs_best_guess

        best_guess_profile = spinup_ensemble(
            ensemble=best_guess_profile0,
            forward_model=forward_model,
            spinup_steps=cfg.spinup_steps,
            get_natural_variablity=False,
        )

    else:
        best_guess_profile = None

    # --- Create synthetic observations
    # Define Observations-error covariance
    R = jnp.eye(obs_operator.num_obs) * cfg.case.obs_noise_variance
    logger.info(f"Observation noise variance: {cfg.case.obs_noise_variance}")
    #  Generate observations from the truth.
    rng_key, obs_key = jax.random.split(rng_key)
    observations = generate_observations(
        rng_key=obs_key,
        true_sol=true_sol,
        obs_operator=obs_operator,
        R=R,
        data_assimilation_steps=cfg.data_assimilation_steps,
        model_integration_steps=cfg.model_integration_steps,
    )

    # --- Instantiate DA method (with case-specific overrides applied).
    da_model = instantiate(
        da_method_cfg,
        ensemble_size=cfg.ensemble_size,
        R=R,
        obs_operator=obs_operator,
        forward_operator=forward_model,
    )
    logger.info(f"DA method: {da_model}")

    # --- GENERATE INITIAL ENSEMBLE !!!!!
    initial_ensemble_generator = instantiate(initial_ensemble_cfg)
    logger.info(f"Prior ensemble: {initial_ensemble_generator}")
    rng_key, key = jax.random.split(rng_key)

    output = initial_ensemble_generator.sample(
        rng_key=key, ensemble_size=cfg.ensemble_size, bg_profile=best_guess_profile
    )
    # Flexibel handling of Ens.Gen. output (Breeding might return additional diagnostics)
    if isinstance(output, tuple):
        reference_ensemble, ensgen_diagnostics = output
    else:
        reference_ensemble = output

    # --- Handle spin-up of ensemble (only when no best-guess is used!!)
    if not BEST_GUESS_FLAG and cfg.spinup_steps:
        logger.info(
            "\nSPIN-UP Initials Ensemble (as it is NOT centered around a best-guess profile)"
        )
        reference_ensemble = spinup_ensemble(
            ensemble=reference_ensemble,
            forward_model=forward_model,
            spinup_steps=cfg.spinup_steps,
        )

    # TODO: Would it not be enough to only rollout best-guess (if present)
    # ---------------- Initialisation of data-containers that will be filled in DA-loop ------------------------------
    # Initialize the posterior ensemble from the Initial ensemble.
    posterior_ensemble = reference_ensemble.copy().reshape(
        cfg.ensemble_size, 1, cfg.case.num_states, cfg.case.state_dim
    )
    # Initialize the Prior-Ensemble from the reference. NOTE: Prior-Ensemble only stores fields at assim-steps before assimilation is done
    prior_ensemble_da = reference_ensemble.copy().reshape(
        cfg.ensemble_size,
        1,
        cfg.case.num_states,
        cfg.case.state_dim,
    )
    # ----------------------------------------------------------------------------------------------------------------

    # --- Rollout the initial ensemble for comparison.
    reference_ensemble = forward_model.rollout(
        reference_ensemble,
        cfg.data_assimilation_steps,
        return_model_integration_steps=True,
    )

    # -------------------- Run the DA loop. ---------------------------------------------------------
    # Initialize empty lists to track chi-square and normalized nnovations (z)
    predicted_obs = []
    logger.info(f"Running DA loop for {cfg.data_assimilation_steps} steps")
    for i in tqdm(range(cfg.data_assimilation_steps)):
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

        # Concatenate prior (1time-step) and posterior (2 time-steps) and innovations
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
    mape = MAPE(ensemble_aggregation="mean", time_aggregation="mean")
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
            "mape": mape(ensemble, true_sol),
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
    if cfg.case.num_states > 1:
        for i in range(cfg.case.num_states):
            metric_dict = get_metric_dict(
                posterior_ensemble[:, :, i, :], true_sol[0, :, i, :]
            )
            post_metric_states.append(metric_dict)

    print_metrics_table(
        reference_metrics, posterior_metrics, title=f"{cfg.case.title} Metrics"
    )

    # --- Plot Hovmöller diagrams and assim-time-series
    logger.info(f"Plotting...")
    plotter = instantiate(plotter_cfg)
    plotter(
        true_sol=true_sol,
        reference_ensemble=reference_ensemble,
        posterior_ensemble=posterior_ensemble,
        title=cfg.case.title,
        da_method_name=cfg.da_method.name,
        ensemble_size=cfg.ensemble_size,
        reference_metrics=reference_metrics,
        posterior_metrics=posterior_metrics,
        state_dim=cfg.case.state_dim,
        data_assimilation_steps=cfg.data_assimilation_steps,
        model_integration_steps=cfg.model_integration_steps,
    )

    # --- Plot time-series of errors
    post_metrics_all = [posterior_metrics] + post_metric_states
    _ = plot_metric_timeseries(post_metrics_all)
    # ====================== For EnKF application plot innovation statistics ======================
    if cfg.da_method["name"] == "enkf":

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

    # --- Plot Initial-Conditions (I.C, best-guess, Initial-Ensemble) + Breeding statitcs if available
    try:
        best_guess_profile = best_guess_profile[0]
    except:
        best_guess_profile = None
    try:
        bv_dict = ensgen_diagnostics
    except:
        bv_dict = None

    plot_initial_fields(
        ensemble=posterior_ensemble,
        truth_t0=true_sol,
        x_before_spinup=ic_ref[0],
        best_guess_profile=best_guess_profile,
        bv_dict=bv_dict,
    )


if __name__ == "__main__":
    main()
