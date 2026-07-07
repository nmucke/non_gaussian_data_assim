"""Unified entrypoint for data-assimilation experiments, configured via Hydra.

Examples:
    python scripts/main.py case=lorenz_63 da_method=enkf
    python scripts/main.py case=lorenz_96 da_method=pff
    python scripts/main.py case=kuramoto da_method=agmf data_assimilation_steps=100

    # Re-run a previously saved experiment from its stored config:
    python scripts/main.py experiment=lorenz_63_enkf_M250_nsteps10_dtstep5
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
from non_gaussian_data_assim.metrics.information_metrics import (
    compute_information_metrics,
)
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
from non_gaussian_data_assim.observations.observation_utils import generate_observations
from non_gaussian_data_assim.plotting.plot_initial_ensemble import plot_initial_fields
from non_gaussian_data_assim.plotting.plot_innov_stats import plot_innov
from non_gaussian_data_assim.plotting.plot_metrics import plot_metric_timeseries
from non_gaussian_data_assim.utils.saving import ExperimentSaver, creat_exp_name
from non_gaussian_data_assim.utils.spinup import spinup_ensemble

logger = logging.getLogger(__name__)


@hydra_main(config_path="../configs", config_name="config", version_base=None)  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    # -- Optionally re-run a saved experiment from its stored config.
    if cfg.experiment is not None:
        saved_config = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / cfg.experiment
            / "config"
            / "config.yaml"
        )
        logger.info(f"Loading config from saved experiment: {saved_config}")
        cfg = OmegaConf.load(saved_config)

    print(OmegaConf.to_yaml(cfg))

    forward_model_cfg = cfg.case.forward_model
    true_initial_state_cfg = cfg.case.true_initial_state
    obs_operator_cfg = cfg.case.obs_operator
    val_obs_operator_cfg = cfg.case.val_obs_operator
    initial_ensemble_cfg = cfg.case.initial_ensemble
    plotter_cfg = cfg.case.plotter
    da_method_cfg = OmegaConf.merge(
        cfg.da_method, cfg.case.da_method_overrides[cfg.da_method.name]
    )

    # ------------------------------------------------------
    # --- Init saving settings
    if cfg.save.experiment:
        REPO_ROOT = Path(__file__).resolve().parents[1]
        experiment_folder = REPO_ROOT / "experiments"
        # -- Create unique experiment name
        save_name = creat_exp_name(config=cfg)

        infl_str = str(cfg.inflation_factor).replace(".", "_")
        save_name = f"{save_name}_inflf{infl_str}"

        # -- Create directory in ./experiments/ folder
        saver = ExperimentSaver.create(save_name, root=experiment_folder)
        # -- Save config
        saver.save_config(cfg)
        flag_savedata = True
        # -- Create path and folder for figures (optional)
        if cfg.save.figures:
            path_savefig = saver.subdir("figures")
    else:
        flag_savedata = False
        path_savefig = None
    # ------------------------------------------------------

    rng_key = jax.random.PRNGKey(cfg.seed)
    rng_key, initial_state_key, initial_ensemble_key, obs_key, val_key, info_key = (
        jax.random.split(rng_key, 6)
    )

    ########## Forward model ##########
    forward_model = instantiate(forward_model_cfg)
    logger.info(f"Forward model: {forward_model}")

    ########## True state ##########
    # Initial state
    initial_state_generator = instantiate(
        true_initial_state_cfg, forward_model=forward_model
    )
    # Get spun-up (t0) and before spin-up version of truth
    true_initial_state = initial_state_generator.sample(rng_key=initial_state_key)
    truth_before_spinup = initial_state_generator.state_before_spinup

    logger.info(f"Initial state generator: {initial_state_generator}")

    # Rollout true solution
    true_sol = forward_model.rollout(
        true_initial_state,
        cfg.data_assimilation_steps,
        return_model_integration_steps=True,
    )
    # Standard deviation of truth during spin-up
    natural_variability_truth = initial_state_generator.natural_variability

    ########## Initial ensemble ##########
    initial_ensemble_generator = instantiate(
        initial_ensemble_cfg, forward_model=forward_model
    )
    initial_ensemble = initial_ensemble_generator.sample(
        rng_key=initial_ensemble_key,
        ensemble_size=cfg.ensemble_size,
        best_guess_pert_scale=natural_variability_truth,
        true_state=true_initial_state if initial_ensemble_cfg.use_best_guess else None,
    )
    best_guess_profile = initial_ensemble_generator.best_guess_profile
    logger.info(f"Initial ensemble generator: {initial_ensemble_generator}")

    ########## Observation operator ##########
    obs_operator = instantiate(obs_operator_cfg)
    val_obs_operator = instantiate(val_obs_operator_cfg)
    R = jnp.eye(obs_operator.num_obs) * cfg.case.obs_noise_variance
    logger.info(f"Observation operator: {obs_operator}")

    # Observe true solution
    observations = generate_observations(
        rng_key=obs_key,
        true_sol=true_sol,
        obs_operator=obs_operator,
        R=R,
        data_assimilation_steps=cfg.data_assimilation_steps,
        model_integration_steps=cfg.model_integration_steps,
    )

    validation_info = generate_observations(
        rng_key=info_key,
        true_sol=true_sol,
        obs_operator=obs_operator,
        R=R,
        data_assimilation_steps=cfg.data_assimilation_steps,
        model_integration_steps=cfg.model_integration_steps,
    )

    R_val = jnp.eye(val_obs_operator.num_obs) * cfg.case.obs_noise_variance
    validation = generate_observations(
        rng_key=val_key,
        true_sol=true_sol,
        obs_operator=val_obs_operator,
        R=R_val,
        data_assimilation_steps=cfg.data_assimilation_steps,
        model_integration_steps=cfg.model_integration_steps,
    )

    ########## DA model ##########
    da_model = instantiate(
        da_method_cfg,
        ensemble_size=cfg.ensemble_size,
        R=R,
        obs_operator=obs_operator,
        forward_operator=forward_model,
    )
    logger.info(f"DA model: {da_model}")

    ########## Prepare ensembles ##########
    # Collect the trajectories in Python lists (one block per DA step) and
    # concatenate once after the loop; growing a device array with
    # jnp.concatenate every step is O(T^2) copying.
    posterior_blocks = [
        initial_ensemble.copy().reshape(
            cfg.ensemble_size, 1, cfg.case.num_states, cfg.case.state_dim
        )
    ]
    prior_blocks = [
        initial_ensemble.copy().reshape(
            cfg.ensemble_size, 1, cfg.case.num_states, cfg.case.state_dim
        )
    ]

    ########## Rollout the initial ensemble for comparison ##########
    reference_ensemble = forward_model.rollout(
        initial_ensemble,
        cfg.data_assimilation_steps,
        return_model_integration_steps=True,
    )

    ########## Run the DA loop. ##########
    # Initialize empty lists to track chi-square and normalized nnovations (z)
    predicted_prior_obs, validation_obs = [], []
    logger.info(f"Running DA loop for {cfg.data_assimilation_steps} steps")
    for i in tqdm(range(cfg.data_assimilation_steps)):
        rng_key, key = jax.random.split(rng_key)

        prior_current = posterior_blocks[-1][:, -1]

        posterior_next = da_model(
            prior_ensemble=prior_current,
            obs_vect=observations[i],
            rng_key=key,
            return_model_integration_steps=True,
        )
        if jnp.isnan(posterior_next).any():
            # A NaN means the filter diverged. Fail loudly here: silently
            # break-ing leaves the posterior trajectory shorter than the truth,
            # which later surfaces as a cryptic vmap shape mismatch in the
            # metrics rather than the actual cause.
            raise RuntimeError(
                f"Filter '{cfg.da_method['name']}' diverged: NaN in the posterior "
                f"at assimilation step {i} (of {cfg.data_assimilation_steps}). "
                "This usually means inflation is too strong or the step/"
                "regularization needs tuning for this case."
            )

        # ------------Track:  Forecast ensemble (in obs space) at the obs time -----------
        # The innovation diagnostics (chi^2, whitened innovations) must compare the
        # observation against the FORECAST valid AT the observation time -- the
        # ensemble propagated one full outer step from the previous analysis -- not
        # the pre-forecast analysis `prior_current`, which is `model_integration_steps`
        # inner steps stale (that misalignment made the primary spread-consistency
        # check meaningless). Propagate one outer step and take the last inner step;
        # this reproduces the forecast da_model forms internally before its analysis.
        forecast_at_obs = forward_model(
            prior_current, return_model_integration_steps=True, is_ensemble=True
        )[:, -1]
        HXf = da_model.obs_operator(forecast_at_obs)  # shape: (EnsSize, N_obs)
        HXf_val = val_obs_operator(forecast_at_obs)  # shape: (EnsSize, N_obs)
        predicted_prior_obs.append(HXf)
        validation_obs.append(HXf_val)
        # -----------------------------------------------------------------

        prior_blocks.append(prior_current[:, None, :, :])
        posterior_blocks.append(posterior_next)

    prior_ensemble_da = jnp.concatenate(prior_blocks, axis=1)
    posterior_ensemble = jnp.concatenate(posterior_blocks, axis=1)

    logger.info(f"Finished DA loop")

    forecast = OmegaConf.select(cfg, "forecast_steps", default=None)
    if forecast is not None:
        logger.info(f"\nForecast -- Run Ensemble for {forecast} steps")

        # --- Select last time-step of assim as start-point for forecasting
        posterior_ensemble_tend = posterior_ensemble[:, -1]
        # --- Forcasting has same logic as spin-up --> Model-runs without data-assim
        forecast_ensemble = spinup_ensemble(
            ensemble=posterior_ensemble_tend,
            forward_model=forward_model,
            spinup_steps=forecast,
            return_model_integration_steps=True,
        )

        true_forecast = spinup_ensemble(
            ensemble=true_sol[:, -1],
            forward_model=forward_model,
            spinup_steps=forecast,
            return_model_integration_steps=True,
        )

    else:
        forecast_ensemble = None
        true_forecast = None

    # Metrics for the forecast are only computed when a forecast was produced;
    # initialize here so the save path can reference it unconditionally.
    forecast_metrics = None

    # --- Extrac predicted observation (assimilation and valdiation points)
    pred_prior_obs = jnp.stack(
        predicted_prior_obs, axis=1
    )  # shape: [EnsSize, Assim-Step, N_obs (comined for all states)]
    pred_val_obs = jnp.stack(validation_obs, axis=1)

    predobs_states, obs_states, R_states = [], [], []
    idx = 0
    for obs_state in da_model.obs_operator.obs_indices_per_state:
        i = obs_state.shape[0]
        predobs_states.append(pred_prior_obs[:, :, idx : idx + i])
        obs_states.append(observations[:, idx : idx + i])
        R_states.append(R[idx : idx + i, idx : idx + i])
        idx += i

    predobs_val_states, val_obs_states, R_val_states = [], [], []
    idx = 0
    for valobs_state in val_obs_operator.obs_indices_per_state:
        i = valobs_state.shape[0]
        predobs_val_states.append(pred_val_obs[:, :, idx : idx + i])
        val_obs_states.append(validation[:, idx : idx + i])
        R_val_states.append(R_val[idx : idx + i, idx : idx + i])
        idx += i

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
    if forecast_ensemble is not None:
        forecast_metrics = get_metric_dict(forecast_ensemble, true_forecast[0])

    # --- Compute metrics of Assimilation / Validation Obs against predicted Obs
    assim_obs_metrics = get_metric_dict(predobs_states[0], obs_states[0])
    valid_obs_metrics = get_metric_dict(predobs_val_states[0], val_obs_states[0])

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

    if cfg.case.information_metrics.enabled:
        # ======================== information metrics ======================================
        # X  = open-loop/reference model prediction
        # Y  = assimilated obs
        # Z  = independent validation_info obs (NOT same as validation ops! HERE: same location as assim-obs, but with different obs-noise sample)
        # X+ = analysis/posterior model prediction
        #
        # We assume thwt Y and Z are aligned (spatially/temporally) samples:
        if observations.shape != validation_info.shape:
            raise ValueError(
                "Information metrics reuqire aligned & paired samples. "
                f"Got observations.shape={observations.shape} and "
                f"validation_info.shape={validation_info.shape}."
            )

        n_info_bins = cfg.case.information_metrics.n_bins
        bin_range = cfg.case.information_metrics.bin_range
        info_value_range = None if bin_range == "auto" else bin_range

        # --- Indices of the assimilation times in the full-resolution trajectory.
        # The analysis at DA step i lands at inner-step index (i + 1) * m, matching
        # how generate_observations samples the truth.
        obs_time_idx = cfg.model_integration_steps * (
            np.arange(cfg.data_assimilation_steps) + 1
        )

        X_openloop = []
        X_plus = []
        for t in obs_time_idx:
            X_avg = np.asarray(obs_operator(reference_ensemble[:, t])).mean(axis=0)
            Xp_avg = np.asarray(obs_operator(posterior_ensemble[:, t])).mean(axis=0)
            X_openloop.append(X_avg)
            X_plus.append(Xp_avg)
        x_model = np.stack(X_openloop, axis=0)
        x_analysis = np.stack(X_plus, axis=0)

        # x_model = np.stack([np.asarray(obs_operator(reference_ensemble[:, t])).mean(axis=0) for t in obs_time_idx], axis=0,)
        # x_analysis = np.stack([np.asarray(obs_operator(posterior_ensemble[:, t])).mean(axis=0) for t in obs_time_idx], axis=0,)

        information_metrics = compute_information_metrics(
            x_model=x_model,
            y_data=np.asarray(observations),
            z_eval=np.asarray(validation_info),
            x_analysis=x_analysis,
            n_bins=n_info_bins,
            value_range=info_value_range,
        )

        print()
        print("Information metrics:")
        for key in (
            "i_z_x",
            "i_z_x_analysis",
            "data_assimilation_efficiency",
            "data_efficiency",
        ):
            print(f"{key:3s}: {information_metrics[key]:.3f}")
    else:
        information_metrics = None

    # ==========================================================================================================================
    # ==========================================================================================================================

    # ======================== Plotting  ======================================
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
        path_savefig=path_savefig,
    )

    # --- Plot time-series of errors
    post_metrics_all = [posterior_metrics] + post_metric_states
    _ = plot_metric_timeseries(post_metrics_all, path_savefig=path_savefig)

    if forecast_ensemble is not None:
        _ = plot_metric_timeseries(
            [forecast_metrics], path_savefig=path_savefig, figname_appendix="forecast"
        )

    # --- Compute Innovation metrics
    if cfg.da_method["name"] == "enkf":
        innov_metric_list = []
        for p_obs, obs, R_state in zip(predobs_states, obs_states, R_states):
            innovation_metrics = {
                "chi_sq_mean": chi2_mean(predicted_obs=p_obs, obs=obs, R=R_state),
                "chi_sq_time": chi2_time(predicted_obs=p_obs, obs=obs, R=R_state),
                "z": innov_white(predicted_obs=p_obs, obs=obs, R=R_state),
            }
            innov_metric_list.append(innovation_metrics)

        # # --- Plot Innovations Diagnostics and Skill-Scores
        # _ = plot_innov(
        #     innov_stats_list=innov_metric_list,
        #     bins=74,
        #     hist_range=None,
        #     show_fig=True,
        #     path_savefig=path_savefig,
        # )
    else:
        innov_metric_list = None

    # --- Plot Initial-Conditions (I.C, best-guess, Initial-Ensemble) + Breeding statitcs if available
    try:
        best_guess_profile = best_guess_profile[0]
    except (TypeError, IndexError):
        best_guess_profile = None
    bv_dict = None

    # plot_initial_fields(
    #     ensemble=posterior_ensemble,
    #     truth_t0=true_sol,
    #     x_before_spinup=true_initial_state[0],
    #     best_guess_profile=best_guess_profile,
    #     bv_dict=bv_dict,
    # )

    # -------------------------------------------------------
    # -- Save fields
    if flag_savedata:
        arrays_to_save = {
            "truth_sol": true_sol,
            "reference_ensemble": reference_ensemble,
            "posterior_ensemble": posterior_ensemble,
        }

        if forecast_ensemble is not None:
            arrays_to_save["forecast_ensemble"] = forecast_ensemble

        metrics_to_save = {
            "reference_metrics": reference_metrics,
            "posterior_metrics": posterior_metrics,
            "assim_obs_metrics": assim_obs_metrics,
            "valid_obs_metrics": valid_obs_metrics,
        }

        if information_metrics is not None:
            metrics_to_save["information_metrics"] = information_metrics

        if innov_metric_list is not None:
            # Save innovation diagnostics for every observed state (not just the
            # last one). Single-state cases keep the plain "innovation_metrics"
            # name for backward compatibility.
            for j, innov in enumerate(innov_metric_list):
                name = (
                    "innovation_metrics"
                    if len(innov_metric_list) == 1
                    else f"innovation_metrics_state{j}"
                )
                metrics_to_save[name] = innov

        if forecast_metrics is not None:
            metrics_to_save["forecast_metrics"] = forecast_metrics

        # if bv_dict is not None:
        #     metrics_to_save["breeding_metrics"] = bv_dict

        # -- Save fields
        for key, value in arrays_to_save.items():
            saver.save_array(name=key, array=value)

        # -- Save metrics
        for key, value in metrics_to_save.items():
            saver.save_metrics(name=key, metrics=value)
    # -------------------------------------------------------


if __name__ == "__main__":
    main()
