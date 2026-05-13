"""Unified entrypoint for data-assimilation experiments, configured via Hydra.

Examples:
    python scripts/main.py case=lorenz_63 da_method=enkf
    python scripts/main.py case=lorenz_96 da_method=pff
    python scripts/main.py case=kuramoto da_method=agmf data_assimilation_steps=100
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
from hydra import main as hydra_main
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

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
from non_gaussian_data_assim.plotting import plot_da_diagnostics
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

    ic_ref = true_initial_state_profile.sample(rng_key=key)

    # --- Spin-Up truth Run (optional)
    if cfg.spinup_steps:
        logger.info(
            f"Run Spin-up of {cfg.spinup_steps * cfg.model_integration_steps} model-steps"
        )
        true_sol = spinup_ensemble(
            ensemble=ic_ref,
            forward_model=forward_model,
            spinup_steps=cfg.spinup_steps,
        )
    else:
        logger.info("No SPINUP")
        true_sol = ic_ref

    # --- Rollout the truth.
    x0_truth = true_sol
    true_sol = forward_model.rollout(
        x0_truth, cfg.data_assimilation_steps, return_model_integration_steps=True
    )

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

    # --- BEST-GUESS: Reference state for ensemble members --> determines how difficulat assimilation task will be
    centered_around_bestguess = initial_ensemble_cfg.centered_around_bestguess
    if centered_around_bestguess:
        # NOTE: Just a design choice to add noise of with 10% scale compared to sclae of I.C. of truth
        # NOTE: Maybe 'better' way is to take some 'climatological' scale --> E.g. Bulk variance of spin-up
        BG_SCALE = 0.5  # Noise compared to truth

        rng_key, bg_key = jax.random.split(rng_key)
        perturbs_best_guess = true_initial_state_profile.sample(rng_key=bg_key)
        best_guess_profile = x0_truth + perturbs_best_guess * BG_SCALE

    else:
        best_guess_profile = None

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

    # --- Handle spin-up of ensemble (# TODO: Decide if we want to do this always or only when no best-guess is used!!)
    if not centered_around_bestguess and cfg.spinup_steps:
        logger.info(
            "\nSPIN-UP Initials Ensemble (as it is NOT centered around a best-guess profile)"
        )
        reference_ensemble = spinup_ensemble(
            ensemble=reference_ensemble,
            forward_model=forward_model,
            spinup_steps=cfg.spinup_steps,
        )

    # TODO: Here it would be enough to only rollout best-guess (if present)
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
        if cfg.da_method["name"] == "enkf":
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

    reference_metrics = {
        "rmse": rmse(reference_ensemble, true_sol[0]),
        "mae": mae(reference_ensemble, true_sol[0]),
        "mape": mape(reference_ensemble, true_sol[0]),
        "crps": crps(reference_ensemble, true_sol[0]),
        "rmse_time": rmse_time(reference_ensemble, true_sol[0]),
        "spread_time": ensemble_spread(reference_ensemble),
        "crps_time": crps_time(reference_ensemble, true_sol[0]),
    }
    posterior_metrics = {
        "rmse": rmse(posterior_ensemble, true_sol[0]),
        "mae": mae(posterior_ensemble, true_sol[0]),
        "mape": mape(posterior_ensemble, true_sol[0]),
        "crps": crps(posterior_ensemble, true_sol[0]),
        "rmse_time": rmse_time(posterior_ensemble, true_sol[0]),
        "spread_time": ensemble_spread(posterior_ensemble),
        "crps_time": crps_time(posterior_ensemble, true_sol[0]),
    }

    if cfg.da_method["name"] == "enkf":
        predicted_obs = jnp.stack(
            predicted_obs, axis=0
        )  # shape: (Assim-Step, N_obs, EnsSize)
        innovation_metrics = {
            "chi_sq_mean": chi2_mean(
                predicted_obs=predicted_obs, obs=observations, R=R
            ),
            "chi_sq_time": chi2_time(
                predicted_obs=predicted_obs, obs=observations, R=R
            ),
            "z": innov_white(predicted_obs=predicted_obs, obs=observations, R=R),
        }
    else:
        innovation_metrics = {}

    print_metrics_table(
        reference_metrics, posterior_metrics, title=f"{cfg.case.title} Metrics"
    )

    # Plot.
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

    # --- Plot Innovations Diagnostics and Skill-Scores
    if cfg.da_method["name"] == "enkf":
        fig, axs = plot_da_diagnostics(
            z=innovation_metrics.get("z", None),
            chi_sq=innovation_metrics.get("chi_sq_time", None),
            chi_sq_mean=innovation_metrics.get("chi_sq_mean", None),
            crps_time=posterior_metrics["crps_time"],
            rmse_time=posterior_metrics["rmse_time"],
            spread_time=posterior_metrics["spread_time"],
            bins=51,
            show_fig=True,
        )

    # --- Plot Initial-Conditions (I.C, best-guess, Initial-Ensemble) + Breeding statitcs if available

    # ==============================================================================================================


if __name__ == "__main__":
    main()
