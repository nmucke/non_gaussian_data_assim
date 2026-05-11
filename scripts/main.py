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
    prior_ensemble_cfg = cfg.case.prior_ensemble
    plotter_cfg = cfg.case.plotter
    da_method_cfg = OmegaConf.merge(
        cfg.da_method, cfg.case.da_method_overrides[cfg.da_method.name]
    )

    rng_key = jax.random.PRNGKey(cfg.seed)

    # Forward model and observation operator.
    forward_model = instantiate(forward_model_cfg)
    logger.info(f"Forward model: {forward_model}")

    # Observation operator.
    obs_operator = instantiate(obs_operator_cfg)
    logger.info(f"Observation operator: {obs_operator}")

    # Initial state for the truth.
    true_initial_state_profile = instantiate(true_initial_state_cfg)
    logger.info(f"Initial state: {true_initial_state_profile}")
    rng_key, key = jax.random.split(rng_key)
    true_sol = true_initial_state_profile.sample(rng_key=key, ensemble_size=1)

    # Spin-Up truth Run (optional)
    if cfg.spinup_steps:
        true_sol = spinup_ensemble(
            ensemble=true_sol,
            forward_model=forward_model,
            spinup_steps=cfg.spinup_steps,
        )

    # ----------- BEST-GUESS: Reference state for ensemble members --> determines how difficulat assimilation task will be
    rng_key, bg_key = jax.random.split(rng_key)
    perturbs_best_guess = true_initial_state_profile.sample(
        rng_key=bg_key, ensemble_size=1
    )
    # NOTE: Just a design choice to add noise of with 10% scale compared to sclae of I.C. of truth
    best_guess = true_sol + perturbs_best_guess / 10
    # NOTE: Maybe 'better' way is to take some 'climatological' scale --> E.g. Bulk variance of spin-up

    # Rollout the truth.
    true_sol = forward_model.rollout(
        true_sol, cfg.data_assimilation_steps, return_model_integration_steps=True
    )

    # Observation noise covariance.
    R = jnp.eye(obs_operator.num_obs) * cfg.case.obs_noise_variance
    logger.info(f"Observation noise variance: {cfg.case.obs_noise_variance}")

    # Generate observations from the truth.
    rng_key, obs_key = jax.random.split(rng_key)
    observations = generate_observations(
        rng_key=obs_key,
        true_sol=true_sol,
        obs_operator=obs_operator,
        R=R,
        data_assimilation_steps=cfg.data_assimilation_steps,
        model_integration_steps=cfg.model_integration_steps,
    )

    # DA method (with case-specific overrides applied).
    da_model = instantiate(
        da_method_cfg,
        ensemble_size=cfg.ensemble_size,
        R=R,
        obs_operator=obs_operator,
        forward_operator=forward_model,
    )
    logger.info(f"DA method: {da_model}")

    # Prior ensemble.
    prior_ensemble_generator = instantiate(prior_ensemble_cfg)
    logger.info(f"Prior ensemble: {prior_ensemble_generator}")
    rng_key, key = jax.random.split(rng_key)
    reference_ensemble = prior_ensemble_generator.sample(
        rng_key=key, ensemble_size=cfg.ensemble_size, profile_bg=best_guess
    )

    # !!!!!!!!
    # # NOTE: Not necessary?? --> Since we use spun-up version of truth as best-guess
    # if cfg.spinup_steps:
    #     reference_ensemble = spinup_ensemble(
    #         ensemble=reference_ensemble,
    #         forward_model=forward_model,
    #         spinup_steps=cfg.spinup_steps,
    #     )

    # Initialize the posterior ensemble from the prior.
    posterior_ensemble = reference_ensemble.copy().reshape(
        cfg.ensemble_size, 1, cfg.case.num_states, cfg.case.state_dim
    )
    # Initialize the prior ensemble from the reference. NOTE: Prior ensemble only stores fields at assim-steps before assimilation is done
    prior_ensemble_da = reference_ensemble.copy().reshape(
        cfg.ensemble_size,
        1,
        cfg.case.num_states,
        cfg.case.state_dim,
    )

    # Initialize empty lists to track chi-square and normalized nnovations (z)
    predicted_obs = []

    # Rollout the prior ensemble for comparison.
    reference_ensemble = forward_model.rollout(
        reference_ensemble,
        cfg.data_assimilation_steps,
        return_model_integration_steps=True,
    )

    # Run the DA loop.
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

    predicted_obs = jnp.stack(
        predicted_obs, axis=0
    )  # shape: (Assim-Step, N_obs, EnsSize)

    logger.info(f"Finished DA loop")

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
        "crps_time": crps_time(reference_ensemble, true_sol[0]),
    }
    posterior_metrics = {
        "rmse": rmse(posterior_ensemble, true_sol[0]),
        "mae": mae(posterior_ensemble, true_sol[0]),
        "mape": mape(posterior_ensemble, true_sol[0]),
        "crps": crps(posterior_ensemble, true_sol[0]),
        "rmse_time": rmse_time(posterior_ensemble, true_sol[0]),
        "crps_time": crps_time(posterior_ensemble, true_sol[0]),
    }

    innovation_metrics = {
        "chi_sq_mean": chi2_mean(predicted_obs=predicted_obs, obs=observations, R=R),
        "chi_sq_time": chi2_time(predicted_obs=predicted_obs, obs=observations, R=R),
        "z": innov_white(predicted_obs=predicted_obs, obs=observations, R=R),
    }

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

    fig, axs = plot_da_diagnostics(
        z=innovation_metrics["z"],
        chi_sq=innovation_metrics["chi_sq_time"],
        chi_sq_mean=innovation_metrics["chi_sq_mean"],
        crps_time=posterior_metrics["crps_time"],
        rmse_time=posterior_metrics["rmse_time"],
        bins=51,
        show_fig=True,
    )


if __name__ == "__main__":
    main()
