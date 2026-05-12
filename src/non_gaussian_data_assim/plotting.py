"""Plotting helpers for DA experiments."""

from typing import Mapping, Optional, Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def plot_low_dim_trajectory(
    *,
    true_sol: jnp.ndarray,
    reference_ensemble: jnp.ndarray,
    posterior_ensemble: jnp.ndarray,
    title: str,
    da_method_name: str,
    ensemble_size: int,
    reference_metrics: Mapping[str, float],
    posterior_metrics: Mapping[str, float],
    state_dim: int,
    state_names: Sequence[str],
    data_assimilation_steps: int,
    model_integration_steps: int,
) -> None:
    """Per-state-dimension subplots for low-dimensional systems (e.g. Lorenz 63)."""
    true_sol_2d = true_sol.reshape(
        data_assimilation_steps * model_integration_steps + 1, state_dim
    )
    mean_reference = reference_ensemble.mean(axis=(0, 2))
    mean_post = posterior_ensemble.mean(axis=(0, 2))
    std_post = posterior_ensemble.std(axis=(0, 2))
    time_axis = np.arange(posterior_ensemble.shape[1])

    plt.figure()
    plt.suptitle(
        f"{title}, DA Method: {da_method_name}, Ensemble Size: {ensemble_size}, \n"
        f"Reference RMSE: {reference_metrics['rmse']:.4f}, "
        f"Posterior RMSE: {posterior_metrics['rmse']:.4f}"
    )
    for state_idx in range(state_dim):
        plt.subplot(state_dim, 1, state_idx + 1)
        for state_name, state_data, color in zip(
            ["Reference Ensemble Mean", "Posterior Ensemble Mean", "True Solution"],
            [mean_reference, mean_post, true_sol_2d],
            ["tab:blue", "tab:red", "black"],
        ):
            plt.plot(
                time_axis,
                state_data[:, state_idx],
                label=state_name,
                color=color,
                linewidth=3,
                linestyle="--" if state_name == "True Solution" else "-",
            )
        plt.fill_between(
            time_axis,
            mean_post[:, state_idx] - std_post[:, state_idx],
            mean_post[:, state_idx] + std_post[:, state_idx],
            color="tab:blue",
            alpha=0.2,
            label="Posterior ± Std",
        )
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel(state_names[state_idx])
        plt.ylim(true_sol_2d[:, state_idx].min(), true_sol_2d[:, state_idx].max())
    plt.show()


def plot_high_dim_field(
    *,
    true_sol: jnp.ndarray,
    reference_ensemble: jnp.ndarray,
    posterior_ensemble: jnp.ndarray,
    title: str,
    da_method_name: str,
    ensemble_size: int,
    reference_metrics: Mapping[str, float],
    posterior_metrics: Mapping[str, float],
    state_dim: int,
    data_assimilation_steps: int,
    model_integration_steps: int,
    state_names: Optional[Sequence[str]] = None,
) -> None:
    """Spatial-field heatmaps + per-point time series for high-dim systems (L96, KS)."""
    del state_names
    true_sol_2d = true_sol.reshape(
        data_assimilation_steps * model_integration_steps + 1, state_dim
    )
    ids_to_plot = [state_dim // 4, state_dim // 2, 3 * state_dim // 4]

    fields = [
        true_sol_2d,
        reference_ensemble.mean(axis=(0, 2)),
        posterior_ensemble.mean(axis=(0, 2)),
        true_sol_2d - reference_ensemble.mean(axis=(0, 2)),
        true_sol_2d - posterior_ensemble.mean(axis=(0, 2)),
        posterior_ensemble.var(axis=(0, 2)),
    ]
    field_names = [
        "True Solution",
        "Reference Ensemble Mean",
        "Posterior Ensemble Mean",
        "|True - Reference| difference",
        "|True - Posterior| difference",
        "Posterior Ensemble Variance",
    ]

    plt.figure()
    plt.suptitle(
        f"{title}, DA Method: {da_method_name}, Ensemble Size: {ensemble_size}, \n"
        f"Reference RMSE: {reference_metrics['rmse']:.4f}, "
        f"Posterior RMSE: {posterior_metrics['rmse']:.4f}"
    )

    for i, (field, field_name) in enumerate(zip(fields, field_names)):
        vmin = true_sol_2d.min() if i < 3 else np.percentile(field, 5)
        vmax = true_sol_2d.max() if i < 3 else np.percentile(field, 95)
        plt.subplot(3, 3, 1 + i)
        plt.imshow(
            field[-state_dim * 2 :],
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        plt.colorbar()
        plt.title(field_name)

    for i, idx_to_plot in enumerate(ids_to_plot):
        plt.subplot(3, 3, 7 + i)
        mean_post_pt = posterior_ensemble.mean(axis=(0, 2))[:, idx_to_plot]
        std_post_pt = posterior_ensemble.std(axis=(0, 2))[:, idx_to_plot]
        time_axis = np.arange(posterior_ensemble.shape[1])
        plt.fill_between(
            time_axis,
            mean_post_pt - std_post_pt,
            mean_post_pt + std_post_pt,
            color="tab:blue",
            alpha=0.2,
            label="Posterior ± Std",
        )
        for state_at_point, state_name, color in zip(
            [
                reference_ensemble.mean(axis=(0, 2))[:, idx_to_plot],
                posterior_ensemble.mean(axis=(0, 2))[:, idx_to_plot],
                true_sol_2d[:, idx_to_plot],
            ],
            ["Reference Ensemble Mean", "Posterior Ensemble Mean", "True Solution"],
            ["tab:red", "tab:blue", "black"],
        ):
            plt.plot(
                state_at_point,
                label=state_name,
                color=color,
                linewidth=3,
                linestyle="--" if state_name == "True Solution" else "-",
            )
        plt.legend()
        plt.xlabel("Time")
        plt.title(f"State at grid point {idx_to_plot}")
        plt.ylim(
            true_sol_2d[:, idx_to_plot].min()
            - np.abs(true_sol_2d[:, idx_to_plot].min()) * 0.2,
            true_sol_2d[:, idx_to_plot].max()
            + np.abs(true_sol_2d[:, idx_to_plot].max()) * 0.2,
        )
        plt.grid(True)
    plt.show()


def plot_da_diagnostics(
    z: jnp.ndarray,
    chi_sq: jnp.ndarray,
    chi_sq_mean: float,
    crps_time: jnp.ndarray,
    rmse_time: jnp.ndarray,
    spread_time: jnp.ndarray,
    bins: int = 51,
    hist_range: None | tuple[float, float] = None,
    show_fig: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot innovation and posterior diagnostics"""

    # Flatten normalized innovation vector
    z = np.asarray(z)
    z_flat = z.ravel()

    if hist_range is None:
        vmin = jnp.quantile(z_flat, 0.01)
        vmax = jnp.quantile(z_flat, 0.99)
        hist_range = (vmin, vmax)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), constrained_layout=True)
    ax_hist, ax_chi, ax_scores, ax_reserved = axs.ravel()

    # --- Panel A: normalized innovation histogram
    ax_hist.hist(
        z_flat,
        bins=bins,
        range=hist_range,
        density=True,
        alpha=0.5,
        edgecolor="white",
        linewidth=0.6,
        color="k",
    )
    x = np.linspace(hist_range[0], hist_range[1], 500)
    normal_pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
    ax_hist.plot(x, normal_pdf, c="k", linewidth=2, label=r"$\mathcal{N}(0,1)$")
    ax_hist.axvline(0, c="k", linewidth=1, alpha=0.6)
    ax_hist.set_xlim(hist_range)
    ax_hist.set_xlabel("Normalized innovation")
    ax_hist.set_ylabel("Relative frequncey")
    ax_hist.legend(frameon=False)

    stats = (
        f"N    = {z.size}\n"
        f"avg  = {np.nanmean(z_flat):.3f}\n"
        f"std  = {np.nanstd(z_flat):.3f}\n"
        f"chi2 = {chi_sq_mean:.1f}"
    )

    ax_hist.text(
        0.03,
        0.95,
        stats,
        transform=ax_hist.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "alpha": 0.5, "facecolor": "white"},
    )

    # --- Panel B: chi-square
    steps_chi = np.arange(len(chi_sq))
    ax_chi.plot(steps_chi, chi_sq, linewidth=1, c="k")
    ax_chi.axhline(
        1, color="k", linestyle="--", linewidth=1.5, alpha=0.5, label="Optimum"
    )
    ax_chi.set_xlabel("Assimilation step")
    ax_chi.set_ylabel(r"$\chi^2$")
    ax_chi.grid(alpha=0.25)
    ax_chi.legend()

    # --- Panel C: posterior skill metrics
    steps_post = np.arange(len(crps_time))
    ax_scores.plot(steps_post, crps_time, linewidth=2, label="CRPS", color="tab:blue")
    ax_scores.plot(steps_post, rmse_time, linewidth=2, label="RMSE", color="tab:orange")
    ax_scores.plot(
        steps_post,
        spread_time,
        linewidth=5,
        label="Ens.Spread",
        color="tab:orange",
        alpha=0.3,
    )
    ax_scores.set_xlabel("Model time-step")
    ax_scores.set_ylabel("Score")
    ax_scores.legend(frameon=False)
    ax_scores.grid(alpha=0.25)
    ax_scores.set_ylim(0, None)

    # --- Panel D
    ax_reserved.axis("off")

    if show_fig:
        plt.show()

    return fig, axs


def plot_multi_state_high_dim_field(
    *,
    true_sol: jnp.ndarray,
    reference_ensemble: jnp.ndarray,
    posterior_ensemble: jnp.ndarray,
    title: str,
    da_method_name: str,
    ensemble_size: int,
    reference_metrics: Mapping[str, float],
    posterior_metrics: Mapping[str, float],
    state_dim: int,
    data_assimilation_steps: int,
    model_integration_steps: int,
    state_names: Optional[Sequence[str]] = None,
) -> None:
    """Multi-state version of `plot_high_dim_field`.

    Expects arrays shaped `[*, time, num_states, state_dim]` and produces one figure
    per entry along the `num_states` axis.
    """
    num_states = true_sol.shape[-2]
    names = (
        list(state_names)
        if state_names is not None
        else [f"State {s}" for s in range(num_states)]
    )

    for state_idx in range(num_states):
        plot_high_dim_field(
            true_sol=true_sol[:, :, state_idx : state_idx + 1],
            reference_ensemble=reference_ensemble[:, :, state_idx : state_idx + 1],
            posterior_ensemble=posterior_ensemble[:, :, state_idx : state_idx + 1],
            title=f"{title} — {names[state_idx]}",
            da_method_name=da_method_name,
            ensemble_size=ensemble_size,
            reference_metrics=reference_metrics,
            posterior_metrics=posterior_metrics,
            state_dim=state_dim,
            data_assimilation_steps=data_assimilation_steps,
            model_integration_steps=model_integration_steps,
        )


# def plot_starting_conditions(
#     *,
#     true_sol: jnp.ndarray,
#     ic: jnp.ndarray,
#     prior_ensemble: jnp.ndarray,
#     best_guess: Optional[jnp.ndarray] = None,
#     ic_spinup: Optional[jnp.ndarray] = None,
# ) -> tuple[plt.Figure, plt.Axes]:
#     pass
