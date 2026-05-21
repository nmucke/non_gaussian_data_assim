"""Plotting innovation statistics for DA experiments."""

from typing import List, Mapping, Optional, Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def plot_innov(
    innov_stats_list: List[dict],
    bins: int = 51,
    hist_range: None | tuple[float, float] = None,
    show_fig: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot innovation and posterior diagnostics"""

    colors = ["tab:blue", "tab:orange"]

    ncols = 2
    fig, axs = plt.subplots(
        nrows=1, ncols=ncols, figsize=(6 * ncols, 4), constrained_layout=True
    )

    for i, innov_stats_dict in enumerate(innov_stats_list):

        ax_hist = axs[0]
        ax_chi = axs[1]

        # Extract necessary Data
        z = innov_stats_dict.get("z")
        chi_sq = innov_stats_dict.get("chi_sq_time")
        chi_sq_mean = innov_stats_dict.get("chi_sq_mean")

        # Flatten normalized innovation vector
        z = np.asarray(z)
        z_flat = z.ravel()

        if hist_range is None:
            vmin = jnp.quantile(z_flat, 0.01)
            vmax = jnp.quantile(z_flat, 0.99)
            hist_range = (vmin, vmax)

        # --- Left Panel: normalized innovation histogram
        ax_hist.hist(
            z_flat,
            bins=bins,
            range=hist_range,
            density=True,
            alpha=0.75 - 0.3 * i,
            edgecolor="white",
            linewidth=0.6,
            color=colors[i],
        )

        x = np.linspace(hist_range[0], hist_range[1], 500)
        normal_pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

        gauss_label = r"$\mathcal{N}(0,1)$" if i == 0 else None
        ax_hist.plot(x, normal_pdf, c="k", linewidth=2, label=gauss_label)
        ax_hist.axvline(0, c="k", linewidth=1, alpha=0.75)
        ax_hist.set_xlim(hist_range)
        ax_hist.set_xlabel("Normalized innovation")
        ax_hist.set_ylabel("Relative frequncey")
        ax_hist.legend(frameon=False)

        stats = (
            f"State {i}\nN = {z.size}\n"
            f"avg  = {np.nanmean(z_flat):.3f}\n"
            f"std  = {np.nanstd(z_flat):.3f}\n"
            f"chi2 = {chi_sq_mean:.1f}"
        )

        ax_hist.text(
            0.03,
            0.95 - 0.3 * i,
            stats,
            transform=ax_hist.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "alpha": 0.95,
                "facecolor": "white",
                "edgecolor": colors[i],
            },
        )

        # --- Right Panel: chi-square
        if chi_sq is not None:
            steps_chi = np.arange(len(chi_sq))
            ax_chi.plot(steps_chi, chi_sq, linewidth=2, c=colors[i], label=f"State {i}")
            ax_chi.axhline(1, color="k", linestyle="--", linewidth=1.5, alpha=0.5)
            ax_chi.set_xlabel("Assimilation step")
            ax_chi.set_ylabel(r"$\chi^2$")
            ax_chi.grid(alpha=0.25)
            ax_chi.legend()
        else:
            ax_chi.axis("off")

        if i > 0:
            ax_chi.legend()

    if show_fig:
        plt.show()

    return fig, axs
