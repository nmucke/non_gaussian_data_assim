"""Convergence comparison: PFF with KL divergence vs Hellinger distance.

Uses the same test case as da_benjamin_A.py (testcaseA_det.mat).
Benjamin samples are treated as the ground truth posterior.

Convergence is tested with respect to:
  1. Ensemble size
  2. Number of pseudo time steps
  3. Step size (pseudo time horizon = num_steps * step_size)
"""

import itertools
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.stats import gaussian_kde
from scipy.io import loadmat
from scipy.stats import wasserstein_distance

from non_gaussian_data_assim.da_methods.pff import ParticleFlowFilter
from non_gaussian_data_assim.forward_models.identity import IdentityModel
from non_gaussian_data_assim.observation_operator import SineObservationOperatorNoError

jax.config.update("jax_disable_jit", False)
jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
NUM_STATES = 1
STATE_DIM = 1
OBS_IDS = (0,)
OBS_STATES = (0,)
R = jnp.eye(len(OBS_IDS)) * 1.0
OBSERVATIONS = jnp.array([0.0])

OUTPUT_DIR = Path("figures/pff_convergence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reference (ground truth) samples from Benjamin
# ---------------------------------------------------------------------------
_benjamin = loadmat("benjamin_case/testcaseA_det.mat")
BENJAMIN_POSTERIOR = _benjamin["X_PFF"][:, 0, -1]  # (200,)
BENJAMIN_PRIOR = _benjamin["X_PFF"][:, 0, 0]  # (200,)

# KDE grid for visual comparison
X_GRID = np.linspace(-2.5, 5.0, 500)
BENJAMIN_KDE = gaussian_kde(BENJAMIN_POSTERIOR)
BENJAMIN_PDF = BENJAMIN_KDE.pdf(X_GRID)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def kde_l2_error(samples: np.ndarray, ref_pdf: np.ndarray, x_grid: np.ndarray) -> float:
    """L2 error between KDE of samples and reference PDF on a grid."""
    kde = gaussian_kde(samples)
    pdf = kde.pdf(x_grid)
    dx = x_grid[1] - x_grid[0]
    return float(np.sqrt(np.sum((pdf - ref_pdf) ** 2) * dx))


def w1_distance(samples: np.ndarray, ref_samples: np.ndarray) -> float:
    """Wasserstein-1 distance between sample sets."""
    return float(wasserstein_distance(np.asarray(samples), np.asarray(ref_samples)))


# ---------------------------------------------------------------------------
# PFF runner
# ---------------------------------------------------------------------------
def run_pff(
    divergence_type: str,
    ensemble_size: int,
    num_pseudo_time_steps: int,
    step_size: float,
    prior_ensemble: jnp.ndarray,
) -> np.ndarray:
    """Run PFF and return posterior samples as 1-D numpy array."""
    forward_model = IdentityModel(state_dim=STATE_DIM)
    obs_operator = SineObservationOperatorNoError(
        obs_states=OBS_STATES, obs_indices=OBS_IDS, state_dim=STATE_DIM
    )

    pff_kwargs = dict(
        ensemble_size=ensemble_size,
        R=R,
        obs_operator=obs_operator,
        forward_operator=forward_model,
        num_pseudo_time_steps=num_pseudo_time_steps,
        step_size=step_size,
        stepper="forward_euler",
        return_pff_trajectory=True,
        divergence_type=divergence_type,
    )
    if divergence_type == "hellinger":
        pff_kwargs["weight_estimation"] = "gaussian"
        pff_kwargs["hellinger_cov_regularization"] = 1e-6

    da_model = ParticleFlowFilter(**pff_kwargs)

    ens = prior_ensemble[:ensemble_size].reshape(ensemble_size, 1, NUM_STATES, STATE_DIM)
    posterior = da_model(
        prior_ensemble=ens[:, -1],
        obs_vect=OBSERVATIONS,
        return_inner_steps=False,
        prior_mean=jnp.ones(1),
        prior_cov=jnp.eye(1),
    )
    return np.asarray(posterior[:, -1, 0, 0])


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------
def _get_prior(ensemble_size: int, rng_key: jax.Array) -> jnp.ndarray:
    """Resample prior to requested size by bootstrap from Benjamin prior."""
    indices = jax.random.choice(rng_key, len(BENJAMIN_PRIOR), shape=(ensemble_size,), replace=True)
    return jnp.array(BENJAMIN_PRIOR)[indices]


def run_experiment(
    divergence_type: str,
    ensemble_sizes: list[int],
    num_steps_list: list[int],
    step_sizes: list[float],
    rng_key: jax.Array,
) -> dict:
    """Run PFF for a grid of parameters. Returns dict of results."""
    results = {}
    for N, nsteps, dt in itertools.product(ensemble_sizes, num_steps_list, step_sizes):
        key, rng_key = jax.random.split(rng_key)
        prior = _get_prior(N, key)
        t0 = time.time()
        posterior = run_pff(divergence_type, N, nsteps, dt, prior)
        elapsed = time.time() - t0
        l2 = kde_l2_error(posterior, BENJAMIN_PDF, X_GRID)
        w1 = w1_distance(posterior, BENJAMIN_POSTERIOR)
        results[(N, nsteps, dt)] = dict(
            posterior=posterior, l2=l2, w1=w1, time=elapsed
        )
        print(
            f"  {divergence_type:>10s} | N={N:5d} | steps={nsteps:6d} | dt={dt:.4f} "
            f"| L2={l2:.4e} | W1={w1:.4e} | {elapsed:.1f}s"
        )
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_convergence_ensemble_size(
    kl_results: dict,
    hell_results: dict,
    ensemble_sizes: list[int],
    fixed_nsteps: int,
    fixed_dt: float,
):
    """Convergence vs ensemble size."""
    kl_l2 = [kl_results[(N, fixed_nsteps, fixed_dt)]["l2"] for N in ensemble_sizes]
    kl_w1 = [kl_results[(N, fixed_nsteps, fixed_dt)]["w1"] for N in ensemble_sizes]
    hell_l2 = [hell_results[(N, fixed_nsteps, fixed_dt)]["l2"] for N in ensemble_sizes]
    hell_w1 = [hell_results[(N, fixed_nsteps, fixed_dt)]["w1"] for N in ensemble_sizes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(ensemble_sizes, kl_l2, "o-", label="KL", linewidth=2)
    axes[0].plot(ensemble_sizes, hell_l2, "s--", label="Hellinger", linewidth=2)
    axes[0].set_xlabel("Ensemble size")
    axes[0].set_ylabel("KDE L2 error")
    axes[0].set_title(f"L2 error vs Ensemble size\n(steps={fixed_nsteps}, dt={fixed_dt})")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")

    axes[1].plot(ensemble_sizes, kl_w1, "o-", label="KL", linewidth=2)
    axes[1].plot(ensemble_sizes, hell_w1, "s--", label="Hellinger", linewidth=2)
    axes[1].set_xlabel("Ensemble size")
    axes[1].set_ylabel("Wasserstein-1 distance")
    axes[1].set_title(f"W1 distance vs Ensemble size\n(steps={fixed_nsteps}, dt={fixed_dt})")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "convergence_ensemble_size.png", dpi=150)
    plt.close(fig)


def plot_convergence_num_steps(
    kl_results: dict,
    hell_results: dict,
    num_steps_list: list[int],
    fixed_N: int,
    fixed_dt: float,
):
    """Convergence vs number of pseudo time steps."""
    kl_l2 = [kl_results[(fixed_N, nsteps, fixed_dt)]["l2"] for nsteps in num_steps_list]
    kl_w1 = [kl_results[(fixed_N, nsteps, fixed_dt)]["w1"] for nsteps in num_steps_list]
    hell_l2 = [hell_results[(fixed_N, nsteps, fixed_dt)]["l2"] for nsteps in num_steps_list]
    hell_w1 = [hell_results[(fixed_N, nsteps, fixed_dt)]["w1"] for nsteps in num_steps_list]

    # Pseudo time horizon
    horizons = [nsteps * fixed_dt for nsteps in num_steps_list]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(horizons, kl_l2, "o-", label="KL", linewidth=2)
    ax.plot(horizons, hell_l2, "s--", label="Hellinger", linewidth=2)
    ax.set_xlabel("Pseudo time horizon (T = steps × dt)")
    ax.set_ylabel("KDE L2 error")
    ax.set_title(f"L2 error vs Pseudo time horizon\n(N={fixed_N}, dt={fixed_dt})")
    ax.legend()
    ax.grid(True)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax = axes[1]
    ax.plot(horizons, kl_w1, "o-", label="KL", linewidth=2)
    ax.plot(horizons, hell_w1, "s--", label="Hellinger", linewidth=2)
    ax.set_xlabel("Pseudo time horizon (T = steps × dt)")
    ax.set_ylabel("Wasserstein-1 distance")
    ax.set_title(f"W1 distance vs Pseudo time horizon\n(N={fixed_N}, dt={fixed_dt})")
    ax.legend()
    ax.grid(True)
    ax.set_xscale("log")
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "convergence_num_steps.png", dpi=150)
    plt.close(fig)


def plot_convergence_step_size(
    kl_results: dict,
    hell_results: dict,
    step_sizes: list[float],
    fixed_N: int,
    fixed_nsteps: int,
):
    """Convergence vs step size (varying pseudo time horizon T = fixed_nsteps * dt)."""
    kl_l2 = [kl_results[(fixed_N, fixed_nsteps, dt)]["l2"] for dt in step_sizes]
    kl_w1 = [kl_results[(fixed_N, fixed_nsteps, dt)]["w1"] for dt in step_sizes]
    hell_l2 = [hell_results[(fixed_N, fixed_nsteps, dt)]["l2"] for dt in step_sizes]
    hell_w1 = [hell_results[(fixed_N, fixed_nsteps, dt)]["w1"] for dt in step_sizes]

    horizons = [fixed_nsteps * dt for dt in step_sizes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(horizons, kl_l2, "o-", label="KL", linewidth=2)
    ax.plot(horizons, hell_l2, "s--", label="Hellinger", linewidth=2)
    ax.set_xlabel(f"Pseudo time horizon (T = {fixed_nsteps} × dt)")
    ax.set_ylabel("KDE L2 error")
    ax.set_title(f"L2 error vs Step size / Horizon\n(N={fixed_N}, steps={fixed_nsteps})")
    ax.legend()
    ax.grid(True)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim()[0] / fixed_nsteps, ax.get_xlim()[1] / fixed_nsteps)
    ax2.set_xlabel("Step size (dt)")

    ax = axes[1]
    ax.plot(horizons, kl_w1, "o-", label="KL", linewidth=2)
    ax.plot(horizons, hell_w1, "s--", label="Hellinger", linewidth=2)
    ax.set_xlabel(f"Pseudo time horizon (T = {fixed_nsteps} × dt)")
    ax.set_ylabel("Wasserstein-1 distance")
    ax.set_title(f"W1 distance vs Step size / Horizon\n(N={fixed_N}, steps={fixed_nsteps})")
    ax.legend()
    ax.grid(True)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim()[0] / fixed_nsteps, ax.get_xlim()[1] / fixed_nsteps)
    ax2.set_xlabel("Step size (dt)")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "convergence_step_size.png", dpi=150)
    plt.close(fig)


def plot_posterior_comparison(
    kl_results: dict,
    hell_results: dict,
    ensemble_sizes: list[int],
    fixed_nsteps: int,
    fixed_dt: float,
):
    """Compare posterior KDEs for selected ensemble sizes."""
    n_panels = len(ensemble_sizes)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, N in zip(axes, ensemble_sizes):
        kl_post = kl_results[(N, fixed_nsteps, fixed_dt)]["posterior"]
        hell_post = hell_results[(N, fixed_nsteps, fixed_dt)]["posterior"]

        ax.plot(X_GRID, BENJAMIN_PDF, "k-", linewidth=2, label="Reference")
        ax.plot(X_GRID, gaussian_kde(kl_post).pdf(X_GRID), "-", linewidth=2, label="KL")
        ax.plot(X_GRID, gaussian_kde(hell_post).pdf(X_GRID), "--", linewidth=2, label="Hellinger")
        ax.set_title(f"N = {N}")
        ax.set_xlabel("x")
        ax.legend()
        ax.grid(True)
        ax.set_xlim(X_GRID.min(), X_GRID.max())

    axes[0].set_ylabel("p(x|y)")
    fig.suptitle(
        f"Posterior comparison (steps={fixed_nsteps}, dt={fixed_dt})", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "posterior_comparison_ensemble.png", dpi=150)
    plt.close(fig)


def plot_posterior_comparison_steps(
    kl_results: dict,
    hell_results: dict,
    num_steps_list: list[int],
    fixed_N: int,
    fixed_dt: float,
):
    """Compare posterior KDEs for selected pseudo time step counts."""
    n_panels = min(len(num_steps_list), 4)
    indices = np.linspace(0, len(num_steps_list) - 1, n_panels, dtype=int)
    selected_steps = [num_steps_list[i] for i in indices]

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, nsteps in zip(axes, selected_steps):
        kl_post = kl_results[(fixed_N, nsteps, fixed_dt)]["posterior"]
        hell_post = hell_results[(fixed_N, nsteps, fixed_dt)]["posterior"]
        T = nsteps * fixed_dt

        ax.plot(X_GRID, BENJAMIN_PDF, "k-", linewidth=2, label="Reference")
        ax.plot(X_GRID, gaussian_kde(kl_post).pdf(X_GRID), "-", linewidth=2, label="KL")
        ax.plot(X_GRID, gaussian_kde(hell_post).pdf(X_GRID), "--", linewidth=2, label="Hellinger")
        ax.set_title(f"steps={nsteps}, T={T:.1f}")
        ax.set_xlabel("x")
        ax.legend()
        ax.grid(True)
        ax.set_xlim(X_GRID.min(), X_GRID.max())

    axes[0].set_ylabel("p(x|y)")
    fig.suptitle(
        f"Posterior comparison (N={fixed_N}, dt={fixed_dt})", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "posterior_comparison_steps.png", dpi=150)
    plt.close(fig)


def plot_timing_comparison(
    kl_results: dict,
    hell_results: dict,
    ensemble_sizes: list[int],
    fixed_nsteps: int,
    fixed_dt: float,
):
    """Wall-clock time comparison."""
    kl_times = [kl_results[(N, fixed_nsteps, fixed_dt)]["time"] for N in ensemble_sizes]
    hell_times = [hell_results[(N, fixed_nsteps, fixed_dt)]["time"] for N in ensemble_sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ensemble_sizes, kl_times, "o-", label="KL", linewidth=2)
    ax.plot(ensemble_sizes, hell_times, "s--", label="Hellinger", linewidth=2)
    ax.set_xlabel("Ensemble size")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(f"Runtime vs Ensemble size\n(steps={fixed_nsteps}, dt={fixed_dt})")
    ax.legend()
    ax.grid(True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "timing_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng_key = jax.random.PRNGKey(SEED)

    # -----------------------------------------------------------------------
    # Experiment 1: Convergence with ensemble size
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("Experiment 1: Convergence with ensemble size")
    print("=" * 70)

    ensemble_sizes = [50, 100, 200, 500]
    fixed_nsteps_ens = 10000
    fixed_dt_ens = 0.01

    key1, key2, rng_key = jax.random.split(rng_key, 3)
    kl_ens = run_experiment("kl", ensemble_sizes, [fixed_nsteps_ens], [fixed_dt_ens], key1)
    hell_ens = run_experiment("hellinger", ensemble_sizes, [fixed_nsteps_ens], [fixed_dt_ens], key2)

    plot_convergence_ensemble_size(kl_ens, hell_ens, ensemble_sizes, fixed_nsteps_ens, fixed_dt_ens)
    plot_posterior_comparison(kl_ens, hell_ens, ensemble_sizes, fixed_nsteps_ens, fixed_dt_ens)
    plot_timing_comparison(kl_ens, hell_ens, ensemble_sizes, fixed_nsteps_ens, fixed_dt_ens)

    # -----------------------------------------------------------------------
    # Experiment 2: Convergence with number of pseudo time steps
    #   (fixed dt → varying horizon T)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Experiment 2: Convergence with number of pseudo time steps")
    print("=" * 70)

    fixed_N_steps = 200
    fixed_dt_steps = 0.01
    num_steps_list = [10, 50, 100, 500, 1000, 5000, 10000]

    key1, key2, rng_key = jax.random.split(rng_key, 3)
    kl_steps = run_experiment("kl", [fixed_N_steps], num_steps_list, [fixed_dt_steps], key1)
    hell_steps = run_experiment("hellinger", [fixed_N_steps], num_steps_list, [fixed_dt_steps], key2)

    plot_convergence_num_steps(kl_steps, hell_steps, num_steps_list, fixed_N_steps, fixed_dt_steps)
    plot_posterior_comparison_steps(kl_steps, hell_steps, num_steps_list, fixed_N_steps, fixed_dt_steps)

    # -----------------------------------------------------------------------
    # Experiment 3: Convergence with step size
    #   (fixed num_steps → varying horizon T via dt)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Experiment 3: Convergence with step size (pseudo time horizon)")
    print("=" * 70)

    fixed_N_dt = 200
    fixed_nsteps_dt = 1000
    step_sizes = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    key1, key2, rng_key = jax.random.split(rng_key, 3)
    kl_dt = run_experiment("kl", [fixed_N_dt], [fixed_nsteps_dt], step_sizes, key1)
    hell_dt = run_experiment("hellinger", [fixed_N_dt], [fixed_nsteps_dt], step_sizes, key2)

    plot_convergence_step_size(kl_dt, hell_dt, step_sizes, fixed_N_dt, fixed_nsteps_dt)

    # -----------------------------------------------------------------------
    # Summary figure: 2×3 grid
    # -----------------------------------------------------------------------
    print("\nGenerating summary figure...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 0: L2 error
    # Row 1: W1 distance

    # Col 0: vs ensemble size
    for row, metric in enumerate(["l2", "w1"]):
        ax = axes[row, 0]
        kl_vals = [kl_ens[(N, fixed_nsteps_ens, fixed_dt_ens)][metric] for N in ensemble_sizes]
        hell_vals = [hell_ens[(N, fixed_nsteps_ens, fixed_dt_ens)][metric] for N in ensemble_sizes]
        ax.plot(ensemble_sizes, kl_vals, "o-", label="KL", linewidth=2)
        ax.plot(ensemble_sizes, hell_vals, "s--", label="Hellinger", linewidth=2)
        ax.set_xlabel("Ensemble size")
        ax.set_ylabel("KDE L2 error" if metric == "l2" else "Wasserstein-1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True)

    # Col 1: vs num pseudo time steps (as horizon)
    horizons = [n * fixed_dt_steps for n in num_steps_list]
    for row, metric in enumerate(["l2", "w1"]):
        ax = axes[row, 1]
        kl_vals = [kl_steps[(fixed_N_steps, n, fixed_dt_steps)][metric] for n in num_steps_list]
        hell_vals = [hell_steps[(fixed_N_steps, n, fixed_dt_steps)][metric] for n in num_steps_list]
        ax.plot(horizons, kl_vals, "o-", label="KL", linewidth=2)
        ax.plot(horizons, hell_vals, "s--", label="Hellinger", linewidth=2)
        ax.set_xlabel("Pseudo time horizon T")
        ax.set_ylabel("KDE L2 error" if metric == "l2" else "Wasserstein-1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True)

    # Col 2: vs step size
    dt_horizons = [fixed_nsteps_dt * dt for dt in step_sizes]
    for row, metric in enumerate(["l2", "w1"]):
        ax = axes[row, 2]
        kl_vals = [kl_dt[(fixed_N_dt, fixed_nsteps_dt, dt)][metric] for dt in step_sizes]
        hell_vals = [hell_dt[(fixed_N_dt, fixed_nsteps_dt, dt)][metric] for dt in step_sizes]
        ax.plot(dt_horizons, kl_vals, "o-", label="KL", linewidth=2)
        ax.plot(dt_horizons, hell_vals, "s--", label="Hellinger", linewidth=2)
        ax.set_xlabel(f"Pseudo time horizon (steps={fixed_nsteps_dt} × dt)")
        ax.set_ylabel("KDE L2 error" if metric == "l2" else "Wasserstein-1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True)

    axes[0, 0].set_title("vs Ensemble size")
    axes[0, 1].set_title("vs Num pseudo time steps")
    axes[0, 2].set_title("vs Step size")

    fig.suptitle("PFF Convergence: KL vs Hellinger", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "summary_convergence.png", dpi=150)
    plt.close(fig)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
