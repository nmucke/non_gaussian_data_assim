"""Convergence comparison: PFF with KL divergence vs Hellinger distance (linear case).

Uses the same test case as da_linear.py: 2D Gaussian prior with linear observation
operator. The true posterior is computed analytically via the Kalman filter.

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
from scipy.stats import wasserstein_distance

from non_gaussian_data_assim.da_methods.pff import ParticleFlowFilter
from non_gaussian_data_assim.forward_models.identity import IdentityModel
from non_gaussian_data_assim.observation_operator import LinearObservationOperator

jax.config.update("jax_disable_jit", False)
jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
NUM_STATES = 1
STATE_DIM = 2

OBS_IDS = np.arange(0, STATE_DIM)
OBS_STATES = (0,)

R = jnp.eye(len(OBS_IDS))
PRIOR_MEAN = jnp.array([5.0, 5.0])
PRIOR_COV = jnp.eye(STATE_DIM)
OBSERVATIONS = jnp.array([1.0, 1.0])

# Analytic posterior via Kalman filter
OBS_MATRIX = jnp.eye(STATE_DIM)
K = PRIOR_COV @ OBS_MATRIX.T @ jnp.linalg.inv(OBS_MATRIX @ PRIOR_COV @ OBS_MATRIX.T + R)
TRUE_POST_MEAN = PRIOR_MEAN + K @ (OBSERVATIONS - OBS_MATRIX @ PRIOR_MEAN)
TRUE_POST_COV = (jnp.eye(STATE_DIM) - K @ OBS_MATRIX) @ PRIOR_COV

OUTPUT_DIR = Path("figures/pff_convergence_linear")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reference samples (large set from analytic posterior)
_ref_key = jax.random.PRNGKey(999)
REF_SAMPLES = jax.random.multivariate_normal(_ref_key, TRUE_POST_MEAN, TRUE_POST_COV, (10000,))

# KDE grid
X_RANGE = (-1, 7)
Y_RANGE = (-1, 7)
NBINS = 100


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def mean_error(samples: np.ndarray) -> float:
    """L2 error between sample mean and true posterior mean."""
    return float(np.linalg.norm(np.mean(samples, axis=0) - np.asarray(TRUE_POST_MEAN)))


def cov_error(samples: np.ndarray) -> float:
    """Frobenius norm error between sample covariance and true posterior covariance."""
    sample_cov = np.cov(samples.T)
    return float(np.linalg.norm(sample_cov - np.asarray(TRUE_POST_COV), "fro"))


def w1_distance_2d(samples: np.ndarray, ref_samples: np.ndarray) -> float:
    """Average of marginal Wasserstein-1 distances across dimensions."""
    dists = []
    for d in range(samples.shape[1]):
        dists.append(wasserstein_distance(samples[:, d], np.asarray(ref_samples[:, d])))
    return float(np.mean(dists))


def kde_2d_l2_error(samples: np.ndarray, ref_samples: np.ndarray) -> float:
    """L2 error between 2D KDEs evaluated on a grid."""
    xi, yi = np.mgrid[
        X_RANGE[0] : X_RANGE[1] : NBINS * 1j,
        Y_RANGE[0] : Y_RANGE[1] : NBINS * 1j,
    ]
    pts = np.vstack([xi.flatten(), yi.flatten()])

    kde_s = gaussian_kde(samples.T)
    kde_r = gaussian_kde(ref_samples.T)

    pdf_s = kde_s(pts).reshape(xi.shape)
    pdf_r = kde_r(pts).reshape(xi.shape)

    dx = (X_RANGE[1] - X_RANGE[0]) / NBINS
    dy = (Y_RANGE[1] - Y_RANGE[0]) / NBINS
    return float(np.sqrt(np.sum((pdf_s - pdf_r) ** 2) * dx * dy))


# ---------------------------------------------------------------------------
# PFF runner
# ---------------------------------------------------------------------------
def run_pff(
    divergence_type: str,
    ensemble_size: int,
    num_pseudo_time_steps: int,
    step_size: float,
    prior_ensemble: jnp.ndarray,
    rng_key: jax.Array,
) -> np.ndarray:
    """Run PFF and return posterior samples as (N, 2) numpy array."""
    forward_model = IdentityModel(state_dim=STATE_DIM)
    obs_operator = LinearObservationOperator(
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
        pff_kwargs["weight_estimation"] = "kde"
        pff_kwargs["kde_bandwidth"] = 0.1
        pff_kwargs["hellinger_cov_regularization"] = 1e-6

    da_model = ParticleFlowFilter(**pff_kwargs)

    ens = prior_ensemble[:ensemble_size].reshape(ensemble_size, 1, NUM_STATES, STATE_DIM)
    posterior = da_model(
        prior_ensemble=ens[:, -1],
        obs_vect=OBSERVATIONS,
        rng_key=rng_key,
        return_inner_steps=False,
    )
    # posterior shape: (ensemble, time_steps, num_states, state_dim)
    return np.asarray(posterior[:, -1, 0, :])  # (N, STATE_DIM)


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------
def generate_prior(ensemble_size: int, rng_key: jax.Array) -> jnp.ndarray:
    """Generate prior ensemble from N(PRIOR_MEAN, PRIOR_COV)."""
    return jax.random.multivariate_normal(
        rng_key, PRIOR_MEAN, PRIOR_COV, (ensemble_size,)
    )


def run_experiment(
    divergence_type: str,
    ensemble_sizes: list[int],
    num_steps_list: list[int],
    step_sizes: list[float],
    rng_key: jax.Array,
) -> dict:
    """Run PFF for a grid of parameters."""
    results = {}
    for N, nsteps, dt in itertools.product(ensemble_sizes, num_steps_list, step_sizes):
        key1, key2, rng_key = jax.random.split(rng_key, 3)
        prior = generate_prior(N, key1)
        t0 = time.time()
        posterior = run_pff(divergence_type, N, nsteps, dt, prior, key2)
        elapsed = time.time() - t0
        m_err = mean_error(posterior)
        c_err = cov_error(posterior)
        w1 = w1_distance_2d(posterior, REF_SAMPLES)
        results[(N, nsteps, dt)] = dict(
            posterior=posterior, mean_err=m_err, cov_err=c_err, w1=w1, time=elapsed
        )
        print(
            f"  {divergence_type:>10s} | N={N:5d} | steps={nsteps:6d} | dt={dt:.4f} "
            f"| mean_err={m_err:.4e} | cov_err={c_err:.4e} | W1={w1:.4e} | {elapsed:.1f}s"
        )
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_convergence(
    kl_results: dict,
    hell_results: dict,
    x_values: list,
    key_fn,
    xlabel: str,
    title_suffix: str,
    filename: str,
    metrics: list[str] = None,
    metric_labels: dict = None,
):
    """Generic convergence plot."""
    if metrics is None:
        metrics = ["mean_err", "cov_err", "w1"]
    if metric_labels is None:
        metric_labels = {
            "mean_err": "Mean L2 error",
            "cov_err": "Cov Frobenius error",
            "w1": "Wasserstein-1",
        }

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        kl_vals = [kl_results[key_fn(v)][metric] for v in x_values]
        hell_vals = [hell_results[key_fn(v)][metric] for v in x_values]
        ax.plot(x_values, kl_vals, "o-", label="KL", linewidth=2)
        ax.plot(x_values, hell_vals, "s--", label="Hellinger", linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric_labels[metric])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True)

    fig.suptitle(f"PFF Convergence (Linear) — {title_suffix}", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)


def plot_2d_posterior_comparison(
    kl_results: dict,
    hell_results: dict,
    param_values: list,
    key_fn,
    label_fn,
    filename: str,
):
    """2D posterior KDE comparison panels."""
    n_panels = min(len(param_values), 4)
    indices = np.linspace(0, len(param_values) - 1, n_panels, dtype=int)
    selected = [param_values[i] for i in indices]

    fig, axes = plt.subplots(3, n_panels, figsize=(5 * n_panels, 14))
    if n_panels == 1:
        axes = axes.reshape(-1, 1)

    xi, yi = np.mgrid[
        X_RANGE[0] : X_RANGE[1] : NBINS * 1j,
        Y_RANGE[0] : Y_RANGE[1] : NBINS * 1j,
    ]
    pts = np.vstack([xi.flatten(), yi.flatten()])

    # Reference KDE
    ref_kde = gaussian_kde(np.asarray(REF_SAMPLES).T)
    ref_zi = ref_kde(pts).reshape(xi.shape)

    for col, val in enumerate(selected):
        key = key_fn(val)
        kl_post = kl_results[key]["posterior"]
        hell_post = hell_results[key]["posterior"]

        kl_zi = gaussian_kde(kl_post.T)(pts).reshape(xi.shape)
        hell_zi = gaussian_kde(hell_post.T)(pts).reshape(xi.shape)

        axes[0, col].pcolormesh(xi, yi, ref_zi, shading="gouraud")
        axes[0, col].set_title(f"True posterior")
        axes[1, col].pcolormesh(xi, yi, kl_zi, shading="gouraud")
        axes[1, col].set_title(f"KL — {label_fn(val)}")
        axes[2, col].pcolormesh(xi, yi, hell_zi, shading="gouraud")
        axes[2, col].set_title(f"Hellinger — {label_fn(val)}")

        for row in range(3):
            axes[row, col].set_xlim(*X_RANGE)
            axes[row, col].set_ylim(*Y_RANGE)
            axes[row, col].set_aspect("equal")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
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
    fixed_dt_ens = 0.1

    key1, key2, rng_key = jax.random.split(rng_key, 3)
    kl_ens = run_experiment("kl", ensemble_sizes, [fixed_nsteps_ens], [fixed_dt_ens], key1)
    hell_ens = run_experiment("hellinger", ensemble_sizes, [fixed_nsteps_ens], [fixed_dt_ens], key2)

    _plot_convergence(
        kl_ens, hell_ens, ensemble_sizes,
        key_fn=lambda N: (N, fixed_nsteps_ens, fixed_dt_ens),
        xlabel="Ensemble size",
        title_suffix=f"vs Ensemble size (steps={fixed_nsteps_ens}, dt={fixed_dt_ens})",
        filename="convergence_ensemble_size.png",
    )
    plot_2d_posterior_comparison(
        kl_ens, hell_ens, ensemble_sizes,
        key_fn=lambda N: (N, fixed_nsteps_ens, fixed_dt_ens),
        label_fn=lambda N: f"N={N}",
        filename="posterior_2d_ensemble.png",
    )

    # Timing comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    kl_t = [kl_ens[(N, fixed_nsteps_ens, fixed_dt_ens)]["time"] for N in ensemble_sizes]
    hell_t = [hell_ens[(N, fixed_nsteps_ens, fixed_dt_ens)]["time"] for N in ensemble_sizes]
    ax.plot(ensemble_sizes, kl_t, "o-", label="KL", linewidth=2)
    ax.plot(ensemble_sizes, hell_t, "s--", label="Hellinger", linewidth=2)
    ax.set_xlabel("Ensemble size")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Runtime vs Ensemble size (Linear case)")
    ax.legend()
    ax.grid(True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "timing_comparison.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Experiment 2: Convergence with number of pseudo time steps
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Experiment 2: Convergence with number of pseudo time steps")
    print("=" * 70)

    fixed_N_steps = 500
    fixed_dt_steps = 0.1
    num_steps_list = [10, 50, 100, 500, 1000, 5000, 10000]

    key1, key2, rng_key = jax.random.split(rng_key, 3)
    kl_steps = run_experiment("kl", [fixed_N_steps], num_steps_list, [fixed_dt_steps], key1)
    hell_steps = run_experiment("hellinger", [fixed_N_steps], num_steps_list, [fixed_dt_steps], key2)

    horizons = [n * fixed_dt_steps for n in num_steps_list]
    _plot_convergence(
        kl_steps, hell_steps, horizons,
        key_fn=lambda T: (fixed_N_steps, int(round(T / fixed_dt_steps)), fixed_dt_steps),
        xlabel="Pseudo time horizon (T = steps × dt)",
        title_suffix=f"vs Pseudo time steps (N={fixed_N_steps}, dt={fixed_dt_steps})",
        filename="convergence_num_steps.png",
    )
    plot_2d_posterior_comparison(
        kl_steps, hell_steps, num_steps_list,
        key_fn=lambda n: (fixed_N_steps, n, fixed_dt_steps),
        label_fn=lambda n: f"steps={n}",
        filename="posterior_2d_steps.png",
    )

    # -----------------------------------------------------------------------
    # Experiment 3: Convergence with step size
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Experiment 3: Convergence with step size (pseudo time horizon)")
    print("=" * 70)

    fixed_N_dt = 500
    fixed_nsteps_dt = 1000
    step_sizes = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    key1, key2, rng_key = jax.random.split(rng_key, 3)
    kl_dt = run_experiment("kl", [fixed_N_dt], [fixed_nsteps_dt], step_sizes, key1)
    hell_dt = run_experiment("hellinger", [fixed_N_dt], [fixed_nsteps_dt], step_sizes, key2)

    dt_horizons = [fixed_nsteps_dt * dt for dt in step_sizes]
    _plot_convergence(
        kl_dt, hell_dt, dt_horizons,
        key_fn=lambda T: (fixed_N_dt, fixed_nsteps_dt, T / fixed_nsteps_dt),
        xlabel=f"Pseudo time horizon (T = {fixed_nsteps_dt} × dt)",
        title_suffix=f"vs Step size (N={fixed_N_dt}, steps={fixed_nsteps_dt})",
        filename="convergence_step_size.png",
    )

    # -----------------------------------------------------------------------
    # Summary figure: 3×3 grid
    # -----------------------------------------------------------------------
    print("\nGenerating summary figure...")

    metrics = ["mean_err", "cov_err", "w1"]
    metric_labels = ["Mean L2 error", "Cov Frobenius error", "Wasserstein-1"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # Col 0: vs ensemble size
    for row, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[row, 0]
        kl_vals = [kl_ens[(N, fixed_nsteps_ens, fixed_dt_ens)][metric] for N in ensemble_sizes]
        hell_vals = [hell_ens[(N, fixed_nsteps_ens, fixed_dt_ens)][metric] for N in ensemble_sizes]
        ax.plot(ensemble_sizes, kl_vals, "o-", label="KL", linewidth=2)
        ax.plot(ensemble_sizes, hell_vals, "s--", label="Hellinger", linewidth=2)
        ax.set_xlabel("Ensemble size")
        ax.set_ylabel(mlabel)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True)

    # Col 1: vs num pseudo time steps (as horizon)
    horizons_list = [n * fixed_dt_steps for n in num_steps_list]
    for row, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[row, 1]
        kl_vals = [kl_steps[(fixed_N_steps, n, fixed_dt_steps)][metric] for n in num_steps_list]
        hell_vals = [hell_steps[(fixed_N_steps, n, fixed_dt_steps)][metric] for n in num_steps_list]
        ax.plot(horizons_list, kl_vals, "o-", label="KL", linewidth=2)
        ax.plot(horizons_list, hell_vals, "s--", label="Hellinger", linewidth=2)
        ax.set_xlabel("Pseudo time horizon T")
        ax.set_ylabel(mlabel)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True)

    # Col 2: vs step size
    dt_h = [fixed_nsteps_dt * dt for dt in step_sizes]
    for row, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[row, 2]
        kl_vals = [kl_dt[(fixed_N_dt, fixed_nsteps_dt, dt)][metric] for dt in step_sizes]
        hell_vals = [hell_dt[(fixed_N_dt, fixed_nsteps_dt, dt)][metric] for dt in step_sizes]
        ax.plot(dt_h, kl_vals, "o-", label="KL", linewidth=2)
        ax.plot(dt_h, hell_vals, "s--", label="Hellinger", linewidth=2)
        ax.set_xlabel(f"Pseudo time horizon ({fixed_nsteps_dt} × dt)")
        ax.set_ylabel(mlabel)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True)

    axes[0, 0].set_title("vs Ensemble size")
    axes[0, 1].set_title("vs Num pseudo time steps")
    axes[0, 2].set_title("vs Step size")

    fig.suptitle("PFF Convergence (Linear case): KL vs Hellinger", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "summary_convergence.png", dpi=150)
    plt.close(fig)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
