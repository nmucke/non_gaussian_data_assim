"""Shared helpers for the analytical (linear-Gaussian) DA experiments.

Used by both:
  * main_analytical_methods.py       -- sweep over DA methods at fixed N
  * main_analytical_ensemble_size.py -- sweep over N for one DA method

The system is linear-Gaussian by construction, so the exact filtering posterior
is the analytical Kalman filter; both scripts compare ensemble methods against
it.
"""

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.patches import Ellipse

from non_gaussian_data_assim.metrics.ensemble_metrics import CRPS
from non_gaussian_data_assim.metrics.innovation_metrics import (
    ChiSquared,
    NormalizedInnovations,
)
from non_gaussian_data_assim.metrics.probability_metrics import KLDivergence
from non_gaussian_data_assim.metrics.trajectory_metrics import (
    MAE,
    MAPE,
    RMSE,
    ensemble_spread,
)


# --------------------------------------------------------------------------- #
#                          Analytical Kalman filter                           #
# --------------------------------------------------------------------------- #


def kalman_predict(m, P, M, Q):
    return M @ m, M @ P @ M.T + Q


def kalman_update(m, P, H, R, y):
    S = H @ P @ H.T + R
    K = jnp.linalg.solve(S, H @ P).T
    m_post = m + K @ (y - H @ m)
    P_post = (jnp.eye(P.shape[0]) - K @ H) @ P
    return m_post, 0.5 * (P_post + P_post.T)


def gaussian_kl(mu_p, sig_p, mu_q, sig_q):
    d = mu_p.shape[0]
    sig_q_inv = jnp.linalg.inv(sig_q)
    diff = mu_q - mu_p
    _, ldp = jnp.linalg.slogdet(sig_p)
    _, ldq = jnp.linalg.slogdet(sig_q)
    return 0.5 * (jnp.trace(sig_q_inv @ sig_p) + diff @ sig_q_inv @ diff - d + ldq - ldp)


# --------------------------------------------------------------------------- #
#                          Config + ensemble helpers                          #
# --------------------------------------------------------------------------- #


def get_state_dim(cfg):
    state_dim = len(cfg.prior_mean)
    if "state_dim" in cfg and int(cfg.state_dim) != state_dim:
        raise ValueError(
            f"cfg.state_dim ({cfg.state_dim}) disagrees with len(prior_mean) ({state_dim})."
        )
    return state_dim


def np_matrix(x, expected_shape, name):
    """Convert YAML to a float numpy array of ``expected_shape``.

    A scalar is broadcast to ``scalar * I`` when ``expected_shape`` is square,
    so covariance fields can be written as ``0.05`` instead of a full matrix.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0 and len(expected_shape) == 2 and expected_shape[0] == expected_shape[1]:
        arr = float(arr) * np.eye(expected_shape[0])
    if arr.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {arr.shape}")
    return arr


def load_system_matrices(cfg, state_dim, num_obs, obs_operator):
    """Materialize ``(M_eff, Q, R, m0, P0, H)`` from the config."""
    M = jnp.asarray(np_matrix(cfg.transition_matrix, (state_dim, state_dim), "transition_matrix"))
    Q = jnp.asarray(np_matrix(cfg.process_noise_cov, (state_dim, state_dim), "process_noise_cov"))
    R = jnp.asarray(np_matrix(cfg.obs_noise_cov, (num_obs, num_obs), "obs_noise_cov"))
    m0 = jnp.asarray(np_matrix(cfg.prior_mean, (state_dim,), "prior_mean"))
    P0 = jnp.asarray(np_matrix(cfg.prior_cov, (state_dim, state_dim), "prior_cov"))
    M_eff = jnp.linalg.matrix_power(M, cfg.model_integration_steps)
    H = jnp.asarray(obs_operator.obs_matrix.todense())
    return M_eff, Q, R, m0, P0, H


def ensemble_mean_cov(ens_flat):
    return jnp.mean(ens_flat, axis=0), jnp.cov(ens_flat.T)


def to_metric_shape(ens, state_dim):
    return ens.reshape(ens.shape[0], 1, 1, state_dim)


# --------------------------------------------------------------------------- #
#                          Truth + KF rollout, DA loop                        #
# --------------------------------------------------------------------------- #


def run_truth_and_kf(rng_key, n_steps, state_dim, M_eff, Q, R, H, m0, P0, obs_operator):
    """Roll out truth, generate noisy obs, and step the analytical Kalman filter.

    Returns ``(truth_arr [T+1,1,d], obs_arr [T,num_obs],
    kf_mean_arr [T+1,d], kf_cov_arr [T+1,d,d])``.
    """
    rng_key, sub = jax.random.split(rng_key)
    x_true = jax.random.multivariate_normal(sub, m0, P0)
    truth_hist = [x_true]
    obs_hist = []
    kf_mean_hist, kf_cov_hist = [m0], [P0]
    m_kf, P_kf = m0, P0
    for _ in range(n_steps):
        rng_key, sub = jax.random.split(rng_key)
        x_true = M_eff @ x_true + jax.random.multivariate_normal(sub, jnp.zeros(state_dim), Q)
        rng_key, sub = jax.random.split(rng_key)
        y = obs_operator(x_true.reshape(1, 1, state_dim))[0] + \
            jax.random.multivariate_normal(sub, jnp.zeros(obs_operator.num_obs), R)
        m_pred, P_pred = kalman_predict(m_kf, P_kf, M_eff, Q)
        m_kf, P_kf = kalman_update(m_pred, P_pred, H, R, y)
        truth_hist.append(x_true)
        obs_hist.append(y)
        kf_mean_hist.append(m_kf)
        kf_cov_hist.append(P_kf)

    truth_arr = jnp.stack(truth_hist).reshape(n_steps + 1, 1, state_dim)
    obs_arr = jnp.stack(obs_hist)
    kf_mean_arr = jnp.stack(kf_mean_hist)
    kf_cov_arr = jnp.stack(kf_cov_hist)
    return truth_arr, obs_arr, kf_mean_arr, kf_cov_arr


def run_da_method(rng_key, da_model, forward_model, obs_operator,
                  prior_ensemble, observations, Q, n_steps, state_dim):
    """Run one DA method for ``n_steps``.

    Returns ``(ens [N,T+1,1,d], pred_obs [T,N,num_obs])``.
    """
    N = prior_ensemble.shape[0]
    current = prior_ensemble
    traj = [current[:, None]]
    pred_obs = []
    for i in range(n_steps):
        forecast = forward_model(
            current, return_model_integration_steps=False, is_ensemble=True
        )[:, -1]
        rng_key, sub = jax.random.split(rng_key)
        forecast = forecast + jax.random.multivariate_normal(
            sub, jnp.zeros(state_dim), Q, shape=(N,)
        )
        pred_obs.append(np.asarray(obs_operator(forecast)))
        rng_key, sub = jax.random.split(rng_key)
        posterior = da_model._analysis_step(
            forecast, observations[i], rng_key=sub
        ).reshape(N, 1, state_dim)
        current = posterior
        traj.append(posterior[:, None])
    return jnp.concatenate(traj, axis=1), jnp.stack(pred_obs, axis=0)


def sample_analytical_posterior(rng_key, n_steps, kf_mean_arr, kf_cov_arr, n_samples):
    """Sample ``n_samples`` from each step's Gaussian posterior. Shape ``[T+1, n_samples, d]``."""
    keys = jax.random.split(rng_key, n_steps + 1)
    return jnp.stack([
        jax.random.multivariate_normal(keys[t], kf_mean_arr[t], kf_cov_arr[t], shape=(n_samples,))
        for t in range(n_steps + 1)
    ])


# --------------------------------------------------------------------------- #
#                                Metrics                                      #
# --------------------------------------------------------------------------- #


def compute_metrics(state_dim, n_steps, ens, pred_obs, truth_arr, obs_arr, R,
                    kf_mean_arr, kf_cov_arr, analytical_samples_arr):
    """Compute every diagnostic for one ``(ens, pred_obs)`` run."""
    eye = jnp.eye(state_dim)
    rmse = RMSE(ensemble_aggregation="mean", time_aggregation="none")
    mae = MAE(ensemble_aggregation="mean", time_aggregation="none")
    mape = MAPE(ensemble_aggregation="mean", time_aggregation="none")
    crps = CRPS(time_aggregation="none")
    chi2 = ChiSquared(time_aggregation="none")
    z_innov = NormalizedInnovations(time_aggregation="none")
    hist_kl = KLDivergence(time_aggregation="mean", n_bins=40)

    out: dict = {}
    out["rmse"] = np.asarray(rmse(ens, truth_arr))
    out["mae"] = np.asarray(mae(ens, truth_arr))
    out["mape"] = np.asarray(mape(ens, truth_arr))
    out["crps"] = np.asarray(crps(ens, truth_arr))
    out["spread"] = np.asarray(ensemble_spread(ens))
    out["chi_squared"] = np.asarray(chi2(pred_obs, obs_arr, R))
    out["z"] = np.asarray(z_innov(pred_obs, obs_arr, R))

    # ---- observation-space diagnostics: predicted obs vs noisy obs ---- #
    pred_mean = jnp.mean(pred_obs, axis=1)
    pred_var = jnp.var(pred_obs, axis=1)
    residual = obs_arr - pred_mean
    out["innov_rmse"] = np.asarray(jnp.sqrt(jnp.mean(residual ** 2, axis=1)))
    out["innov_mae"] = np.asarray(jnp.mean(jnp.abs(residual), axis=1))
    out["innov_bias"] = np.asarray(jnp.mean(residual, axis=1))
    out["pred_obs_spread"] = np.asarray(jnp.sqrt(jnp.mean(pred_var, axis=1)))
    pred_for_crps = jnp.transpose(pred_obs, (1, 0, 2))[:, :, None, :]
    obs_for_crps = obs_arr[:, None, :]
    out["pred_obs_crps"] = np.asarray(crps(pred_for_crps, obs_for_crps))
    pred_std = jnp.sqrt(pred_var)
    out["coverage_2sigma"] = np.asarray(jnp.mean(
        (jnp.abs(residual) <= 2.0 * pred_std).astype(jnp.float32), axis=1
    ))

    # ---- posterior-aware diagnostics: ensemble vs analytical KF ---- #
    gkl, hkl, merr, cerr = [], [], [], []
    state_bias, state_mahal, state_cov = [], [], []
    for t in range(n_steps + 1):
        ens_t = ens[:, t].reshape(ens.shape[0], -1)
        mu_p, sig_p = ensemble_mean_cov(ens_t)
        sig_p = sig_p + 1e-9 * eye
        P_kf_t = kf_cov_arr[t] + 1e-9 * eye
        gkl.append(float(gaussian_kl(mu_p, sig_p, kf_mean_arr[t], P_kf_t)))
        hkl.append(float(hist_kl(to_metric_shape(ens_t, state_dim),
                                 to_metric_shape(analytical_samples_arr[t], state_dim))))
        diff = mu_p - kf_mean_arr[t]
        merr.append(float(jnp.linalg.norm(diff)))
        cerr.append(float(jnp.linalg.norm(sig_p - kf_cov_arr[t])))
        state_bias.append(float(jnp.mean(-diff)))
        state_mahal.append(float(diff @ jnp.linalg.solve(P_kf_t, diff) / state_dim))
        method_std_t = jnp.sqrt(jnp.clip(jnp.diag(sig_p), 0.0, None))
        state_cov.append(float(jnp.mean((jnp.abs(diff) <= 2.0 * method_std_t).astype(jnp.float32))))
    out["gaussian_kl"] = np.asarray(gkl)
    out["hist_kl"] = np.asarray(hkl)
    out["mean_err"] = np.asarray(merr)
    out["cov_err_fro"] = np.asarray(cerr)
    out["state_bias"] = np.asarray(state_bias)
    out["state_mahalanobis"] = np.asarray(state_mahal)
    out["state_coverage_2sigma"] = np.asarray(state_cov)
    return out


TIME_MEAN_METRICS = frozenset({"chi_squared"})


def summary_value(arr, key):
    """Final-step value; ``chi_squared`` is reported as a time-mean (noisy per step)."""
    a = np.atleast_1d(arr)
    return float(np.mean(a)) if key in TIME_MEAN_METRICS else float(a[-1])


# --------------------------------------------------------------------------- #
#                              Plotting                                       #
# --------------------------------------------------------------------------- #


METRIC_TITLES = {
    "rmse": "RMSE vs truth",
    "mae": "MAE vs truth",
    "mape": "MAPE vs truth",
    "crps": "CRPS vs truth",
    "spread": "ensemble spread",
    "chi_squared": r"$\chi^2$ innovation (ideal = 1)",
    "gaussian_kl": "Gaussian KL(method ‖ analytical)",
    "hist_kl": "histogram KL(method ‖ analytical)",
    "mean_err": r"$\|\mathrm{mean} - m_{KF}\|$",
    "cov_err_fro": r"$\|\Sigma - P_{KF}\|_F$",
    "innov_rmse": r"innovation RMSE: $\sqrt{\langle (y - \langle HX^f \rangle)^2 \rangle}$",
    "innov_mae": r"innovation MAE: $\langle |y - \langle HX^f \rangle| \rangle$",
    "innov_bias": r"innovation bias: $\langle y - \langle HX^f \rangle \rangle$",
    "pred_obs_spread": r"predicted-obs spread: $\sqrt{\langle \mathrm{var}(HX^f) \rangle}$",
    "pred_obs_crps": "CRPS in observation space",
    "coverage_2sigma": r"fraction of obs inside predicted $\pm 2\sigma$ (ideal $\approx 0.95$)",
    "state_bias": r"state-space bias: $\langle m_{KF} - \mathrm{mean}_\mathrm{ens} \rangle$",
    "state_mahalanobis": r"state Mahalanobis: $(m - m_{KF})^\top P_{KF}^{-1} (m - m_{KF}) / d$",
    "state_coverage_2sigma": r"fraction of state dims with $m_{KF}$ inside method's $\pm 2\sigma$",
}


def cov_ellipse(ax, mean, cov, n_std, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * n_std * np.sqrt(np.clip(vals, 0.0, None))
    ax.add_patch(Ellipse(xy=mean, width=width, height=height, angle=angle,
                         fill=False, **kwargs))
