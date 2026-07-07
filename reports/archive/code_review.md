# Code Review — non_gaussian_data_assim

Scope: everything under `src/non_gaussian_data_assim/`, `scripts/` (excluding `scripts/archive/`), `tests/`, and `configs/`. A companion report, `reports/math_numerics_review.md`, covers mathematical and numerical correctness; issues that are both are cross-referenced.

---

## 1. Bugs that should be fixed

### 1.1 `scripts/main.py` is currently broken for 3 of the 4 cases (blocker)

`main.py:69` unconditionally reads `cfg.case.val_obs_operator`, but only `configs/case/kuramoto.yaml` defines that key. Running `case=lorenz_63` (the default!), `lorenz_96`, or `coupled_kuramoto` fails immediately:

```
omegaconf.errors.ConfigAttributeError: Key 'val_obs_operator' is not in struct
```

Verified by running `uv run python scripts/main.py case=lorenz_63 da_method=enkf ...`. This means the pytest smoke suite (`tests/test_main.py`) is currently red for 12 of 16 combinations. Fix: add `val_obs_operator` to the other three case files, or make it optional via `OmegaConf.select(cfg, "case.val_obs_operator", default=None)` and guard the downstream validation logic.

### 1.2 `best_guess_perturbation` config key does not exist in code

`configs/case/lorenz_63.yaml:57`, `lorenz_96.yaml:63`, and `coupled_kuramoto.yaml:74` pass `best_guess_perturbation` inside the `initial_ensemble` block, but `InitialEnsembleGenerator.__init__` (`src/.../ensemble_generation/initial_ensemble.py:28`) has no such parameter — the corresponding mapping is commented out at the top of the file. Once 1.1 is fixed, Hydra's `instantiate` will raise `TypeError: unexpected keyword argument 'best_guess_perturbation'` for these cases. Either implement the option or delete the key from the configs.

### 1.3 Observation timing off-by-one-window

`observation_utils.generate_observations` (`observations/observation_utils.py:37`) samples the truth at index `model_integration_steps * i + 1`, but the DA loop assimilates `observations[i]` into the ensemble *after* a full outer step of `model_integration_steps` inner steps, i.e. at truth index `model_integration_steps * (i + 1)`. The two agree only when `model_integration_steps == 1`; the configs use 5–50. Meanwhile the information-metrics section of `main.py:360` indexes assimilation times as `m * (i + 1)` — inconsistent with how the observations were generated. This is the single most consequential defect in the repo; see the math report §1.1 for the scientific impact. Note `scripts/main_simple.py` gets the convention right (`true_states[1:]` with `MODEL_STEPS`-aligned analysis indices) — use it as the reference.

### 1.4 AGMF mutates Python state inside traced code

`AdaptiveGaussianMixtureFilter._analysis_step` does `self.w_prev = w_t` (`da_methods/agmf.py:131`). Two failure modes:

- Under `BaseDataAssimilationMethod.rollout` (which wraps `_assimilate_data` in `jax.lax.scan` + `jax.jit`), the assignment executes once at trace time with a tracer. The weight recursion silently never happens across scan steps (weights stay at their initial value), and `self.w_prev` is left holding a leaked tracer, so any later eager call on the same object crashes.
- Even in the eager loop of `main.py`, `w_prev` is never reset between experiments if the object is reused.

Fix: thread the weights through the scan carry (make `_analysis_step` return `(ensemble, w_t)` and let the rollout carry them), or explicitly document/enforce that AGMF cannot be used with `.rollout()`.

### 1.5 `main.py` save path: `NameError` and wrong keys

- `main.py:494`: `if forecast_metrics is not None:` — `forecast_metrics` is only assigned when `forecast_ensemble is not None` (line 319-320). With `forecast_steps: null` and `save.experiment: true` this is a `NameError`. Initialize `forecast_metrics = None` next to `forecast_ensemble = None`.
- `main.py:492`: innovation metrics are stored under the key `"metrics_to_save"` (copy-paste typo), and the value is the *loop variable* `innovation_metrics`, i.e. only the last observed state's metrics — `innov_metric_list` (which holds all states) is built and then never saved or plotted.
- `main.py:86`: `rloc = str(int(cfg.localization_distance))` — `localization_distance` is not defined at the top level of `lorenz_63.yaml`, so saving crashes for that case. The variable is only used in a commented-out line; delete it or guard it.

### 1.6 Validation R sliced from the wrong matrix

`main.py:287`: in the validation-observation loop, `R_val_states.append(R[idx : idx + i, idx : idx + i])` slices the *assimilation* `R` instead of `R_val`. If the validation operator has more observations than the assimilation operator, the slice silently truncates (JAX/NumPy slicing does not bounds-check) and the block is wrong. Currently `R_val_states` is unused downstream, which is why it hasn't bitten — fix or remove.

### 1.7 PFF: dead scalar-covariance check after the inverse

`da_methods/pff.py:198-201`: `prior_cov_inv = jnp.linalg.inv(prior_cov)` runs *before* the `if len(prior_cov.shape) == 0: prior_cov = prior_cov.reshape(1, 1)` guard. For a 1-dof problem the `inv` call fails first, so the guard is unreachable. Move the reshape above the inverse (and note it never fixes `prior_cov_inv`).

### 1.8 `RedNoise.sample` returns `None` for `alpha < 0`

`perturbations/red_noise.py:69-99` handles `alpha > 0` and `alpha == 0` and silently falls off the end otherwise. Raise `ValueError` for negative alpha.

### 1.9 `da_rollout` plumbing in `da_methods/base.py`

- `rollout()` (base.py:169-184): the `if return_model_integration_steps:` and `else:` branches are byte-identical — the flag is silently ignored. If it were honored, the shapes inside `scan` would be inconsistent anyway (the concat in `_assimilate_data` produces a different time dimension). Either implement it or raise `NotImplementedError`.
- `**kwargs` is forwarded to `da_rollout(...)`, which accepts no `**kwargs` — any caller passing extra kwargs gets a `TypeError`.
- The docstring of `da_rollout` says "The first observation (index 0) is the initial observation at time 0", but the scan forecasts *before* assimilating `observations[0]` — the doc contradicts the code (and ties into 1.3).

### 1.10 `np.savez(..., allow_pickle=False)` stores a bogus array

`utils/saving.py:51`: `np.savez` has no `allow_pickle` parameter — keyword arguments become arrays in the archive, so every metrics `.npz` contains an array literally named `allow_pickle`. Harmless today but confusing, and it masks the intent (pickle rejection). Remove it (or use `np.savez` + validate dtypes yourself). Related: if any metric value is `None` (the optional information-metric fields), `np.asarray(None)` produces an object array and the save fails.

### 1.11 `NonlinearObservationOperator.grad_obs_operator` only works for scalar h

`observations/observation_operator.py:184-186` uses `jax.grad`, which requires a scalar output. Any nonlinear operator with more than one observation raises at runtime. Use `jax.jacobian` (and fix the PFF likelihood-score contraction accordingly). Also `self.num_states = len(obs_states)` overwrites the semantic "number of physical states" with "number of observed states" — misleading naming.

### 1.12 `BreedingPerturbation.sample` violates the base-class contract

`perturbations/breeding.py:355-384`: the base class declares `sample -> jnp.ndarray`, but breeding always returns `(ensemble, bv_metrics)` (with `return_metrics` hard-coded `True`). The caller (`InitialEnsembleGenerator.sample`) special-cases tuples, but the diagnostics are then discarded (the `return ensemble, diagnostics` path is commented out). Make the return type honest and configurable.

### 1.13 `kernels.get_pairwise_kernel_scalar_fn` would crash if used

`kernels.py:104-106` indexes `K[0, 0]`, but `kernel_matrix_fn` for the scalar kernel returns a 0-d scalar (the `* jnp.eye(...)` is commented out). Currently dead code — delete it or fix the indexing.

---

## 2. Improvements (no new features)

### 2.1 Performance

- **Re-jitting on every call.** `BaseForwardModel.__call__` (`forward_models/base.py:54-64`) builds a fresh closure and calls `jax.jit` on it *every invocation*. The jit cache is keyed by function identity, so every DA step in `main.py`'s eager loop recompiles the model rollout. Same pattern in `BaseForwardModel.rollout` and in `ParticleFlowFilter._analysis_step` (jits the kernels and the pseudo-time rollout per analysis step). Cache the jitted callables in `__init__` (one per `(return_model_integration_steps, is_ensemble)` combination) — this is likely a large constant-factor speedup for every experiment.
- **Quadratic trajectory building.** `main.py:232-237` grows `posterior_ensemble` and `prior_ensemble_da` with `jnp.concatenate` inside the loop — O(T²) copying on device. Append to Python lists and `jnp.concatenate` once after the loop, or use the (fixed) scan-based `rollout`.
- **`prior_ensemble_da` is dead.** It is accumulated every step (`main.py:232`) and never used afterwards. Delete it.
- **`gaussian_mixt`** (`gaussian_mixture.py`): a Python loop with `.at[i].set`, recomputing `jnp.linalg.inv(cov_matrix)` inside every iteration. One `einsum` over the innovation matrix replaces the whole loop; also see math report §1.3 for correctness issues in the same function.
- **EnKF/AGMF form the full `dofs × dofs` covariance.** Fine at current sizes but avoidable; see math report §2.3 for the anomaly-space formulation.

### 2.2 Structure / duplication

- EnKF, AGMF, and PFF each repeat the same preamble: reshape to `[dofs, N]`, localization-closure setup, inflation, observation perturbation. The localization closure is copy-pasted three times (`enkf.py:50-59`, `agmf.py:60-69`, `pff.py:166-175`). Extract a small shared helper (mixin or module function) — this also removes the risk that the inflation/localization order diverges further between filters (it already has; math report §1.4).
- `da_methods/base.py` mixes two calling conventions (eager `__call__` and scan-based `rollout`); `main.py` uses neither `rollout` nor `da_rollout`. Pick one driver, make it work, and use it everywhere (tests, main, analytical harness). The analytical harness calls the private `da_model._analysis_step` directly (`scripts/analytical/_common.py:162`) — expose a public `analysis(ensemble, obs, key)` method instead.
- `main.py` is ~500 lines of straight-line script mixing setup, DA loop, metrics, information theory, plotting, and saving. Split into functions (`build_experiment(cfg)`, `run_da(...)`, `compute_metrics(...)`, `save(...)`) so tests can exercise pieces without a subprocess.

### 2.3 Dead code and leftovers

- `import pdb` in 8 library modules (`enkf.py`, `agmf.py`, `base.py` ×2, `lorenz_63.py`, `lorenz_96.py`, `time_integrators.py`, `observation_operator.py`).
- Large commented-out blocks: output-dict remnants in `enkf.py:124-135` and `agmf.py:173-182`, the old flow loop in `pff.py:268-275`, class-based integrators in `time_integrators.py:176-203`, plotting calls in `main.py`.
- Unused: `rand_utils.randsample` (also uses the global NumPy RNG, inconsistent with the JAX-key discipline everywhere else), `kernels.get_pairwise_kernel_scalar_fn`/`get_pairwise_interactions_fn`, `DEFAULT_B_D` and the `alpha` parameter in `pff.py` (accepted, stored, never used since the bandwidth line is commented out), `ParticleFilter.resample_threshold` (documented as unused — fine, but consider removing after a deprecation window), `IdentityModel`/`SineModel`/`SineObservationOperator*` (only referenced by `scripts/archive/`), `forward_models/bk_rsf_1d.py` (pure NumPy, no `BaseForwardModel` interface, unused outside archive — and its RK4 is wrong, see math report §1.2; either port it properly or move it to `archive/`).
- `ensemble_metrics.crps_ensemble_1d`: an O(N·K²) triple-loop NumPy duplicate of `CRPS`, with personal-note comments ("Max: My method..."). Move it into a test as a reference implementation for the fast estimator, or delete.
- `tests/__init__.py` and empty `src/non_gaussian_data_assim/__init__.py` are fine; `utils/` lacks an `__init__.py`? (it imports fine as a namespace package — consider adding one for consistency).

### 2.4 Config hygiene

- `da_method_overrides` forces every case YAML to enumerate every DA method; a missing entry crashes `main.py:72`. Use `OmegaConf.select(cfg.case, f"da_method_overrides.{cfg.da_method.name}", default={})`.
- `configs/config.yaml` default `model_integration_steps: 50` combined with the timing bug (1.3) makes the default setup maximally wrong — after fixing 1.3, revisit defaults.
- The experiment-name builder (`saving.creat_exp_name` + `main.py:84-87`) hard-codes keys (`inflation_factor`, `localization_distance`) that not all configs define.

### 2.5 Naming, typing, tooling

- Typos in public API: `creat_exp_name`, `retur_nv`, `_singelton_ensemble_axis`, "reuires", "Calcualte", "Metirc", "betweem". Docstrings in `lorenz_63.py` say "Lorenz 96" (three times); `pff.py:187` comment says `[dofs, ensemble]` for an `[ensemble, dofs]` array; `L63_RHS` docstring says "Lorenz 96".
- Many type annotations are wrong: functions returning callables annotated `-> np.ndarray` (`pff.py:27-65`), `rng_key: jax.random.PRNGKey = None`, `time_integrators.rollout(...) -> jnp.ndarray` (returns a function). `mypy` is configured strictly in `pyproject.toml` but clearly not passing — either run it in CI or relax it.
- `pyproject.toml` leftovers from another project: `description = "Add your description here"`, `known_first_party = ["scientific_stochastic_interpolants"]`, mypy override for `scisi.metrics.LSIM.*`.
- `main.py:453-455`: bare `except:` around `best_guess_profile[0]` — catch the specific exception (or better, make `best_guess_profile` always defined).
- `abstractmethod` used without inheriting `ABC` in `BaseForwardModel`, `BaseDataAssimilationMethod`, `ObservationOperator`, `Metric` — the decorator is inert, so instantiating an incomplete subclass fails only at call time. Inherit `abc.ABC`.

### 2.6 Testing

- The only tests are 16 subprocess smoke runs of `main.py` (~minutes each, 180 s timeout). Since the suite is currently red (1.1), it evidently isn't run routinely — add CI (GitHub Actions with `uv sync && uv run pytest`).
- Add fast unit tests that don't shell out: (a) each `_analysis_step` on a small linear-Gaussian problem against the closed-form Kalman posterior (the machinery already exists in `scripts/analytical/_common.py` — turn it into an importable module and reuse it); (b) `generate_observations` index alignment (would have caught 1.3); (c) localization mask PSD-ness and shape; (d) CRPS fast estimator vs. the O(K²) reference; (e) forward-model regression tests (one step of L96/KS against stored values).

---

## 3. Recommended extensions (code-level)

1. **CI + pre-commit enforcement.** `pre-commit`, `black`, `isort`, `mypy` are all in the dev deps but nothing enforces them. A single GitHub Actions workflow running lint + unit tests + one smoke combo would have caught the current breakage at merge time.
2. **End-to-end jitted DA driver.** After fixing 1.4 and 1.9, make `BaseDataAssimilationMethod.rollout` the canonical driver in `main.py` (scan over observations, no per-step recompiles, no O(T²) concatenation). Keep the eager loop as a debug path.
3. **Structured configs.** Replace free-form YAML with Hydra structured configs (dataclasses). Both 1.1 and 1.2 are "config key drifted from code" failures that structured configs catch at compose time.
4. **Experiment result container.** Save trajectories as xarray/netCDF (named dims: `member, time, state, x`) instead of loose `.npy` files; carry metadata (config hash, git SHA, seed). The `ExperimentSaver` is a good seed for this.
5. **Multi-seed / sweep runner.** Everything is single-seed; add a thin runner that fans out seeds (Hydra multirun already gets you most of the way) and aggregates metrics with confidence intervals — essential for method comparisons in the paper context.
6. **Uniform diagnostics across filters.** Innovation statistics are computed only for EnKF (`main.py:430`); they are equally well-defined for AGMF/PFF/PF. Compute them for every method, and save `innov_metric_list` per state.
7. **float64 switch.** Expose `jax_enable_x64` as a config flag; covariance solves and chi² diagnostics benefit, and the analytical KF comparisons are cleaner in double precision.
8. **Public API surface.** Populate `src/non_gaussian_data_assim/__init__.py` (and subpackage `__init__`s) with the intended public classes so users don't import deep module paths; add `__all__`.
9. **Archive quarantine.** `scripts/archive/` and `archive/` still import from the library and rot silently; exclude them from tooling explicitly (mypy already does) and state in the README that they are unsupported.
