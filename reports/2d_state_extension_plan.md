# Plan: Supporting 2-D States `[ensemble, time, num_states, nx, ny]`

Goal: the library should accept states with one **or** two spatial dimensions — `[E, T, S, state_dim]` and `[E, T, S, state_dim_x, state_dim_y]` — with **one implementation of every DA method** (and of localization, observation operators, ensemble generation, metrics). No `if ndim == 2` forks inside the filters.

## 0. Why this is cheap for the DA methods (and where the real work is)

Every analysis step in the repo already flattens to matrix form on entry:

```python
prior_ensemble = prior_ensemble.reshape(self.ensemble_size, -1)   # pff.py
prior_ensemble = prior_ensemble.reshape(self.ensemble_size, -1).T # enkf.py, agmf.py
```

`reshape(E, -1)` works identically for a trailing `(S, nx)` or `(S, nx, ny)` — the EnKF/AGMF covariance algebra, the PFF kernels and scores, and the particle filter are all *already* dimension-agnostic in flattened space. The 4-D assumption lives only at the **edges**. Exhaustive list of shape-coupled sites (verified by grep):

| Site | Assumption |
|---|---|
| `enkf.py:117`, `agmf.py:185`, `pff.py:279,286` | final `reshape(E, num_states, state_dim)` |
| `da_methods/base.py:105` | `[:, :-1, :, :]` / `[:, None, :, :]` (explicit 4-D indexing) |
| `time_integrators.py:60` | `trajectory.reshape(-1, num_states, state_dim)` |
| `localization.py` | pairwise distance from a 1-D grid index, `\|i−j\|` with 1-D periodic wrap |
| `observation_operator.get_obs_matrix` | flat index = `state_idx * state_dim + dim_idx` (1-D spatial index) |
| `perturbations/red_noise.py`, `initial_profiles.py` (GRF, cosine) | 1-D FFTs (`fftfreq`, `irfft`) |
| periodicity enforcement (3 sites) | `.at[..., -1].set(x[..., 0])` — last axis only |
| `metrics/probability_metrics.py:59-63` | unpacks exactly `(n_pred, n_time, n_states, n_dim)` |
| `main.py:190-195,233` | `reshape(E, 1, num_states, state_dim)` and `[:, None, :, :]` |
| `initial_profiles.CosineProfile:137` | `reshape(E, 1, state_dim)` |
| plotting | Hovmöller diagrams assume `[.., T, S, x]` |

So the plan is: introduce a single **`StateSpace`** object as the source of truth for shapes, make the edges consult it (or just use `...`/`reshape(x, -1)` idioms), and generalize the genuinely spatial components (localization distances, H-matrix indexing, spectral noise, plotting). The DA method bodies are not rewritten — requirement satisfied by construction.

### Design decision: shape-polymorphic core (recommended) vs. flat-at-the-boundary

- **Option A — flatten at the boundary**: keep the internal canonical shape `[E, T, S, n_spatial]` and let only forward models/plotting see `(nx, ny)`. Least code, but user-facing arrays are flat, every 2-D forward model wraps itself in reshapes, and configs/analysis notebooks deal in flattened indices.
- **Option B — shape-polymorphic core** *(recommended)*: arrays keep their natural trailing shape `(S, *spatial_shape)` everywhere; library code never mentions trailing dims except through `StateSpace`. Costs ~a dozen edits (table above), and the 5-D arrays flow through natively, which is what you asked for.

The rest of this plan implements Option B.

## 1. The `StateSpace` abstraction (core of PR 1)

```python
# src/non_gaussian_data_assim/state_space.py
@dataclass(frozen=True)
class StateSpace:
    num_states: int
    spatial_shape: tuple[int, ...]          # (nx,) or (nx, ny)
    periodic: tuple[bool, ...] = ()         # per spatial axis; default all-False
    domain_lengths: tuple[float, ...] = ()  # optional physical sizes, per axis

    @property
    def ndim_spatial(self): return len(self.spatial_shape)
    @property
    def n_spatial(self): return math.prod(self.spatial_shape)
    @property
    def dofs(self): return self.num_states * self.n_spatial
    @property
    def member_shape(self): return (self.num_states, *self.spatial_shape)

    def flatten(self, x):        # [..., S, *spatial] -> [..., dofs]
        return x.reshape(*x.shape[: x.ndim - 1 - self.ndim_spatial], self.dofs)
    def unflatten(self, x):      # [..., dofs] -> [..., S, *spatial]
        return x.reshape(*x.shape[:-1], *self.member_shape)

    def ravel_spatial(self, multi_idx):      # (ix,) or (ix, iy) -> flat spatial int (C-order)
        return np.ravel_multi_index(multi_idx, self.spatial_shape)

    def spatial_distance_matrix(self):       # [n_spatial, n_spatial], grid-cell Euclidean,
        ...                                  # per-axis periodic wrap (see PR 2)

    def enforce_periodicity(self, x): ...    # per-axis, replaces the 3 `.at[..., -1]` sites
```

**Conventions, documented once here and nowhere else:** flattening is C-order — `flat = state * n_spatial + ix * ny + iy`. All of localization, H-matrices, and score functions inherit consistency from using `ravel_spatial`/`flatten`.

**Threading it through:** `BaseForwardModel` constructs and owns `self.space` (it knows its grid). Backward compatibility: the existing `__init__(dt, steps, state_dim, num_states)` signature builds `StateSpace(num_states, (state_dim,))`, and `self.state_dim` / `self.num_states` become thin properties (`state_dim` returns `spatial_shape[0]` and raises for 2-D spaces, which flushes out any missed 1-D assumption loudly instead of silently). DA methods, observation operators, and ensemble generators receive the space via the forward model (`forward_operator.space`) exactly as they receive `num_states`/`state_dim` today — no new plumbing pattern.

## 2. PR breakdown

Prerequisite (PR 0, already covered by the earlier reports): land the `main.py`/config fixes and the obs-timing fix so the smoke suite is green — the refactor below must be protected by passing tests to be safe.

---

### PR 1 — `StateSpace` + shape-agnostic core (pure refactor, no behavior change)

1. Add `state_space.py` with the dataclass above (`spatial_distance_matrix` can raise `NotImplementedError` for now).
2. `BaseForwardModel`: own a `StateSpace`; accept either `state_dim: int` (legacy) or `spatial_shape: tuple`.
3. DA methods — the *only* edits they ever get:
   - `enkf.py`, `agmf.py`, `pff.py`: replace the entry flatten with `space.flatten(prior_ensemble)` (identical semantics) and the exit `reshape(E, num_states, state_dim)` with `space.unflatten(...)`. `particle_filter.py` needs nothing (it indexes the unflattened prior directly).
   - `da_methods/base.py:105`: `[:, :-1, :, :]` → `[:, :-1, ...]`, `[:, None, :, :]` → `[:, None, ...]`.
4. `time_integrators.py:60`: `trajectory.reshape(-1, num_states, state_dim)` → `trajectory.reshape((-1, *init.shape))` (the member shape is already in hand; delete the `num_states`/`state_dim` locals).
5. `main.py`: `initial_ensemble.copy().reshape(E, 1, S, D)` → `initial_ensemble[:, None]`; `prior_current[:, None, :, :]` → `[:, None, ...]`. This also decouples the script from `cfg.case.state_dim`.
6. `metrics/probability_metrics.py`: `reshape(n_pred, n_time, n_states * n_dim)` → `reshape(*shape[:2], -1)`; update docstrings in `trajectory_metrics`/`ensemble_metrics` to "trailing state shape" (their `compute` bodies are already reductions over everything, so they need no code change). `ensemble_spread`: rename the misnamed `state_dim` selector arg to `state_index` and reduce over all axes except (ensemble, time).
7. **The regression test that makes the whole plan safe** — 1-D/2-D equivalence: run each (case, method) on Lorenz 96 configured as `spatial_shape=(40,)` and as `spatial_shape=(40, 1)` (same seed) and assert allclose posteriors. Any accidental shape fork in any filter fails this test forever after. Add shape unit tests for `flatten`/`unflatten`/`ravel_spatial` round-trips.

Deliverable: repo behaves byte-identically for all existing configs; 5-D arrays pass through the filters (with localization and 2-D obs still pending).

---

### PR 2 — Localization on arbitrary grids

Current `distance_based_localization` builds `|i−j|` on a 1-D index and tapers with a truncated Gaussian (not PSD — see math report §1.5). Replace the geometry *and* the taper together:

1. `StateSpace.spatial_distance_matrix()`: per-axis index distances via broadcasting, per-axis periodic wrap (`min(d, n−d)` on the axes flagged periodic), combined as Euclidean grid-cell distance:
   ```python
   deltas = [pairwise_axis_dist(n, per) for n, per in zip(spatial_shape, periodic)]
   dist = sqrt(sum(d_a**2 broadcast over the meshgrid))   # [n_spatial, n_spatial]
   ```
   Optionally scaled by `domain_lengths[a] / n_a` for physical distances (keep grid-cell units as the default so existing `localization_distance` configs keep meaning what they meant in 1-D).
2. Gaspari–Cohn taper `gc(dist / r)` — compactly supported and PSD, replacing the truncated Gaussian (drop-in; one function).
3. `distance_based_localization(space, r, cov)` → `cov * tile(gc_mask, (S, S))`. The old signature stays as a deprecated wrapper for one release.
4. Cache the mask: it depends only on `(space, r)` — compute once in each filter's `__init__` (also removes per-step mask construction from today's hot path).
5. **Scaling honesty (must go in the docstring and the PR description):** the mask and the covariance are `n_spatial²`-sized. Rough budget (float32): 32×32 → 1M entries (4 MB) trivial; 64×64 → 16.8M (67 MB) fine; 128×128 → 268M (>1 GB) **not viable** with the current full-covariance EnKF/AGMF/PFF formulation. So this PR makes grids up to ~64² work; genuinely large 2-D grids need PR 6. Setting expectations here avoids "it works on the demo, OOMs on the real case" surprises.
6. Tests: 2-D distance matrix vs. brute force on a 4×5 grid (periodic and not); PSD check — smallest eigenvalue of `gc_mask ≥ 0` and of a localized random-ensemble covariance ≥ −ε; 1-D results unchanged vs. PR 1 baseline (modulo the intended Gaussian→GC taper change, which should be its own commit to keep the geometry change bit-exact).

---

### PR 3 — Observation operators for 2-D sampling

The H-matrix builder needs exactly one change: the flat index becomes `state_idx * n_spatial + space.ravel_spatial(spatial_idx)`. The work is in the *user-facing index specification*:

1. `_normalize_obs_indices` accepts, per observed state:
   - a 1-D array of ints → interpreted as flat spatial indices (works for both 1-D and 2-D; backward compatible);
   - an array of shape `[k, ndim_spatial]` → multi-indices, raveled via `space.ravel_spatial`;
   - a boolean mask of shape `spatial_shape` → `np.flatnonzero`.
   All normalize to flat arrays, so `obs_indices_per_state`, `num_obs`, and the per-state slicing in `main.py` keep working untouched.
2. Config helpers (tiny factory functions, because writing 2-D index lists in YAML is miserable):
   ```python
   def strided_grid_indices(spatial_shape, steps): ...      # every (sx, sy)-th point
   def block_indices(spatial_shape, lo, hi): ...            # observe a subdomain
   ```
   used as `_target_`s in YAML the same way `numpy.arange` is used today.
3. `LinearObservationOperator` takes `space` (or `state_dim`+`num_states` legacy) — `_obs_operator`'s `x.flatten()` already handles any trailing shape.
4. Nonlinear operators: nothing shape-specific once the Jacobian fix from the PFF report lands (`jax.jacobian` of `h(space.flatten(x))`).
5. Tests: H round-trip on a 3×4 grid (place a delta at `(ix, iy)`, observe it at the right row); strided-grid helper vs. manual enumeration; 1-D configs unchanged.

---

### PR 4 — Ensemble generation, profiles, perturbations, periodicity

1. **One spectral-noise engine instead of three.** `red_noise._powerlaw_rednoise_periodic_1d` and `CoupledKuramotoPseudo1DProfile._smooth_gaussian_periodic_1d` are the same algorithm with different spectra. Factor:
   ```python
   def spectral_field(rng_key, spatial_shape, domain_lengths, spectrum_fn):
       k = wavenumber_magnitude_grid(spatial_shape, domain_lengths)   # rfftn freqs, |k|
       coefs = (normal + 1j*normal) * spectrum_fn(k);  coefs[0-mode] = 0
       field = jnp.fft.irfftn(coefs, s=spatial_shape)
       return (field - mean) / std
   ```
   `jnp.fft.rfftn/irfftn` + a meshgrid of per-axis `fftfreq`s make this dimension-generic with **zero** 1-D/2-D branching. `RedNoise` passes `spectrum_fn = k^(-α/2)` (with the k=0 guard), the GRF profile passes `exp(−(kL)²/2)`. Rename `CoupledKuramotoPseudo1DProfile` → `GaussianRandomFieldProfile` (keep an alias).
2. `WhiteNoise` and `ConstantProfile`: already shape-driven — just build shapes from `space.member_shape`.
3. `CosineProfile`: generalize to per-axis modes (`cos(2πx/Lx)`, optionally `· cos(2πy/Ly)`), or keep 1-D-only with a clear error — decide by whether a 2-D case needs it (the GRF profile is the more useful 2-D initializer).
4. Periodicity: replace the three `.at[..., -1].set(x[..., 0])` sites with `space.enforce_periodicity(x)` looping the flagged axes. Also worth reconsidering whether this operation should exist at all for spectral models (they are periodic by construction; overwriting the last grid point *breaks* spectral fields) — recommend making it opt-in per axis via `StateSpace.periodic` and defaulting it off for the KS-family cases.
5. `perturbations/base.py`, `InitialState`, `InitialEnsembleGenerator`: accept `space` (or legacy `num_states`+`state_dim`); all their array logic is already shape-agnostic.
6. Breeding: audit rather than rewrite — norms already `ravel`, rollouts are shape-agnostic; generalize `_singelton_ensemble_axis` to check against `space.member_shape` instead of `ndim in (2, 3)`. Add a breeding smoke test on a small 2-D model.
7. `spinup` and `saving`: no changes needed (verified: reductions and `jnp.save` are shape-agnostic); `creat_exp_name` should render `spatial_shape` as `64x64`.

---

### PR 5 — A 2-D case end-to-end: forward models, configs, plotting, smoke tests

1. **2-D linear advection** (`u_t + c·∇u = 0`, periodic, semi-Lagrangian or upwind — ~30 lines): the validation model. It is *linear*, so a small grid version can also run through the analytical-KF harness in `scripts/analytical/` for an exactness test of the whole 2-D pipeline against the closed-form posterior.
2. **2-D Kuramoto–Sivashinsky** (`u_t + ½|∇u|² + Δu + Δ²u = 0`): the production 2-D chaotic testbed. The existing ETDRK machinery generalizes directly — the linear operator becomes `k² − k⁴` with `k² = kx² + ky²` on the `rfftn` grid, dealiasing mask per axis; ~80% shared with the 1-D implementation (worth factoring an `ETDRKSpectralModel` base out of the two KS classes at this point). Alternative/additional, more geophysical: barotropic vorticity on a doubly periodic domain — defer unless the paper needs it.
3. `configs/case/advection_2d.yaml`, `ks_2d.yaml`: `spatial_shape: [64, 64]` replaces `state_dim` (schema: accept either key; `state_dim: N` ≡ `spatial_shape: [N]`), strided-grid observations from PR 3, GRF initial profile + red-noise perturbation from PR 4, localization radius from PR 2.
4. Plotting: `plot_2d_field` — panel of snapshots (truth / ensemble mean / error / spread) at selected assimilation times, plus RMSE-vs-time reuse from the existing metric plots. Hovmöller stays for 1-D; dispatch on `space.ndim_spatial` in the case config (`plotter:` target), not in library code.
5. Tests: add the 2-D case × all four DA methods to `tests/test_main.py` (small grid, e.g. 32×32, few steps); keep the PR 1 equivalence test as the permanent guard.

---

### PR 6 (follow-up, optional) — scalable localization for large 2-D grids

Beyond ~64², the explicit `dofs×dofs` covariance is the binding constraint (PR 2 note). Two paths that both preserve "one implementation per method":

1. **Obs-space (R-/B-hybrid) localization inside the existing EnKF/AGMF**: work with anomalies, localize `X'(HX')ᵀ` (`dofs × p`) and `HX'(HX')ᵀ` (`p × p`) with distance masks between grid points and observation locations — never form `P`. Memory `O(dofs·p)`. This is an internal rewrite of the *gain computation* in the one EnKF implementation; the analysis semantics and interface are unchanged, and the 1-D/2-D equivalence test plus the analytical-KF test verify it.
2. **LETKF** as a new method (already recommended in the math report §3.1): domain-localized, trivially parallel over grid points, no covariance matrix at all — the standard answer for 2-D geophysical grids.

The PFF needs its own treatment at that scale (its Gaussian prior score wants `P⁻¹`): the ensemble-space score form (score computed via anomaly factors, `O(dofs·N)`) slots into the `flows/scores.py` module from the PFF report — another reason to land that refactor first.

---

## 3. Summary of guarantees and risks

- **"Write no DA method twice"** is achieved structurally: PR 1 removes the only shape-aware lines the filters contain, and the equivalence test (`(N,)` vs. `(N,1)`) enforces it permanently. Localization, observation indexing, and noise generation are the components that genuinely change — each is generalized in one place (`StateSpace`, one H-builder, one spectral engine) rather than per consumer.
- **Interactions with the other reports:** land the bug fixes (obs timing, config breakage) first so tests protect the refactor; the PFF `flows/` refactor is independent of PRs 1–5 but PR 6's ensemble-space score assumes it. The Gaspari–Cohn change in PR 2 fixes the PSD defect from the math review as a side effect.
- **Known limits stated up front:** full-covariance methods cap out around 64×64 grids until PR 6; `CosineProfile` and the Hovmöller plots stay 1-D; `LinearModel` works in 2-D automatically (it already operates on the flattened state) but writing a `4096×4096` transition matrix in YAML is not sensible — the 2-D advection model covers that role.
- **Effort estimate:** PR 1 ≈ 1 day (mostly tests), PR 2 ≈ 1 day, PR 3 ≈ 0.5–1 day, PR 4 ≈ 1–1.5 days, PR 5 ≈ 1.5–2 days (2-D KS + plots dominate), PR 6 open-ended. PRs 2–4 are independent of each other after PR 1 and can proceed in any order or in parallel.
