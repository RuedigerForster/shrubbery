# Changelog

All notable changes to this project will be documented in this file.

> Version 0.1.1 is the parallel `mult_err` branch (multiplicative error model);
> this `add_err` branch goes 0.1.0 → 0.1.2 to keep the two version lines distinct.

## [0.1.2] — 2026-06-28

Additive-error (`add_err`) branch — validated arithmetic-mean centers with χ²
center refinement; corrected residuals judged in linear space.

### Changed

- Dropped JAX — `correct()` is pure NumPy; removed all jax/jnp usage and the
  `jax` dependency. Renamed `shrubbery/appac_jax.py` → `shrubbery/appac.py`
  (and the test files); the module no longer uses jax.
- `run_appac_on_data.py` and `appac_demo.ipynb` evaluate corrected residuals in
  linear space `(area − centre)/centre` (additive-Gaussian model), not
  `log(area/centre)`.

### Added

- `shrubbery.run_pipeline` / `PipelineResult` — the canonical end-to-end pipeline
  (unbiased χ²-refined κ → instrument-wide breakpoints → refit → outlier ∪
  dirty-window flagging). `run_appac_on_data.py` and `plot_real_data.py` now both
  drive it, so the text report and the reference plots can no longer diverge.
- `run_appac_on_data.py` — headless end-to-end pipeline with a text report.
- `appac_demo.ipynb` — tutorial notebook with embedded diagnostic plots;
  `requirements.txt` for Binder.

### Fixed

- `run_pipeline` now re-centers each cylinder on the mean of its fully-corrected
  areas (the "true value"), matching appac_v3's `the_true_value`. `fit` centers on
  the κ-corrected mean, but the episode-bias/drift terms shift the corrected mean
  off it for the minor peaks, leaving a per-cylinder offset (~0.17%) that
  dominated the heavy-peak residual variance. Re-centering removes it; pooled
  residual sd vs the true value drops 0.22% → 0.15%, matching the R reference.
- `pyproject.toml` build backend `hatchling.core` → `hatchling.build` (the former
  is not a valid PEP 517 backend, so `pip install .` / the Binder build failed).
- `run_appac_on_data.py` variance-explained now trims raw and corrected residuals
  with the same mask, so the ratio compares the identical set of points (was
  dividing a trimmed corrected variance by an untrimmed raw variance → optimistic).

### Removed

- Database layer: dropped `pymongo` + `influxdb-client-3` dependencies and
  untracked `shrubbery/appac_db.py`. InfluxDB3 failed to recover from a WAL
  corruption in field testing; a PostgreSQL backend will replace it.

## [0.1.0] — 2026-04-29

Initial release.

### Added

- `shrubbery.appac_jax` — JAX/NumPy implementation of the full APPAC correction
  pipeline: dispersion normalisation, PCA, κ estimation, covariate correction,
  episode bias, correlated daily factor, PC2 uncorrelated drift, χ² center
  refinement, GUM covariance propagation, outlier flagging, and breakpoint
  detection (per-sample Bhattacharyya + global PPCA-based).
- `shrubbery.load_rda` — load long-format `.rda` files into `SampleData` objects;
  handles R `IDate` integers and string dates, pivots to wide, drops sparse peaks.
- `shrubbery.control_updater` — sequential Bayesian bias tracking via Normal-
  Inverse-Gamma conjugate priors and Shiryaev-Roberts online change detection.
  `ControlUpdater` targets known-composition control cylinders; `GeneralizedUpdater`
  targets composition-independent peak-mean log-ratios (valid for unknown samples).
- `shrubbery.appac_db` — write pipeline results to InfluxDB 3
  (`corrected_areas`, `ppca_scores`, `updater_drift` tables) and MongoDB
  (`pipeline_runs`, `component_meta` collections) with a pending quality gate.
- Generalised correction pipeline for unknown-composition samples:
  `fit_generalized_correction`, `build_multiplier_unknown`,
  `detect_breakpoints_global`, `apply_global_breakpoints`, `flag_dirty_windows`.
- Python package structure (`pyproject.toml`, `shrubbery/__init__.py`) with full
  public API re-exports.
