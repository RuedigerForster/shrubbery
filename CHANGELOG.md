# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] — 2026-06-27

### Added

- `run_appac_on_data.py` — run the full pipeline on a dataset and print a text
  report (κ, breakpoints, per-component correction quality, variance explained,
  and a GUM combined-uncertainty sample).
- `appac_demo.ipynb` — tutorial notebook: narrated pipeline walkthrough with
  diagnostic plots (κ fit, before/after time series, RSD, residual normality,
  GUM uncertainty).

### Changed

- **Dropped the JAX dependency — now pure NumPy.** `correct()` is plain
  `Y / multiplier`; the package imports and runs on a stock NumPy/SciPy install.
  (JAX had been used for a single element-wise division and was 13–17× slower
  than NumPy once the array conversions are counted.)
- **Renamed module `shrubbery.appac_jax` → `shrubbery.appac`** (it is no longer a
  JAX module); test files renamed accordingly. *Breaking: update imports.*
- **κ is now a single common value per covariate** (was per-peak), estimated on
  ep0-referenced log-area deviations pooled over peaks and cylinders — κ is a
  detector property and an FID is carbon-equimolar.
  *Breaking: `model.kappa[cv]` and `kappa_var[cv]` are now scalars, not arrays.*
- **Centers are geometric means** `exp(mean(log Y))` by design (the unbiased
  "true value" for log-normal data).
- κ binning vectorised with `np.bincount`; introduced a shared `_wls`
  weighted-least-squares helper.

### Removed

- **`chi_square_fit`** — obsolete: κ is unbiased via ep0-referencing and centers
  are geometric means by design, so the χ² center-refinement step (which targeted
  the biased arithmetic mean) is no longer needed.
- The InfluxDB 3 / MongoDB write layer (`appac_db`) is no longer part of the
  versioned package; it is distributed separately.

### Fixed

- `_estimate_kappa` now handles datasets where samples have different peak sets
  (it previously assumed a single global peak count and crashed — e.g. on the
  bundled PLOT_FID, where one cylinder has 4 peaks and the rest have 5).
- `propagate_covariance` now skips a non-finite κ variance (degenerate/constant
  covariate) instead of producing an infinite output covariance.

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
