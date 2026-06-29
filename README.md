<p align="center">
  <img src="shrubbery_logo.png" alt="shrubbery logo" width="300"/>
</p>

# APPAC — Atmospheric Pressure Peak Area Correction

A pure-NumPy implementation of the APPAC correction pipeline for GC peak areas,
with full GUM-compliant covariance propagation.  Port of the R package 
by the same author; R-specific extras (S4 classes, tidyverse, plotting) are omitted.

## Purpose

Gas chromatograph peak areas depend on ambient pressure, instrument drift, and
cross-instrument correlated noise. APPAC fits a multiplicative correction
model for individual GC channels using sensor measurements of the ambient 
conditions which prevailed during a GC run. Applying these corrections to 
raw peak areas of unknown samples gives pressure-independent, drift-free results, 
which are traceable to a GUM uncertainty budget.

## Correction model

```
multiplier[t, j] = 1
    + episode_bias[episode(t), j]
    + (pc2_common[t] × pc2_amp[s] + poly_score(t, ep)) × pc2_load[j, s]
    + corr[t] × corr_scaling[j]
    + Σ_cv  kappa[cv] × (covariate[cv][t] − ref[cv])

corrected = raw / multiplier
```

The PC2 drift term is only present after `fit_uncorrelated_drift()` has been
called.

## Repository layout

```
shrubbery/
  appac.py            Main module — fit, correct, covariance propagation
  pipeline.py         run_pipeline — the canonical end-to-end pipeline
  load_rda.py         Load dataset files (Parquet or .rda) into SampleData objects
  control_updater.py  Sequential Bayesian bias tracking for new observations
  __init__.py         Public API re-exports
pyproject.toml
run_appac_on_data.py  Headless pipeline run with a text report
appac_demo.ipynb      Tutorial notebook (Binder-runnable)
plot_real_data.py     Reference pipeline script on PLOT_FID.parquet
test_appac.py     Integration smoke test (full pipeline on synthetic data)
test_appac_unit.py  Targeted unit tests (25 tests, no data file required)
data/
  PLOT_FID.parquet    Identical to APPAC's dataset: 6 samples × 5 peaks × ~14 000 injections
```

## Installation

**From a wheel** (recommended):

```bash
pip install shrubbery-0.1.0-py3-none-any.whl
```

**From source**:

```bash
pip install numpy pandas pyarrow scipy matplotlib
pip install -e .
```

> **Note on the RData reader:** the dataset ships as Parquet and requires no
> extra reader.  If you have legacy `.rda` files, install `rdata` separately
> (`pip install rdata`); do **not** use `pyreadr` — it segfaults on ARM64.

## Quick start

```python
from shrubbery import fit, fit_uncorrelated_drift, build_multiplier, correct
from shrubbery.load_rda import load_samples

samples, breakpoints, p_ref = load_samples("data/PLOT_FID.parquet")

model = fit(
    samples,
    covariate_refs={"pressure": p_ref},
    breakpoints=breakpoints,
    exclude=["CTL-4"],         # omit unstable instruments from shared parameters
)

model = fit_uncorrelated_drift(samples, model)

for s in samples:
    M  = build_multiplier(s, model)
    Yc = correct(s.Y, M)
```

## Dependencies

```
numpy                >= 1.24   Core numerical backend
pandas               >= 2.0    Long-format dataset loading
scipy                          Moments, F-test for polynomial degree selection
pyarrow                        Parquet reader for the bundled dataset
matplotlib                     Plotting (plot_*.py scripts, notebook)
rdata                >= 0.9   legacy .rda reader (optional, not in wheel deps)
```

## Pipeline steps

| Step | Function | Description |
|------|----------|-------------|
| 1 | `_fit_dispersion` | Fit sd ~ center + center² across all peaks |
| 2 | `_transform_features` | Per-sample PCA; dispersion-normalised log-ratios |
| 3 | `_estimate_kappa` | Log-space WLS on binned means; κ per covariate |
| 4 | `_covariate_correct` | Divide by 1 + κ·ΔP |
| 5 | Second PCA pass | On covariate-corrected data |
| 6 | `_estimate_episode_bias` | Geometric mean per breakpoint episode |
| 7 | `_estimate_correlated_model` | Cross-sample daily factor via PCA |
| 8 | `fit_uncorrelated_drift` | PC2 slow drift: common + per-sample polynomial |
| 9 | `chi_square_fit` | χ² refinement of center values — essential for unbiased results |
| 10 | `build_multiplier` | Assemble full correction multiplier M[t, j] |
| 11 | `propagate_covariance` | Analytic GUM Σ from 5 uncertainty sources |

## Uncertainty budget (GUM Eq. 13)

```
Σ_out[i,i'] = Σ_Y[i,i'] / (M_i · M_i')                  [measurement]
            + Σ_cv  var(κ_cv) · s_κ[i] · s_κ[i']         [κ per covariate]
            + Σ_{e,j}  var(b_{e,j}) · s_b[i] · s_b[i']   [episode bias]
            + Σ_t  dcv[t] · s_c[i] · s_c[i']             [daily_corr between-sample]
            + Σ_{e,ci}  var(a_{e,ci}) · s_p[i] · s_p[i'] [PC2 polynomial coefficients]
```

## Generalised correction for unknown samples

The full APPAC model requires knowing which peaks to fit and uses
composition-specific center values.  It cannot be applied to a sample of
unknown composition.  The *generalised* correction removes this restriction by
exploiting a single physically universal parameter — the pressure sensitivity κ
— together with the cross-instrument drift signal derived from the control
samples.

### Four-step pipeline architecture

The generalised pipeline is deliberately separated into four steps, each with a
clear physical justification.

**Step 1 — Unbiased κ**

```python
model0 = fit(samples, covariate_refs, exclude=EXCLUDE)          # no breakpoints
model0 = fit_uncorrelated_drift(samples, model0)
model0 = chi_square_fit(samples, model0, covariate_refs)
kappa_unbiased = model0.kappa                                    # fixed from here on
```

κ is the pressure sensitivity of the GC detector.  It is a physical constant
shared by all samples on the same channel.  Two sources of bias affect κ if
it is estimated naively:

- *Arithmetic/geometric mean confusion.* `_estimate_kappa` forms log-ratios
  `log(Y / ct)` where `ct` is the arithmetic mean of the peak areas.  For
  log-normal data the arithmetic mean exceeds the geometric mean by
  `exp(σ²/2)`, and because σ² depends on the peak level which itself varies
  with pressure, this introduces a spurious pressure-dependent offset in the
  log-ratios.  `chi_square_fit` corrects the centers toward the geometric mean
  and thereby removes this offset.  κ is only unbiased after this step.

- *Breakpoint contamination.* If breakpoints (cylinder changes, detector
  cleaning) are inserted before κ is estimated, each episode acquires its own
  mean offset.  In short episodes the pressure range is limited and the episode
  offset can be confounded with the pressure slope, distorting κ.  Estimating
  κ on the full dataset with a single episode per sample — before any
  breakpoints — gives the cleanest regression.

κ is therefore fixed after Step 1 and never re-estimated.

**Step 2 — Pressure correction**

```python
Y_kappa = _covariate_correct(Y, covariates, kappa_unbiased, covariate_refs)
```

This is the only correction step that generalises to samples of unknown
composition.  It depends only on the physical properties of the detector, not
on the composition of a sample.  After this step the areas are
pressure-independent.

**Step 3 — Missing value imputation (PPCA)**

The control samples are not all measured on every day.  Forming the
(n_days × n_samples) matrix of daily-mean log-ratios leaves many gaps.
Simple mean imputation (filling gaps with each sample's average) destroys
the cross-instrument correlation structure and makes step changes invisible on
days with sparse coverage.

Instead, PPCA soft-impute is used: starting from the mean-filled matrix, the
code iterates — fit a low-rank PCA, reconstruct the missing cells from the PCA
model, repeat — until the imputed values converge.  The imputed values reflect
the cross-instrument correlation structure, so a step that is visible in two
contemporaneous samples is propagated coherently to the days when a third
sample was not running.  This is the Python equivalent of `bpca()` from the
BioConductor *pcaMethods* package used in the original R pipeline.

**Step 4 — Breakpoint and drift detection**

Binary segmentation (Bhattacharyya distance + sharpness filter) is run on the
PPCA-imputed PC2 temporal score.  PC2 is the cross-instrument common drift
signal; a step that appears in only one sample averages to near-zero in PC2
and does not pass the detection threshold.  Only instrument-wide events — gas
cylinder swap, detector service — produce a sharp spike in the Bhattacharyya
profile and are promoted to global breakpoints.

```python
gen_model0 = fit_generalized_correction(samples, model0,
                                         kappa_fixed=kappa_unbiased)
global_bps  = detect_breakpoints_global(gen_model0)
breakpoints = apply_global_breakpoints(samples, global_bps)
```

The episode bias and drift terms are then fitted with the detected breakpoints,
always using the fixed κ from Step 1.

### Generalised correction model

The output of `fit_generalized_correction` is a `GeneralizedCorrectionModel`
containing:

- `ep0_ref[s]` — geometric mean of κ-corrected areas in episode 0 for each
  reference sample.  All subsequent observations are expressed as fractional
  deviations from this baseline, tracing every injection back to the first
  episode.
- `pc1_scores`, `pc2_scores` — PPCA temporal scores (one value per day) for
  the correlated daily factor (PC1) and slow drift/bias (PC2).
- `pc1_loading[s]`, `pc2_loading[s]` — per-sample amplitudes.

The corrected peak-mean log-ratio for any injection is:

```
r[t] = mean_j( log(Y_κ[t,j] / ep0_ref[j]) )
     ≈ pc1_score[t] × pc1_loading[instrument]
     + pc2_score[t] × pc2_loading[instrument]
```

Because equal peak weights are used (not composition-weighted), `r[t]` is
independent of the sample composition and valid for unknown samples measured on
a known instrument.

### Sequential drift tracking (GeneralizedUpdater)

`GeneralizedUpdater` in `control_updater.py` tracks drift in real time as new
injections arrive, without needing to know the sample composition.  It uses:

- A Normal-Inverse-Gamma (NIG) conjugate prior for sequential Bayesian
  updating — closed-form posterior after each injection, no iteration required.
- A Shiryaev-Roberts (SR) statistic for online breakpoint detection.  When the
  SR statistic exceeds a threshold, a new episode is opened and the prior is
  reset.  This catches events that were not present in the training data.

The updater is initialised from a fitted `GeneralizedCorrectionModel` via
`GeneralizedUpdater.from_generalized_model()`, which calibrates the prior
observation noise from the within-episode variance of the reference cylinders.

### Design rationale — separation of concerns

| Parameter | Estimated from | Generalises to unknown samples? |
|---|---|---|
| κ (pressure sensitivity) | all stable samples, no breakpoints, full pipeline | yes — physical constant of the detector |
| episode bias | per sample, per episode, after κ correction | no — sample-specific |
| PC1/PC2 drift scores | cross-sample PPCA, after κ correction | yes — instrument-wide signal |
| PC1/PC2 loadings | per sample from training data | yes — for samples in training set |
| ep0_ref | geometric mean of episode-0 areas, per sample | yes — used as baseline reference |

The clean separation means that the generalised correction can be applied to
any sample on a known instrument without retraining the model.  Only the
κ correction and the temporal PC1/PC2 scores are needed; the sample
composition plays no role.

### Detecting degrading control samples

A control sample whose composition is changing (contamination, permeation,
adsorption) will show a drift that is *not* correlated with the other samples
and therefore does not appear in PC2.  Such a sample is identified by a large
and growing PC2 loading relative to the others, or by a systematic residual
after the PPCA reconstruction.  It should be moved to the `exclude` list so
that it does not contaminate the shared κ and drift estimates.

In the PLOT_FID dataset, CTL-4 was identified this way and excluded.

## Running the tests

```bash
# Integration test (requires data/PLOT_FID.parquet)
python test_appac.py

# Unit tests (no data file needed)
python test_appac_unit.py

# Full pipeline on real data with time series plots
python plot_real_data.py
```

## A note on the name

This repository is an homage to *Monty Python and the Holy Grail*.  A shrubbery
— one that looks nice, but is not too expensive — was the tribute the Knights
Who Say "Ni!" demanded from King Arthur and his entourage in exchange for safe
passage through the Knights' forest.

## License

Copyright 2026 Rüdiger Forster. All rights reserved.  
See [LICENSE](LICENSE) for terms.
