<p align="center">
  <img src="shrubbery_logo.png" alt="shrubbery logo" width="300"/>
</p>

# APPAC — Atmospheric Pressure Peak Area Correction

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RuedigerForster/shrubbery/main?labpath=appac_demo.ipynb)

A pure-**NumPy** implementation of the APPAC correction pipeline for gas-chromatograph
peak areas, with full GUM-compliant (analytic) covariance propagation.  Port of the
R package by the same author; R-specific extras (S4 classes, tidyverse, plotting)
are omitted.

> **Experimental branch.** Created to decide whether the data prefer a
> multiplicative or an additive error model. The result: undecided — the
> differences are marginal. (The additive variant is the `add_err` branch.)

## Purpose

Gas-chromatograph peak areas depend on ambient pressure, instrument drift, and
cross-instrument correlated noise.  APPAC fits a multiplicative correction model
for individual GC channels using sensor measurements of the ambient conditions
that prevailed during a run.  Applying the correction to the raw peak areas of an
unknown sample gives pressure-independent, drift-free results that are traceable
to a GUM uncertainty budget.

## Correction model

```
multiplier[t, j] = 1
    + episode_bias[episode(t), j]
    + (pc2_common[t] × pc2_amp[s] + poly_score(t, ep)) × pc2_load[j, s]
    + corr[t] × corr_scaling[j]
    + Σ_cv  kappa[cv] × (covariate[cv][t] − ref[cv])

corrected = raw / multiplier
```

- **κ is a single common value per covariate** (a detector property; an FID is
  carbon-equimolar, so the pressure sensitivity is not component-specific).
- **Centers are geometric means** `exp(mean(log Y))` — the unbiased "true value"
  for log-normal data.
- The PC2 drift term is present only after `fit_uncorrelated_drift()` has been called.

## Repository layout

```
shrubbery/
  appac.py            Main module — fit, correct, covariance propagation
  load_rda.py         Load dataset files (Parquet or .rda) into SampleData objects
  control_updater.py  Sequential Bayesian bias tracking for new observations
  __init__.py         Public API re-exports
pyproject.toml
appac_demo.ipynb      Tutorial notebook: pipeline walkthrough + diagnostic plots
run_appac_on_data.py  Run the pipeline on a dataset and print a text report
plot_real_data.py     Reference pipeline script on PLOT_FID (with plots)
test_appac.py         Integration smoke test (full pipeline on synthetic data)
test_appac_unit.py    Targeted unit tests (25 tests, no data file required)
test_control_updater.py
data/
  PLOT_FID.parquet    Identical to APPAC's dataset: 6 samples × 5 peaks × ~14 000 injections
```

The InfluxDB/MongoDB write layer (`appac_db.py`) is shipped **separately** and is
not part of the versioned package or the wheel.

## Installation

**From a wheel** (recommended):

```bash
pip install shrubbery-0.1.1-py3-none-any.whl
```

**From source**:

```bash
pip install numpy scipy pandas pyarrow matplotlib
pip install -e .
```

> **Reading `.rda` files:** the bundled dataset is Parquet and needs no extra
> reader.  For legacy `.rda` files install `rdata` (`pip install rdata`); do **not**
> use `pyreadr` — it segfaults on ARM64.

The package depends only on NumPy and SciPy at runtime; there is **no JAX
dependency**.  It imports and runs on a stock scientific-Python install.

## Quick start

```python
from shrubbery import fit, fit_uncorrelated_drift, build_multiplier, correct
from shrubbery.load_rda import load_samples

samples, breakpoints, p_ref = load_samples("data/PLOT_FID.parquet")

model = fit(
    samples,
    covariate_refs={"pressure": p_ref},
    breakpoints=breakpoints,
    exclude=["CTL-4"],          # omit unstable instruments from shared parameters
)
model = fit_uncorrelated_drift(samples, model)   # optional PC2 slow-drift term

for s in samples:
    M  = build_multiplier(s, model)
    Yc = correct(s.Y, M)        # corrected = raw / multiplier
```

For an end-to-end run with a printed report (κ, detected breakpoints,
per-component correction quality, variance explained, and a GUM uncertainty
sample) on the bundled dataset:

```bash
python run_appac_on_data.py                     # uses data/PLOT_FID.parquet
python run_appac_on_data.py DATA --out report.txt
```

For a narrated walkthrough with diagnostic plots (κ fit, before/after time series,
RSD, residual normality, GUM uncertainty), open `appac_demo.ipynb` in Jupyter and
run it from the repository root — or click the **Binder** badge at the top to run
it in your browser with no local install.

## Dependencies

| Package | Required | Used for |
|---|---|---|
| numpy | core | all numerics |
| scipy | core | F-test for polynomial degree selection |
| pandas | data loading | `load_samples` (Parquet/`.rda` → `SampleData`) |
| pyarrow | data loading | Parquet reader |
| matplotlib | optional | the `plot_*.py` scripts |
| rdata | optional | legacy `.rda` reader (not a wheel dependency) |

## Pipeline steps

| Step | Function | Description |
|------|----------|-------------|
| 1 | `_estimate_kappa` | Common κ per covariate — WLS on binned, **ep0-referenced** log-area deviations pooled over peaks and cylinders (no PCA, no z-score) |
| 2 | `_covariate_correct` | Divide by `1 + Σ κ·(covariate − ref)` |
| 3 | centers + `_fit_dispersion` | Centers = geometric mean `exp(mean(log Y))`; fit dispersion `sd ~ center + center²` and z-score-normalise (prep for the daily-factor PCA) |
| 4 | `_transform_features` | PCA on the κ-corrected data → correlated component (PC1) |
| 5 | `_estimate_episode_bias` | Geometric mean per breakpoint episode |
| 6 | `_estimate_correlated_model` | Cross-sample daily factor (PC1 across concurrent cylinders) |
| 7 | `fit_uncorrelated_drift` | PC2 slow drift: common component + per-sample polynomial |
| 8 | `build_multiplier` + `correct` | Assemble `M[t,j]` and divide |
| 9 | `propagate_covariance` | Analytic GUM Σ from five uncertainty sources |

The arithmetic/geometric-mean bias is removed **by design** — κ is ep0-referenced
and centers are geometric means — so there is no separate χ² center-refinement step.

## Uncertainty budget (GUM Eq. 13)

```
Σ_out[i,i'] = Σ_Y[i,i'] / (M_i · M_i')                  [measurement]
            + Σ_cv  var(κ_cv) · s_κ[i] · s_κ[i']         [κ per covariate, shared across peaks]
            + Σ_{e,j}  var(b_{e,j}) · s_b[i] · s_b[i']   [episode bias]
            + Σ_t  dcv[t] · s_c[i] · s_c[i']             [daily_corr between-sample]
            + Σ_{e,ci}  var(a_{e,ci}) · s_p[i] · s_p[i'] [PC2 polynomial coefficients]
```

All Jacobians are written out analytically — no automatic differentiation.

## Generalised correction for unknown samples

The full APPAC model uses composition-specific center values and cannot be applied
to a sample of unknown composition.  The *generalised* correction removes this
restriction by exploiting a single physically universal parameter — the pressure
sensitivity κ — together with the cross-instrument drift signal derived from the
control samples.

### Four-step pipeline architecture

**Step 1 — Unbiased κ**

```python
model0 = fit(samples, covariate_refs, exclude=EXCLUDE)   # no breakpoints
model0 = fit_uncorrelated_drift(samples, model0)
kappa_unbiased = model0.kappa                            # fixed from here on
```

κ is the pressure sensitivity of the GC detector — a physical constant shared by
all samples on the same channel.  It is **unbiased by construction**:

- **No arithmetic/geometric bias.** κ is regressed on *ep0-referenced* log-area
  deviations `log(Y) − log(ep0)`, where `ep0` is the geometric mean of episode 0;
  the model centers are geometric means as well.  The arithmetic mean of
  log-normal areas exceeds the geometric mean by `exp(σ²/2)`, but that offset
  never enters here — so no χ² center-refinement step is required.
- **No breakpoint contamination.** κ is estimated on the full dataset with a
  single episode per sample (before any breakpoints), giving the widest pressure
  range and the cleanest regression.  The per-cylinder ep0 reference is an
  additive log offset and does not bias the slope, so cylinders of differing
  composition can be pooled.

κ is fixed after Step 1 and never re-estimated.

**Step 2 — Pressure correction**

```python
Y_kappa = _covariate_correct(Y, covariates, kappa_unbiased, covariate_refs)
```

The only correction step that generalises to unknown composition: it depends only
on the detector, not on the sample.

**Step 3 — Missing-value imputation (PPCA)**

The control samples are not all measured every day.  PPCA soft-impute fills the
gaps in the `(n_days × n_samples)` daily-log-ratio matrix using the cross-instrument
correlation structure, so a step visible in two contemporaneous samples is
propagated coherently to days when a third was not running (the Python equivalent
of `bpca()` from the BioConductor *pcaMethods* package).

**Step 4 — Breakpoint and drift detection**

Binary segmentation (Bhattacharyya distance + sharpness filter) runs on the
PPCA-imputed PC2 score — the cross-instrument common drift.  Only instrument-wide
events (cylinder swap, detector service) produce a sharp spike and are promoted to
global breakpoints; single-sample steps average to near-zero in PC2.

```python
gen_model0  = fit_generalized_correction(samples, model0, kappa_fixed=kappa_unbiased)
global_bps  = detect_breakpoints_global(gen_model0)
breakpoints = apply_global_breakpoints(samples, global_bps)
```

The episode-bias and drift terms are then fitted with the detected breakpoints,
always using the fixed κ from Step 1.

### Generalised correction model

`fit_generalized_correction` returns a `GeneralizedCorrectionModel` with:

- `ep0_ref[s]` — geometric mean of κ-corrected areas in episode 0; the baseline
  every injection is referenced to.
- `pc1_scores`, `pc2_scores` — PPCA temporal scores (one per day) for the
  correlated daily factor (PC1) and slow drift/bias (PC2).
- `pc1_loading[s]`, `pc2_loading[s]` — per-sample amplitudes.

The corrected peak-mean log-ratio for any injection is

```
r[t] = mean_j( log(Y_κ[t,j] / ep0_ref[j]) )
     ≈ pc1_score[t] × pc1_loading[instrument]
     + pc2_score[t] × pc2_loading[instrument]
```

Equal peak weights make `r[t]` independent of composition — valid for unknown
samples measured on a known instrument.

### Sequential drift tracking (GeneralizedUpdater)

`GeneralizedUpdater` (in `control_updater.py`) tracks drift in real time as new
injections arrive, without knowing the sample composition.  It uses a
Normal-Inverse-Gamma conjugate prior (closed-form posterior after each injection)
and a Shiryaev-Roberts statistic for online breakpoint detection.  Initialise it
from a fitted `GeneralizedCorrectionModel` via
`GeneralizedUpdater.from_generalized_model()`.

### Design rationale — separation of concerns

| Parameter | Estimated from | Generalises to unknown samples? |
|---|---|---|
| κ (pressure sensitivity) | all stable samples, no breakpoints, ep0-referenced | yes — physical constant of the detector |
| episode bias | per sample, per episode, after κ correction | no — sample-specific |
| PC1/PC2 drift scores | cross-sample PPCA, after κ correction | yes — instrument-wide signal |
| PC1/PC2 loadings | per sample from training data | yes — for samples in the training set |
| ep0_ref | geometric mean of episode-0 areas, per sample | yes — baseline reference |

Only the κ correction and the temporal PC1/PC2 scores are needed to correct any
sample on a known instrument; the composition plays no role.

### Detecting degrading control samples

A control sample whose composition is changing (contamination, permeation,
adsorption) drifts in a way that is *not* correlated with the others and so does
not appear in PC2.  It shows up as a large, growing PC2 loading or a systematic
PPCA residual, and should be moved to the `exclude` list so it does not
contaminate the shared κ and drift estimates.  In the PLOT_FID dataset, CTL-4 was
identified this way and excluded.

## Running the tests

```bash
python test_appac.py            # integration test (needs data/PLOT_FID.parquet)
python test_appac_unit.py       # unit tests (no data file; runs on stock NumPy)
python test_control_updater.py  # sequential updater tests
python plot_real_data.py        # full pipeline on real data + time-series plots
```

## A note on the name

This repository is an homage to *Monty Python and the Holy Grail*.  A shrubbery —
one that looks nice, but is not too expensive — was the tribute the Knights Who Say
"Ni!" demanded from King Arthur in exchange for safe passage through their forest.

## License

Copyright © 2026 Rüdiger Forster.

**shrubbery** is free software: you can redistribute it and/or modify it under the
terms of the **GNU General Public License v3.0** as published by the Free Software
Foundation.  This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the [LICENSE](LICENSE) file for the full,
verbatim GPL-3 text.
