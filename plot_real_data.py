"""
Full pipeline + time series plots on real PLOT_FID.rda data.

Pipeline  (four-step architecture)
--------
Step 1 — Unbiased κ
  1a. fit (no breakpoints)
  1b. fit_uncorrelated_drift
  1c. chi_square_fit  ← first unbiased κ; fixed for all subsequent steps

Step 2 — Pressure correction
  Applied inside fit_generalized_correction using kappa_fixed.

Step 3 — PPCA missing-value imputation
  Applied inside fit_generalized_correction (_ppca_nan).

Step 4 — Breakpoint and drift detection
  2a. fit_generalized_correction (PPCA, kappa_fixed) → gen_model0
  2b. detect_breakpoints_global on PC2 scores
  2c. fit (breakpoints, kappa_fixed) + fit_uncorrelated_drift
  2d. flag_outliers
  2e. fit_generalized_correction (final, PPCA, kappa_fixed) → gen_model
  2f. GeneralizedUpdater on pilot sample

Step 5 — Time series plots
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shrubbery.load_rda import load_samples
from shrubbery import run_pipeline
from shrubbery.appac import _covariate_correct
from shrubbery.control_updater import GeneralizedUpdater

EXCLUDE = ["CTL-4"]

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading PLOT_FID.rda...")
samples, _, p_ref = load_samples("data/PLOT_FID.rda")
stable_names = [s.name for s in samples if s.name not in EXCLUDE]
print(f"\n  Stable samples: {stable_names}")
print(f"  Excluded:       {EXCLUDE}")

# ══ Pipeline — shared single source with run_appac_on_data.py ═════════════════
# The full validated sequence (unbiased χ²-refined κ → instrument-wide
# breakpoints → refit with κ fixed → outlier ∪ dirty-window flagging) lives in
# shrubbery.run_pipeline, so these plots and the headless report never diverge.
print("\n── Running validated APPAC pipeline ──")
pr = run_pipeline(samples, {"pressure": p_ref}, exclude=EXCLUDE)
kappa_unbiased = pr.kappa
model1     = pr.model
gen_model  = pr.gen_model
masks      = pr.masks
bp_result  = pr.bpr

# κ diagnostics from the χ²-refined no-breakpoint model
kappa_se  = float(np.sqrt(pr.m0.kappa_var["pressure"]))
kappa_rsd = kappa_se / abs(kappa_unbiased["pressure"])
dp_b, w_b, y_b, y_pred = pr.m0.kappa_bins["pressure"]
kap_wmse  = float(np.sqrt(np.sum(w_b * (y_b - y_pred) ** 2) / np.sum(w_b)))
corr_rng  = float(kappa_unbiased["pressure"] * (dp_b.max() - dp_b.min()))
print(f"  κ (unbiased) = {kappa_unbiased['pressure']:.4e} hPa⁻¹  "
      f"R²={pr.m0.kappa_rsq['pressure']:.4f}  SE={kappa_se:.2e} hPa⁻¹  RSD={kappa_rsd*100:.4f}%")
print(f"  κ bin WMSE = {kap_wmse*100:.4f}%  |  total correction range = {corr_rng*100:.3f}%")
print(f"  breakpoints: {bp_result.breakpoints.tolist()}  |  dirty windows: {bp_result.dirty_windows}")
for s in samples:
    n = int(masks[s.name].sum())
    print(f"  {s.name}: {n} flagged / {len(s.dates)} ({n/len(s.dates)*100:.1f}%)")
for name in stable_names:
    print(f"  pc1_loading[{name}] = {gen_model.pc1_loading[name]:+.4f}  "
          f"pc2_loading[{name}] = {gen_model.pc2_loading[name]:+.4f}")

# ══ STEP 4f: GeneralizedUpdater on pilot sample ═══════════════════════════════
pilot = next(s for s in samples if s.name == stable_names[0])
print(f"\n── Step 4e: GeneralizedUpdater on {pilot.name} ──")
updater = GeneralizedUpdater.from_generalized_model(
    gen_model, model1, samples, instrument=pilot.name
)
valid = ~masks[pilot.name]
results = updater.update_batch(
    pilot.Y[valid],
    {cv: arr[valid] for cv, arr in pilot.covariates.items()},
    pilot.dates[valid],
)
print(updater.summary())

# ══ STEP 4f: Write to databases ═══════════════════════════════════════════════
# Database persistence is intentionally OMITTED here.  The old MongoDB/InfluxDB3
# layer (shrubbery.appac_db) has been dropped — InfluxDB3 failed to recover from
# a WAL corruption in field testing and is unfit for production.  A PostgreSQL
# backend will replace it; until then this script only reports and plots.

# ══ STEP 5: Time series plots ══════════════════════════════════════════════════
print("\nGenerating plots...")

cmap        = plt.get_cmap("tab10")
sample_cols = {s.name: cmap(i) for i, s in enumerate(samples)}
ep_colors   = ["C0", "C2", "C3", "C4", "C5", "C6", "C7"]

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
fig.suptitle("APPAC — Generalised Correction Model  |  PLOT_FID real data", fontsize=13)

# ── Panel 1: per-sample daily log-ratios + PCA reconstruction ────────────────
ax = axes[0]
for s in samples:
    col    = sample_cols[s.name]
    Y_kap  = _covariate_correct(s.Y, {"pressure": s.covariates["pressure"]},
                                 kappa_unbiased, model1.covariate_refs)
    ct_ep0 = gen_model.ep0_ref[s.name]
    r      = np.log(Y_kap / ct_ep0[None, :]).mean(axis=1)
    valid_s  = ~masks[s.name]
    unique_d = np.unique(s.dates[valid_s])
    r_daily  = np.array([r[valid_s][s.dates[valid_s] == d].mean() for d in unique_d])
    ax.scatter(unique_d, r_daily * 100, s=2, alpha=0.25, color=col, zorder=2)

    idx   = np.searchsorted(gen_model.dates_global, unique_d)
    idx   = np.clip(idx, 0, len(gen_model.dates_global) - 1)
    pc1_t = np.where(np.isnan(gen_model.pc1_scores[idx]), 0.0, gen_model.pc1_scores[idx])
    pc2_t = np.where(np.isnan(gen_model.pc2_scores[idx]), 0.0, gen_model.pc2_scores[idx])
    recon = (pc1_t * gen_model.pc1_loading[s.name]
             + pc2_t * gen_model.pc2_loading[s.name])
    ax.plot(unique_d, recon * 100, color=col, lw=1.0,
            label=s.name + (" [excl]" if s.name in EXCLUDE else ""), zorder=3)

    for bp in model1.breakpoints.get(s.name, []):
        ax.axvline(bp, color=col, lw=0.6, ls="--", alpha=0.4)

ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.set_ylabel("Peak-mean log-ratio (%)")
ax.set_title("κ-corrected (unbiased κ), ep0-referenced log-ratios (dots) + PPCA reconstruction (lines)"
             "   |   dashed verticals = global breakpoints")
ax.legend(fontsize=8, ncol=len(samples), loc="upper left")
ax.grid(True, alpha=0.2)

# ── Panel 2: PC1 and PC2 temporal scores ──────────────────────────────────────
ax = axes[1]
v1 = np.isfinite(gen_model.pc1_scores)
v2 = np.isfinite(gen_model.pc2_scores)
ax.plot(gen_model.dates_global[v1], gen_model.pc1_scores[v1],
        lw=0.8, alpha=0.85, label="PC1 — correlated daily factor")
ax.plot(gen_model.dates_global[v2], gen_model.pc2_scores[v2],
        lw=0.8, alpha=0.85, label="PC2 — drift / bias")
for bp in bp_result.breakpoints:
    ax.axvline(bp, color="k", lw=1.0, ls="--", alpha=0.5)
if len(bp_result.breakpoints):
    ax.axvline(bp_result.breakpoints[0], color="k", lw=1.0, ls="--",
               alpha=0.5, label="clean breakpoints")
for lo, hi in bp_result.dirty_windows:
    ax.axvspan(lo, hi, color="r", alpha=0.10)
if bp_result.dirty_windows:
    ax.axvspan(*bp_result.dirty_windows[0], color="r", alpha=0.10,
               label="dirty window (contamination)")
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.set_ylabel("PCA score")
ax.set_title("PPCA cross-sample temporal scores  (breakpoints detected on PC2)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# ── Panel 3: GeneralizedUpdater sequential drift for pilot sample ─────────────
ax = axes[2]

pilot_dates = pilot.dates[valid]
for date, r_dict in zip(pilot_dates, results):
    ep = min(r_dict["episode"], len(ep_colors) - 1)
    ax.scatter(date, r_dict["peak_mean_log_ratio"] * 100,
               s=3, color=ep_colors[ep], alpha=0.3, zorder=2)

drift_by_date: dict[int, list] = {}
for date, r_dict in zip(pilot_dates, results):
    drift_by_date.setdefault(int(date), []).append(r_dict["drift_mean"])
plot_dates  = sorted(drift_by_date)
plot_drifts = [float(np.mean(drift_by_date[d])) for d in plot_dates]
ax.plot(plot_dates, np.array(plot_drifts) * 100,
        color="k", lw=1.5, label="posterior drift mean", zorder=4)

for rec in updater.history:
    ax.axvline(rec.end_date, color="r", lw=1.2, ls="--", alpha=0.6)
if updater.history:
    ax.axvline(updater.history[0].end_date, color="r", lw=1.2, ls="--",
               alpha=0.6, label="SR alarm")

ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.set_xlabel("Day (R integer date, days since 1970-01-01)")
ax.set_ylabel("Peak-mean log-ratio (%)")
ax.set_title(f"{pilot.name} — GeneralizedUpdater sequential drift tracking  "
             "(dots coloured by episode, red dashed = SR alarms)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

plt.tight_layout()
out = "generalized_timeseries_real.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"\nSaved → {out}")
