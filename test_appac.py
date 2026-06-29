"""
Smoke test for appac.py — episode-bias model (no drift).

Signal calibrated from Slide 11 of Präsentation.pdf:
  PC1 (correlated artifacts) ≈ ±2–3 % peak area — dominant effect
  PC2 (instrument bias)      ≈ flat, ±0.05 %    — no drift at all
  Noise                      ≈ negligible

After dispersion normalisation removes nonlinearity compression, the true
κ is larger than the raw-data table values.  Using κ = −2×10⁻³ hPa⁻¹
(≈ 3× the paper table, consistent with the slide's observed ±2 % at σ_P=8 hPa).

Two episodes per sample: one breakpoint at day 200 simulates a maintenance
event (gas cylinder change) that shifts the bias by 0.3 %.

Checks:
  1. fit() runs without error
  2. κ sign correct (negative) and within one order of magnitude of truth
  3. κ R² > 0.3
  4. Corrected RSD < raw RSD for ≥ 80 % of peak/sample pairs
  5. Episode bias close to injected step (±50 % tolerance)
  6. propagate_covariance() returns positive diagonal
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed — saves to file
import matplotlib.pyplot as plt

from shrubbery.appac import (SampleData, fit, build_multiplier, correct,
                                  propagate_covariance, flag_outliers, fit_uncorrelated_drift)

# ── Ground truth ───────────────────────────────────────────────────────────────

N_SAMPLES      = 3
N_PEAKS        = 5
N_OBS          = 440       # per sample (73 % of 600 days)
N_DAYS         = 600
TRUE_KAPPA     = -2e-3     # hPa⁻¹ — post-dispersion-normalisation estimate
P_REF          = 1000.0    # hPa
P_STD          = 8.0       # hPa  → ±1.6 % pressure signal (matches slide)
TRUE_AREAS     = np.array([1.4e4, 1.1e4, 5.4e3, 1.4e3, 6.2e2], dtype=float)
NOISE_FRAC     = 0.001     # 0.1 % noise — negligible (slide confirms)
BIAS_EP0       = 0.000     # episode 0: no bias offset (reference)
BIAS_EP1       = 0.003     # episode 1: +0.3 % step at breakpoint
BREAKPOINT_DAY = 200       # maintenance event at day 200

def make_sample(name, seed_offset):
    rng_s = np.random.default_rng(42 + seed_offset)
    days = np.sort(rng_s.choice(N_DAYS, size=N_OBS, replace=False)).astype(int)
    pressure = rng_s.normal(P_REF, P_STD, size=N_OBS)
    episode = (days >= BREAKPOINT_DAY).astype(float)
    bias = BIAS_EP0 * (1 - episode) + BIAS_EP1 * episode     # step at day 200
    pressure_factor = 1.0 + TRUE_KAPPA * (pressure - P_REF)
    Y = (
        TRUE_AREAS[None, :]
        * (1.0 + bias)[:, None]
        * pressure_factor[:, None]
        * rng_s.normal(1.0, NOISE_FRAC, size=(N_OBS, N_PEAKS))
    )
    return SampleData(
        name=name,
        Y=Y,
        dates=days,
        covariates={"pressure": pressure},
    )

samples = [make_sample(f"S{i+1}", i * 100) for i in range(N_SAMPLES)]

# Provide known breakpoints (from maintenance log)
known_breakpoints = {s.name: np.array([BREAKPOINT_DAY], dtype=int)
                     for s in samples}

# ── Fit ───────────────────────────────────────────────────────────────────────

print("Fitting model (two episodes, no drift)...")
model = fit(
    samples,
    covariate_refs={"pressure": P_REF},
    covariate_names=["pressure"],
    breakpoints=known_breakpoints,
)

kappa_est = float(np.mean(model.kappa["pressure"]))   # per-peak; mean for the single-κ synthetic test
print(f"\n  κ  true={TRUE_KAPPA:.2e}  estimated (mean)={kappa_est:.2e}  hPa⁻¹")
print(f"     per-peak κ = {np.round(model.kappa['pressure'], 6)}")
print(f"  κ R²: {model.kappa_rsq['pressure']:.4f}")
print(f"  κ σ (mean):  {float(np.mean(model.kappa_var['pressure']))**0.5:.2e}  hPa⁻¹")

# ── Scatter diagnostic: binned means vs ΔP with κ fit ────────────────────────
dp_b, w_b, y_b, y_pred = model.kappa_bins["pressure"]
raw_res = y_b - y_pred

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(
    f"κ fit — binned means vs ΔP  (κ_mean={kappa_est:.2e} hPa⁻¹,"
    f"  R²={model.kappa_rsq['pressure']:.4f})",
    fontsize=11,
)

# Left: scatter of bin means + fitted line (marker size ~ bin count)
sz = np.sqrt(w_b) * 4
axes[0].scatter(dp_b, y_b * 100, s=sz, alpha=0.5, label="bin means")
axes[0].plot(dp_b, y_pred * 100, "r-", lw=1.5, label="κ fit")
axes[0].set_xlabel("ΔP (hPa)")
axes[0].set_ylabel("Mean fractional deviation (%)")
axes[0].set_title("Binned means vs ΔP")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Right: residuals vs ΔP (should show no trend)
axes[1].scatter(dp_b, raw_res * 100, s=sz, alpha=0.5)
axes[1].axhline(0, color="r", lw=1.5)
axes[1].set_xlabel("ΔP (hPa)")
axes[1].set_ylabel("Residual (%)")
axes[1].set_title("Residuals vs ΔP")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out_path = "kappa_scatter.png"
plt.savefig(out_path, dpi=150)
print(f"\n  Scatter plot saved → {out_path}")

print("\n  Episode bias (S1):")
for e, bias_row in enumerate(model.episode_bias["S1"]):
    print(f"    episode {e}: mean bias = {bias_row.mean()*100:+.4f} %"
          f"  (true: {[BIAS_EP0, BIAS_EP1][e]*100:+.3f} %)")

# ── Correction quality ────────────────────────────────────────────────────────

print("\nCorrection quality:")
improved = 0
total = 0
for s in samples:
    M = build_multiplier(s, model)
    Y_c = correct(s.Y, M)
    for j in range(N_PEAKS):
        if Y_c[:, j].std() / Y_c[:, j].mean() < s.Y[:, j].std() / s.Y[:, j].mean():
            improved += 1
        total += 1
    rsd_r = (s.Y.std(axis=0) / s.Y.mean(axis=0)).mean() * 100
    rsd_c = (Y_c.std(axis=0) / Y_c.mean(axis=0)).mean() * 100
    print(f"  {s.name}: RSD raw={rsd_r:.3f}%  corrected={rsd_c:.3f}%")

print(f"\n  Peaks improved: {improved}/{total}")

# ── Auto breakpoint detection ─────────────────────────────────────────────────
# A uniform step (same Δ for all peaks) is absorbed by the correlated daily
# factor and is therefore invisible to detect_breakpoints.  The test uses a
# peak-differential step so only the residual after correlated correction is
# tested; the sharpness test must also reject a linear ramp of equal amplitude.

print("\ndetect_breakpoints: step vs ramp sharpness test...")
rng_bp = np.random.default_rng(77)
_N, _P = 600, 5
_t = np.arange(_N, dtype=float)
_dates = _t.astype(int)
_step_load = np.array([0.001, 0.003, 0.001, 0.005, 0.004])  # non-uniform: survives corr model
_step_signal = np.where((_t >= 300)[:, None], _step_load[None, :], 0.0)
_ramp_signal = (_t[:, None] / _N) * _step_load[None, :]     # same total Δ, spread linearly
_noise = rng_bp.normal(0, 0.001, (_N, _P))

from shrubbery.appac import _bhattacharyya_distance
_min_seg = 14
def _profile(R):
    n = len(R)
    db = np.zeros(n)
    for k in range(_min_seg, n - _min_seg):
        db[k] = _bhattacharyya_distance(R[:k], R[k:])
    cands = db[_min_seg : n - _min_seg]
    best = db[_min_seg + int(np.argmax(cands))]
    return best / max(cands.mean(), 1e-12)

sharpness_step = _profile(_step_signal + _noise)
sharpness_ramp = _profile(_ramp_signal + _noise)
print(f"  Sharpness — step: {sharpness_step:.2f}  ramp: {sharpness_ramp:.2f}")
assert sharpness_step > 2.0,  f"step sharpness too low: {sharpness_step:.2f}"
assert sharpness_ramp < 1.5,  f"ramp sharpness too high: {sharpness_ramp:.2f}"
print("  Sharpness assertions passed.")

model_single = fit(samples, {"pressure": P_REF}, ["pressure"])

# ── Covariance propagation ────────────────────────────────────────────────────

print("\nPropagating covariance for S1 (first 10 obs)...")
import time
s0 = samples[0]
s_sub = SampleData(
    name=s0.name,
    Y=s0.Y[:10],
    dates=s0.dates[:10],
    covariates={"pressure": s0.covariates["pressure"][:10]},
)
var_Y = (s_sub.Y * NOISE_FRAC) ** 2
Y_corr, cov_out = propagate_covariance(s_sub.Y, var_Y, s_sub, model)

diag_out = np.diag(cov_out)
print(f"  All diagonal > 0: {np.all(diag_out > 0)}")

print(f"\nTiming propagate_covariance (full S1, {s0.Y.shape[0]}×{s0.Y.shape[1]} obs)...")
var_full = (s0.Y * NOISE_FRAC) ** 2
t0 = time.perf_counter()
_, cov_full = propagate_covariance(s0.Y, var_full, s0, model)
dt = time.perf_counter() - t0
n = s0.Y.size
print(f"  Analytic GUM: {dt:.3f} s  ({n}→{n}×{n} covariance)")


# ── Assertions ────────────────────────────────────────────────────────────────

assert np.sign(kappa_est) == np.sign(TRUE_KAPPA), \
    f"κ wrong sign: {kappa_est:.2e}"
assert abs(kappa_est) / abs(TRUE_KAPPA) < 10.0, \
    f"κ order of magnitude off: {kappa_est:.2e} vs {TRUE_KAPPA:.2e}"
assert model.kappa_rsq["pressure"] > 0.3, \
    f"κ R² too low: {model.kappa_rsq['pressure']:.4f}"
assert improved / total >= 0.8, \
    f"Too few peaks improved: {improved}/{total}"
# Episode bias step: estimated mean(ep1) - mean(ep0) within 50 % of truth
bias_step_true = BIAS_EP1 - BIAS_EP0
bias_step_est  = (model.episode_bias["S1"][1].mean()
                  - model.episode_bias["S1"][0].mean())
assert abs(bias_step_est - bias_step_true) / max(abs(bias_step_true), 1e-9) < 0.5, \
    f"Episode bias step wrong: est={bias_step_est:.4f} true={bias_step_true:.4f}"
assert np.all(diag_out > 0), "Negative variance in covariance output"

# ── geometric-mean centers (unbiased by design; replaces chi_square_fit) ───────
# model.centers must be the geometric mean exp(mean(log Y_κcorrected)) — the
# unbiased "true value" for log-normal data, which sits at or below the arithmetic
# mean by exp(σ²/2).  This is the by-design replacement for the old chi_square_fit
# center refinement.
print("\ngeometric-mean centers (unbiased true value)...")
kap = model.kappa["pressure"]; ref = model.covariate_refs["pressure"]
for s in samples:
    Yk  = s.Y / (1.0 + kap * (s.covariates["pressure"] - ref))[:, None]   # κ-corrected
    geo = np.exp(np.log(Yk).mean(axis=0))
    assert np.allclose(model.centers[s.name], geo, rtol=1e-6), \
        f"{s.name}: center is not the geometric mean of the κ-corrected areas"
    assert np.all(geo <= Yk.mean(axis=0) * (1.0 + 1e-12)), \
        f"{s.name}: geometric mean exceeds arithmetic mean"
print("  centers = geometric mean of κ-corrected areas (≤ arithmetic) ✓")

print("\nAll assertions passed.")

# ── flag_outliers ─────────────────────────────────────────────────────────────
print("\nflag_outliers: Hampel k=3 on corrected residuals...")

# Inject a handful of gross outliers into one sample
rng   = np.random.default_rng(7)
s_inj = samples[0]
Y_inj = s_inj.Y.copy()
outlier_rows = rng.choice(len(Y_inj), size=5, replace=False)
Y_inj[outlier_rows, 0] *= 1.10    # +10% on peak 0 — well above 3×MAD
s_inj_mod = SampleData(s_inj.name, Y_inj, s_inj.dates,
                       s_inj.covariates)
samples_mod = [s_inj_mod] + samples[1:]
model_mod   = fit(samples_mod, covariate_refs={"pressure": P_REF},
                  breakpoints={"S1": np.array([200]), "S2": np.array([200]),
                               "S3": np.array([200])})

masks = flag_outliers(samples_mod, model_mod, k=3.0)

# All 5 injected outliers must be caught
detected = set(np.where(masks["S1"])[0])
missed   = set(outlier_rows) - detected
extra    = detected - set(outlier_rows)
print(f"  Injected outliers: {sorted(outlier_rows)}")
print(f"  Detected:          {sorted(detected)}")
print(f"  Missed: {len(missed)}   False positives: {len(extra)}")
assert len(missed) == 0, f"flag_outliers missed injected outliers: {missed}"

# Check that outlier_mask flows through propagate_covariance without error
mask_s1  = masks["S1"]
cov_diag = np.full_like(Y_inj, (Y_inj.mean() * 0.002) ** 2)
Y_c_full, cov_full   = propagate_covariance(Y_inj, cov_diag, s_inj_mod, model_mod)
Y_c_clean, cov_clean = propagate_covariance(Y_inj, cov_diag, s_inj_mod, model_mod,
                                             outlier_mask=mask_s1)
n_valid = int((~mask_s1).sum())
assert Y_c_clean.shape == (n_valid, Y_inj.shape[1]), \
    f"Wrong shape after masking: {Y_c_clean.shape}"
assert cov_clean.shape == (n_valid * Y_inj.shape[1],) * 2, \
    f"Wrong cov shape after masking: {cov_clean.shape}"
print(f"  propagate_covariance with mask: {Y_c_full.shape} → {Y_c_clean.shape}  ✓")

# ── fit_uncorrelated_drift ─────────────────────────────────────────────────────
print("\nfit_uncorrelated_drift: PC2 cross-sample model...")
model_drift = fit_uncorrelated_drift(samples, model)

for s in samples:
    assert s.name in model_drift.pc2_peak_load,  f"pc2_peak_load missing for {s.name}"
    assert s.name in model_drift.pc2_poly,        f"pc2_poly missing for {s.name}"
    assert s.name in model_drift.pc2_common_amp,  f"pc2_common_amp missing for {s.name}"
    n_ep = len(known_breakpoints[s.name]) + 1
    assert len(model_drift.pc2_poly[s.name]) == n_ep, (
        f"{s.name}: expected {n_ep} episode poly entries, "
        f"got {len(model_drift.pc2_poly[s.name])}"
    )
    assert model_drift.pc2_peak_load[s.name].shape == (N_PEAKS,), \
        f"{s.name}: peak_load shape {model_drift.pc2_peak_load[s.name].shape}"

assert len(model_drift.pc2_dates) > 0, "pc2_dates empty"

print("  RSD comparison (no_drift vs with_drift):")
for s in samples:
    M_nd = build_multiplier(s, model)
    M_d  = build_multiplier(s, model_drift)
    Y_nd = correct(s.Y, M_nd)
    Y_d  = correct(s.Y, M_d)
    rsd_nd = (Y_nd.std(axis=0) / Y_nd.mean(axis=0)).mean() * 100
    rsd_d  = (Y_d.std(axis=0)  / Y_d.mean(axis=0)).mean()  * 100
    print(f"    {s.name}: no_drift={rsd_nd:.4f}%  with_drift={rsd_d:.4f}%")

print("\nAll assertions passed.")

# ── fit_generalized_correction ────────────────────────────────────────────────
print("\nfit_generalized_correction: ep0-referenced generalised drift model...")

from shrubbery.appac import (fit_generalized_correction, build_multiplier_unknown,
                                  _covariate_correct)
from shrubbery.control_updater import GeneralizedUpdater

masks_clean = flag_outliers(samples, model)
gen_model = fit_generalized_correction(samples, model,
                                        outlier_masks=masks_clean)

# Shape checks
n_days = len(gen_model.dates_global)
assert gen_model.pc1_scores.shape == (n_days,), \
    f"pc1_scores shape {gen_model.pc1_scores.shape}"
assert gen_model.pc2_scores.shape == (n_days,), \
    f"pc2_scores shape {gen_model.pc2_scores.shape}"
for s in samples:
    assert s.name in gen_model.ep0_ref,      f"ep0_ref missing for {s.name}"
    assert s.name in gen_model.pc1_loading,  f"pc1_loading missing for {s.name}"
    assert s.name in gen_model.pc2_loading,  f"pc2_loading missing for {s.name}"
    assert gen_model.ep0_ref[s.name].shape == (N_PEAKS,), \
        f"ep0_ref shape wrong for {s.name}"
    assert np.all(gen_model.ep0_ref[s.name] > 0), \
        f"ep0_ref non-positive for {s.name}"

# ep0_ref should be within 2 % of the true areas (κ-corrected ep0 geometric mean)
for s in samples:
    rel_err = np.abs(gen_model.ep0_ref[s.name] / TRUE_AREAS - 1.0)
    assert np.all(rel_err < 0.02), \
        f"ep0_ref deviates >2 % from TRUE_AREAS for {s.name}: {rel_err}"

# PC scores must be finite for dates covered by all three samples
finite_frac = float(np.mean(np.isfinite(gen_model.pc2_scores)))
assert finite_frac > 0.5, f"Too many NaN in pc2_scores: {finite_frac:.2f} finite"
print(f"  dates_global: {n_days} days,  pc2 finite: {finite_frac*100:.0f}%")
print(f"  pc2_loading: { {k: f'{v:.4f}' for k, v in gen_model.pc2_loading.items()} }")

# ── build_multiplier_unknown ──────────────────────────────────────────────────
print("\nbuild_multiplier_unknown: synthetic unknown sample on S1 instrument...")

# Simulate a different gas mixture measured on the same instrument as S1
TRUE_AREAS_UNK = TRUE_AREAS * 0.6 + 500.0
rng_unk = np.random.default_rng(999)
unk_dates = np.sort(
    rng_unk.choice(gen_model.dates_global, size=50, replace=False)
).astype(int)
unk_pressure = rng_unk.normal(P_REF, P_STD, size=50)
Y_unk = (
    TRUE_AREAS_UNK[None, :]
    * (1.0 + TRUE_KAPPA * (unk_pressure - P_REF))[:, None]
    * rng_unk.normal(1.0, NOISE_FRAC, (50, N_PEAKS))
)
unknown = SampleData(
    name="UNKNOWN",
    Y=Y_unk,
    dates=unk_dates,
    covariates={"pressure": unk_pressure},
)

M_unk = build_multiplier_unknown(unknown, gen_model, model, instrument="S1")

assert M_unk.shape == (50, N_PEAKS), f"multiplier shape {M_unk.shape}"
assert np.all(M_unk > 0),            "multiplier has non-positive values"
assert np.all(np.isfinite(M_unk)),   "multiplier has non-finite values"

# Correction should reduce scatter — pressure is the dominant signal
Y_unk_kap  = _covariate_correct(Y_unk, {"pressure": unk_pressure},
                                  model.kappa, model.covariate_refs)
Y_unk_corr = correct(Y_unk, M_unk)
rsd_raw    = (Y_unk.std(axis=0)      / Y_unk.mean(axis=0)).mean()      * 100
rsd_kap    = (Y_unk_kap.std(axis=0)  / Y_unk_kap.mean(axis=0)).mean()  * 100
rsd_corr   = (Y_unk_corr.std(axis=0) / Y_unk_corr.mean(axis=0)).mean() * 100
print(f"  Unknown: RSD raw={rsd_raw:.4f}%  κ-only={rsd_kap:.4f}%  full={rsd_corr:.4f}%")
assert rsd_corr < rsd_raw, "build_multiplier_unknown did not reduce scatter"
print(f"  Multiplier range: [{M_unk.min():.5f}, {M_unk.max():.5f}]")

# ── GeneralizedUpdater ────────────────────────────────────────────────────────
print("\nGeneralizedUpdater: sequential update on S1 reference data...")

updater = GeneralizedUpdater.from_generalized_model(
    gen_model, model, samples, instrument="S1"
)

s1 = samples[0]
results = updater.update_batch(s1.Y, s1.covariates, s1.dates)

assert len(results) == len(s1.dates), "update_batch result length mismatch"
expected_keys = ("drift_mean", "drift_std", "sr_stat", "breakpoint_detected",
                  "episode", "n_obs", "Y_kappa", "peak_mean_log_ratio")
for key in expected_keys:
    assert key in results[0], f"key '{key}' missing from update result"

# Peak-mean log-ratio in ep0 should be near zero (ep0 is the reference)
ep0_drifts = [r["peak_mean_log_ratio"] for r in results if r["episode"] == 0]
mean_ep0   = float(np.mean(ep0_drifts))
print(f"  Mean peak-mean log-ratio in ep0: {mean_ep0*100:+.4f}%  (expected ≈ 0)")
assert abs(mean_ep0) < 0.01, \
    f"ep0 peak-mean log-ratio too large: {mean_ep0:.4f}"

bps_found = sum(1 for r in results if r["breakpoint_detected"])
print(f"  Breakpoints detected by SR: {bps_found}")
print(updater.summary())

# export_to_generalized: feed future dates and verify extension
future_start = int(gen_model.dates_global.max()) + 1
future_dates = np.arange(future_start, future_start + 10, dtype=int)
for d in future_dates:
    y_fut = TRUE_AREAS * rng_unk.normal(1.0, NOISE_FRAC, N_PEAKS)
    updater.update(y_fut, {"pressure": float(P_REF)}, int(d))

gen_model_ext = updater.export_to_generalized(gen_model)
assert len(gen_model_ext.dates_global) > len(gen_model.dates_global), \
    "export_to_generalized did not extend dates_global"
assert np.all(np.isin(future_dates, gen_model_ext.dates_global)), \
    "future dates missing from extended gen_model"
n_new = len(gen_model_ext.dates_global) - len(gen_model.dates_global)
new_idx = np.searchsorted(gen_model_ext.dates_global, future_dates)
n_finite_new = int(np.sum(np.isfinite(gen_model_ext.pc2_scores[new_idx])))
print(f"  export_to_generalized: +{n_new} dates,  new pc2 finite: {n_finite_new}/{len(future_dates)}")
assert n_finite_new == len(future_dates), \
    f"Not all future pc2_scores are finite: {n_finite_new}/{len(future_dates)}"

print("\nAll generalised correction assertions passed.")

# ── Time series plots ─────────────────────────────────────────────────────────
print("\nGenerating time series plots...")

SAMPLE_COLORS = ["C0", "C1", "C2"]
EP_COLORS     = ["C0", "C2", "C3", "C4", "C5"]

fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)
fig.suptitle("Generalised Correction Model — Time Series Diagnostics", fontsize=13)

# ── Panel 1: per-sample daily-mean log-ratios + PCA reconstruction ────────────
ax = axes[0]
for s, col in zip(samples, SAMPLE_COLORS):
    Y_kap  = _covariate_correct(s.Y, {"pressure": s.covariates["pressure"]},
                                 model.kappa, model.covariate_refs)
    ct_ep0 = gen_model.ep0_ref[s.name]
    r      = np.log(Y_kap / ct_ep0[None, :]).mean(axis=1)
    unique_d = np.unique(s.dates)
    r_daily  = np.array([r[s.dates == d].mean() for d in unique_d])
    ax.scatter(unique_d, r_daily * 100, s=3, alpha=0.35, color=col, zorder=2)

    # PCA reconstruction (PC1 + PC2) for this sample
    idx   = np.searchsorted(gen_model.dates_global, unique_d)
    pc1_t = np.where(np.isnan(gen_model.pc1_scores[idx]), 0.0, gen_model.pc1_scores[idx])
    pc2_t = np.where(np.isnan(gen_model.pc2_scores[idx]), 0.0, gen_model.pc2_scores[idx])
    recon = (pc1_t * gen_model.pc1_loading[s.name]
             + pc2_t * gen_model.pc2_loading[s.name])
    ax.plot(unique_d, recon * 100, color=col, lw=1.2, label=s.name, zorder=3)

ax.axvline(BREAKPOINT_DAY, color="k", lw=1.2, ls="--", alpha=0.5, label="breakpoint day 200")
ax.axhline(0,              color="k", lw=0.5, alpha=0.3)
ax.axhline(BIAS_EP1 * 100, color="gray", lw=1.0, ls=":", alpha=0.6,
           label=f"true ep1 bias ({BIAS_EP1*100:.1f} %)")
ax.set_ylabel("Peak-mean log-ratio (%)")
ax.set_title("κ-corrected, ep0-referenced log-ratios (dots) + PCA reconstruction (lines)")
ax.legend(fontsize=8, ncol=5)
ax.grid(True, alpha=0.2)

# ── Panel 2: PC1 and PC2 temporal scores ──────────────────────────────────────
ax = axes[1]
v1 = np.isfinite(gen_model.pc1_scores)
v2 = np.isfinite(gen_model.pc2_scores)
ax.plot(gen_model.dates_global[v1], gen_model.pc1_scores[v1],
        lw=0.9, label="PC1 — correlated daily factor", alpha=0.85)
ax.plot(gen_model.dates_global[v2], gen_model.pc2_scores[v2],
        lw=0.9, label="PC2 — drift / bias", alpha=0.85)
ax.axvline(BREAKPOINT_DAY, color="k", lw=1.2, ls="--", alpha=0.5)
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.set_ylabel("PCA score")
ax.set_title("Cross-sample PCA temporal scores")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# ── Panel 3: S1 GeneralizedUpdater sequential drift tracking ─────────────────
ax = axes[2]

# Individual injections, coloured by episode
for date, r_dict in zip(s1.dates, results):
    ep  = min(r_dict["episode"], len(EP_COLORS) - 1)
    ax.scatter(date, r_dict["peak_mean_log_ratio"] * 100,
               s=4, color=EP_COLORS[ep], alpha=0.45, zorder=2)

# Posterior drift_mean — daily average of the sequential estimates
drift_by_date: dict[int, list] = {}
for date, r_dict in zip(s1.dates, results):
    drift_by_date.setdefault(int(date), []).append(r_dict["drift_mean"])
plot_dates  = sorted(drift_by_date)
plot_drifts = [float(np.mean(drift_by_date[d])) for d in plot_dates]
ax.plot(plot_dates, np.array(plot_drifts) * 100, color="k", lw=1.5,
        label="posterior drift mean", zorder=4)

# Detected episode boundaries
for rec in updater.history:
    ax.axvline(rec.end_date, color="r", lw=1.5, ls="--", alpha=0.7,
               label=f"SR alarm ep{rec.episode_index}→{rec.episode_index+1}")

# True bias levels
ax.axhline(BIAS_EP0 * 100, color="gray", lw=1.0, ls=":",
           label=f"true bias ep0 ({BIAS_EP0*100:.1f} %)")
ax.axhline(BIAS_EP1 * 100, color="gray", lw=1.0, ls="-.",
           label=f"true bias ep1 ({BIAS_EP1*100:.1f} %)")

ax.set_xlabel("Day")
ax.set_ylabel("Peak-mean log-ratio (%)")
ax.set_title("S1 — GeneralizedUpdater sequential tracking  "
             "(dots = injections coloured by episode, red dashed = SR alarms)")
handles, labels = ax.get_legend_handles_labels()
# deduplicate SR alarm labels
seen, uhandles, ulabels = set(), [], []
for h, l in zip(handles, labels):
    if l not in seen:
        seen.add(l); uhandles.append(h); ulabels.append(l)
ax.legend(uhandles, ulabels, fontsize=8, ncol=2)
ax.grid(True, alpha=0.2)

plt.tight_layout()
out_path = "generalized_timeseries.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"  Saved → {out_path}")
