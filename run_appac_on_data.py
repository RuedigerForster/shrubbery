#!/usr/bin/env python3
"""
run_appac_on_data.py — run the full APPAC pipeline on a dataset and print a report.

A self-contained reference run: load a dataset, fit the correction model, apply
it, and print a human-readable report (κ, detected breakpoints, per-component
correction quality, variance explained, residual statistics, and a GUM
uncertainty sample).  Text only — no plotting, no external services, pure NumPy.

Usage
-----
    python run_appac_on_data.py [DATA] [--exclude NAMES] [--input-cv CV] [--out FILE]

    DATA         dataset path (default: data/PLOT_FID.parquet)
    --exclude    comma-separated sample names to omit from the shared parameters
                 (κ, drift); default "CTL-4" (the degrading cylinder in PLOT_FID).
                 Pass --exclude "" to keep all samples.
    --input-cv   assumed per-injection measurement CV for the GUM demo (default 0.001)
    --out        also write the report to this file
"""
from __future__ import annotations

import argparse
import warnings
import numpy as np

from shrubbery import (
    fit, fit_uncorrelated_drift, fit_generalized_correction,
    detect_breakpoints_global, apply_global_breakpoints,
    flag_outliers, build_multiplier, correct, propagate_covariance,
)
from shrubbery.load_rda import load_samples


def _moments(x: np.ndarray) -> tuple[float, float, float]:
    """Return (skew, excess_kurtosis, std) of a 1-D array."""
    x = x - x.mean()
    v = float((x ** 2).mean())
    if v == 0:
        return 0.0, 0.0, 0.0
    return float((x ** 3).mean()) / v ** 1.5, float((x ** 4).mean()) / v ** 2 - 3.0, np.sqrt(v)


def _robust_keep(x: np.ndarray, k: float = 6.0) -> np.ndarray:
    """Boolean mask dropping points beyond k·MAD of the median."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return np.ones_like(x, bool) if mad == 0 else (np.abs(x - med) <= k * mad)


def run(path: str, exclude: list[str], input_cv: float) -> str:
    lines: list[str] = []
    def out(s: str = "") -> None:
        lines.append(s)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples, _, p_ref = load_samples(path)

        # ── Pipeline ──────────────────────────────────────────────────────────
        # 1. unbiased κ (no breakpoints; κ is ep0-referenced, centers geometric)
        m0 = fit(samples, covariate_refs={"pressure": p_ref}, exclude=exclude or None)
        m0 = fit_uncorrelated_drift(samples, m0)
        kappa = m0.kappa
        # 2. detect instrument-wide breakpoints on the PPCA PC2 score
        masks0 = flag_outliers(samples, m0)
        gen0   = fit_generalized_correction(samples, m0, outlier_masks=masks0,
                                            kappa_fixed=kappa)
        bpr    = detect_breakpoints_global(gen0)
        breakpoints = apply_global_breakpoints(samples, bpr)
        # 3. refit with breakpoints, κ fixed
        model = fit(samples, covariate_refs={"pressure": p_ref},
                    breakpoints=breakpoints, exclude=exclude or None, kappa_fixed=kappa)
        model = fit_uncorrelated_drift(samples, model)
        masks = flag_outliers(samples, model)

    # ── Report ────────────────────────────────────────────────────────────────
    all_peaks = sorted({p for s in samples for p in s.peaks})
    n_inj = sum(len(s.dates) for s in samples)
    d0 = min(int(s.dates.min()) for s in samples)
    d1 = max(int(s.dates.max()) for s in samples)
    pres = np.concatenate([s.covariates["pressure"] for s in samples])

    out("=" * 72)
    out(f"APPAC correction report — {path}")
    out("=" * 72)
    out(f"samples       : {len(samples)}  ({', '.join(s.name for s in samples)})")
    out(f"peaks         : {len(all_peaks)}  ({', '.join(all_peaks)})")
    out(f"injections    : {n_inj}   days {d0}–{d1}")
    out(f"pressure      : {pres.min():.1f}–{pres.max():.1f} hPa   ref = {p_ref:.2f} hPa")
    out(f"excluded      : {exclude or '(none)'}  (omitted from shared κ / drift)")
    out("")

    # κ — common per covariate
    out("Pressure sensitivity κ (one common value per covariate)")
    out("-" * 72)
    for cv in kappa:                                   # diagnostics from m0 (estimated κ)
        se = float(np.sqrt(m0.kappa_var.get(cv, 0.0)))  # the refit model fixes κ → var 0
        out(f"  {cv:10s} κ = {float(kappa[cv]):+.4e} hPa⁻¹   SE = {se:.2e}   "
            f"R² = {m0.kappa_rsq.get(cv, float('nan')):.4f}   (fixed as a constant downstream)")
    out("")

    # breakpoints
    out("Instrument-wide breakpoints (detected on the cross-sample PC2 score)")
    out("-" * 72)
    out(f"  clean breakpoints : {bpr.breakpoints.tolist() or '(none)'}")
    out(f"  dirty windows     : {bpr.dirty_windows or '(none)'}")
    out("")

    # per-component correction quality + final residual, aggregated by peak name
    raw_by, res_by = {p: [] for p in all_peaks}, {p: [] for p in all_peaks}
    rsd_raw = {p: [] for p in all_peaks}
    rsd_cor = {p: [] for p in all_peaks}
    for s in samples:
        M  = build_multiplier(s, model)
        Yc = correct(s.Y, M)
        ct = model.centers[s.name]
        keep = ~masks[s.name]
        r_raw = np.log(s.Y / ct[None, :]); r_raw -= r_raw.mean(0)
        r_cor = np.log(Yc   / ct[None, :]); r_cor -= r_cor.mean(0)
        for j, p in enumerate(s.peaks):
            raw_by[p].append(r_raw[keep, j])
            res_by[p].append(r_cor[keep, j])
            rsd_raw[p].append(s.Y[keep, j].std() / s.Y[keep, j].mean())
            rsd_cor[p].append(Yc[keep, j].std() / Yc[keep, j].mean())

    out("Per-component correction quality")
    out("-" * 72)
    out(f"  {'peak':9s} {'RSD raw':>9s} {'RSD corr':>9s} {'var expl':>9s} "
        f"{'resid sd':>9s} {'skew':>7s} {'exkurt':>7s}")
    tot_raw = tot_res = 0.0
    for p in all_peaks:
        rr = np.concatenate(raw_by[p]); rc = np.concatenate(res_by[p])
        rc = rc[_robust_keep(rc)]
        vexp = 1.0 - rc.var() / rr.var() if rr.var() > 0 else 0.0
        sk, ek, sd = _moments(rc)
        tot_raw += rr.var() * len(rr); tot_res += rc.var() * len(rc)
        out(f"  {p:9s} {np.mean(rsd_raw[p])*100:8.3f}% {np.mean(rsd_cor[p])*100:8.3f}% "
            f"{vexp*100:8.1f}% {sd:9.4f} {sk:+7.2f} {ek:+7.2f}")
    out(f"\n  overall variance explained: {(1 - tot_res/tot_raw)*100:.1f}%")
    out("")

    # GUM uncertainty sample (first non-excluded sample, first 60 valid injections)
    s = next(s for s in samples if s.name not in (exclude or []))
    keep = np.where(~masks[s.name])[0][:60]
    Yk = s.Y[keep]
    cov_Y = (input_cv * Yk) ** 2                          # diagonal measurement variance
    s_sub = type(s)(name=s.name, Y=Yk, dates=s.dates[keep],
                    covariates={c: a[keep] for c, a in s.covariates.items()},
                    peaks=s.peaks)
    Yc, cov = propagate_covariance(Yk, cov_Y, s_sub, model)
    d = np.sqrt(np.clip(np.diag(cov), 0, None)).reshape(Yc.shape)
    rel_uc = np.median(d / Yc, axis=0) * 100
    eig = np.linalg.eigvalsh((cov + cov.T) / 2).min()
    out("GUM combined uncertainty (analytic, GUM Eq. 13)")
    out("-" * 72)
    out(f"  sample {s.name}, first {len(keep)} injections, input CV = {input_cv*100:.2f}%")
    out("  median relative combined standard uncertainty u_c(corrected) per peak:")
    out("    " + "  ".join(f"{p}={u:.3f}%" for p, u in zip(s.peaks, rel_uc)))
    out(f"  covariance is positive semi-definite: {eig > -1e-9} (min eig {eig:+.2e})")
    out("=" * 72)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run APPAC on a dataset and print a report.")
    ap.add_argument("data", nargs="?", default="data/PLOT_FID.parquet")
    ap.add_argument("--exclude", default="CTL-4",
                    help='comma-separated samples to omit from shared parameters (default "CTL-4")')
    ap.add_argument("--input-cv", type=float, default=0.001,
                    help="assumed per-injection measurement CV for the GUM demo (default 0.001)")
    ap.add_argument("--out", default=None, help="also write the report to this file")
    a = ap.parse_args()
    exclude = [x for x in a.exclude.split(",") if x]
    report = run(a.data, exclude, a.input_cv)
    print(report)
    if a.out:
        with open(a.out, "w") as f:
            f.write(report + "\n")
        print(f"\n[report written to {a.out}]")


if __name__ == "__main__":
    main()
