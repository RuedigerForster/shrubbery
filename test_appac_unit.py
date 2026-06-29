"""
Unit tests for appac.py — targeted coverage for:
  1. check_inputs  — error and warning paths
  2. fit(exclude=) — excluded samples don't influence κ; are still corrected
  3. detect_breakpoints — step injection and recovery
  4. _pca_nan(standardize=False) — scale preservation
  5. _fit_polynomial_trend — polynomial degree selection
"""

import warnings as _warnings_mod
import numpy as np
import sys

from shrubbery.appac import (
    SampleData, AppacModel,
    fit, build_multiplier, detect_breakpoints, check_inputs,
    _pca_nan, _fit_polynomial_trend,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

P_REF      = 1000.0
TRUE_KAPPA = -2e-3


def _make_sample(name, n=500, n_peaks=4, seed=0, kappa=TRUE_KAPPA, p_ref=P_REF):
    """Synthetic sample: n consecutive days, white noise, known κ."""
    rng   = np.random.default_rng(seed)
    days  = np.arange(n, dtype=int)
    pres  = rng.normal(p_ref, 8.0, size=n)
    areas = np.array([1e4, 5e3, 2e3, 8e2][:n_peaks], dtype=float)
    Y = (areas[None, :]
         * (1.0 + kappa * (pres - p_ref))[:, None]
         * rng.normal(1.0, 1e-3, (n, n_peaks)))
    return SampleData(name=name, Y=Y, dates=days, covariates={"pressure": pres})


def _assert_raises(exc_type, fn, *args, match=None, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as e:
        if match and match not in str(e):
            raise AssertionError(
                f"Exception raised but {str(e)!r} does not contain {match!r}"
            ) from e
        return
    raise AssertionError(f"Expected {exc_type.__name__} but no exception raised")


_failures = []

def _run(label, fn):
    try:
        fn()
        print(f"  [PASS] {label}")
    except Exception as exc:
        print(f"  [FAIL] {label}: {exc}")
        _failures.append((label, exc))


# Three generic clean samples reused across sections
_s1 = _make_sample("A", seed=0)
_s2 = _make_sample("B", seed=1)
_s3 = _make_sample("C", seed=2)
_refs = {"pressure": P_REF}


# ─────────────────────────────────────────────────────────────────────────────
# 1. check_inputs — error and warning paths
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== 1. check_inputs ===")


def _t_duplicate_names():
    _assert_raises(ValueError, check_inputs,
                   [_s1, _make_sample("A", seed=99)], _refs,
                   match="Duplicate")


def _t_missing_covariate_ref():
    # covariate_names requests an unknown key not present in _refs
    _assert_raises(ValueError, check_inputs, [_s1], _refs,
                   covariate_names=["nonexistent"],
                   match="covariate_names")


def _t_y_not_2d():
    s_bad = SampleData("X", _s1.Y[:, 0], _s1.dates, _s1.covariates)
    _assert_raises(ValueError, check_inputs, [s_bad], _refs, match="2-D")


def _t_nan_in_y():
    Y_nan = _s1.Y.copy(); Y_nan[0, 0] = np.nan
    s_bad = SampleData("X", Y_nan, _s1.dates, _s1.covariates)
    _assert_raises(ValueError, check_inputs, [s_bad], _refs, match="NaN")


def _t_nonpositive_y():
    Y_neg = _s1.Y.copy(); Y_neg[5, 2] = -1.0
    s_bad = SampleData("X", Y_neg, _s1.dates, _s1.covariates)
    _assert_raises(ValueError, check_inputs, [s_bad], _refs, match="non-positive")


def _t_unsorted_dates():
    d = _s1.dates.copy(); d[0], d[1] = d[1], d[0]
    s_bad = SampleData("X", _s1.Y, d, _s1.covariates)
    _assert_raises(ValueError, check_inputs, [s_bad], _refs, match="sorted")


def _t_bp_at_first_date():
    # breakpoint == first date → episode 0 would be empty
    bps = {"A": np.array([_s1.dates[0]], dtype=int)}
    _assert_raises(ValueError, check_inputs, [_s1], _refs,
                   breakpoints=bps, match="episode 0 would be empty")


def _t_bp_beyond_last_date():
    bps = {"A": np.array([_s1.dates[-1] + 1], dtype=int)}
    _assert_raises(ValueError, check_inputs, [_s1], _refs,
                   breakpoints=bps, match="last episode would be empty")


def _t_few_samples_warning():
    w = check_inputs([_s1, _s2], _refs)
    assert any("sample" in ww.lower() for ww in w), \
        f"Expected sample-count warning; got {w}"


def _t_ref_outside_range_warning():
    w = check_inputs([_s1, _s2, _s3], {"pressure": 500.0})
    assert any("outside" in ww.lower() for ww in w), \
        f"Expected range warning; got {w}"


_run("duplicate names → ValueError",                    _t_duplicate_names)
_run("covariate missing from refs → ValueError",        _t_missing_covariate_ref)
_run("Y not 2-D → ValueError",                         _t_y_not_2d)
_run("NaN in Y → ValueError",                          _t_nan_in_y)
_run("non-positive Y → ValueError",                    _t_nonpositive_y)
_run("unsorted dates → ValueError",                    _t_unsorted_dates)
_run("breakpoint == first date → ValueError",           _t_bp_at_first_date)
_run("breakpoint > last date → ValueError",             _t_bp_beyond_last_date)
_run("< 3 samples → warning, not error",               _t_few_samples_warning)
_run("ref outside observed range → warning",           _t_ref_outside_range_warning)


# ─────────────────────────────────────────────────────────────────────────────
# 2. fit(exclude=)
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== 2. fit(exclude=) ===")

# Bad sample has kappa = +5e-3 (opposite sign to TRUE_KAPPA = -2e-3),
# strongly contaminating the pooled κ estimate when not excluded.
_bad = _make_sample("bad", seed=99, kappa=+5e-3)


def _t_exclude_reduces_kappa_error():
    """κ with exclude=[bad] must be closer to TRUE_KAPPA than κ without it."""
    m_all  = fit([_s1, _s2, _s3, _bad], _refs)
    m_excl = fit([_s1, _s2, _s3, _bad], _refs, exclude=["bad"])
    err_all  = abs(m_all.kappa["pressure"]  - TRUE_KAPPA)
    err_excl = abs(m_excl.kappa["pressure"] - TRUE_KAPPA)
    assert err_excl < err_all, (
        f"exclude= did not reduce κ error: all={err_all:.2e} excl={err_excl:.2e}"
    )
    assert np.sign(m_excl.kappa["pressure"]) == np.sign(TRUE_KAPPA), \
        f"κ wrong sign after exclude: {m_excl.kappa['pressure']:.2e}"


def _t_excluded_sample_still_correctable():
    """build_multiplier must succeed for the excluded sample."""
    m = fit([_s1, _s2, _s3, _bad], _refs, exclude=["bad"])
    assert "bad" in m.centers,      "centers missing for excluded sample"
    assert "bad" in m.episode_bias, "episode_bias missing for excluded sample"
    assert "bad" in m.corr_scaling, "corr_scaling missing for excluded sample"
    M = build_multiplier(_bad, m)
    assert M.shape == _bad.Y.shape,  f"Multiplier shape mismatch: {M.shape}"
    assert np.all(np.isfinite(M)),   "NaN/Inf in multiplier for excluded sample"
    assert np.all(M > 0),            "Non-positive multiplier for excluded sample"


def _t_excluded_samples_field_set():
    m = fit([_s1, _s2, _s3, _bad], _refs, exclude=["bad"])
    assert "bad" in m.excluded_samples, \
        f"model.excluded_samples not populated: {m.excluded_samples}"


def _t_exclusion_fires_one_warning():
    """The exclusion warning fires exactly once from the outer fit() call.
    Internal fit() calls inside chi_square_fit suppress it via catch_warnings."""
    with _warnings_mod.catch_warnings(record=True) as caught:
        _warnings_mod.simplefilter("always")
        fit([_s1, _s2, _s3, _bad], _refs, exclude=["bad"])
    excl = [w for w in caught
            if "bad" in str(w.message) and "excluded" in str(w.message).lower()]
    assert len(excl) == 1, f"Expected 1 exclusion warning, got {len(excl)}"


_run("exclude= reduces κ error from unstable sample",  _t_exclude_reduces_kappa_error)
_run("excluded sample has all model fields",            _t_excluded_sample_still_correctable)
_run("excluded_samples field is populated",             _t_excluded_samples_field_set)
_run("exclusion fires exactly one warning",             _t_exclusion_fires_one_warning)


# ─────────────────────────────────────────────────────────────────────────────
# 3. detect_breakpoints — step injection and recovery
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== 3. detect_breakpoints ===")

_STEP_LOAD = np.array([0.025, -0.015, 0.030, -0.020])   # peak-differential


def _minimal_model(Y: np.ndarray, dates: np.ndarray, pres: np.ndarray) -> tuple:
    """Return (SampleData, AppacModel) with zero corrections.

    With kappa={}, episode_bias=0, and daily_corr=0, build_multiplier returns
    M = 1, so log(Y_c/ct) = log(Y/ct) and any injected step/ramp is directly
    visible in the residuals at full amplitude.  This isolates the D_B +
    sharpness algorithm from corr_t × sc_corr noise introduced by the
    cross-sample model, which would reduce SNR when the step-driven variance
    dominates sc_corr.
    """
    n_pk = Y.shape[1]
    ct   = Y.mean(axis=0)
    s = SampleData("test", Y, dates, {"pressure": pres})
    m = AppacModel(
        kappa={},
        kappa_rsq={},
        covariate_refs={"pressure": P_REF},
        centers={"test": ct},
        episode_bias={"test": np.zeros((1, n_pk))},
        breakpoints={"test": np.array([], dtype=int)},
        corr_scaling={"test": np.zeros(n_pk)},
        dates_global=dates,
        daily_corr=np.zeros(len(dates)),
    )
    return s, m


def _t_bp_finds_step():
    """D_B binary segmentation must find the injected step within ±14 days."""
    TRUE_DAY = 250; n = 500
    rng   = np.random.default_rng(7)
    days  = np.arange(n, dtype=int)
    pres  = rng.normal(P_REF, 8.0, n)
    areas = np.array([1e4, 5e3, 2e3, 8e2], dtype=float)
    bias  = np.where(days[:, None] >= TRUE_DAY, _STEP_LOAD[None, :], 0.0)
    Y     = areas[None, :] * (1.0 + bias) * rng.normal(1.0, 1e-3, (n, 4))
    s, m  = _minimal_model(Y, days, pres)
    bps   = detect_breakpoints(s, m)
    assert len(bps) >= 1, f"No breakpoints found; expected one near day {TRUE_DAY}"
    closest = int(min(bps, key=lambda d: abs(int(d) - TRUE_DAY)))
    assert abs(closest - TRUE_DAY) <= 14, (
        f"Detected bp {closest} > 14 days from step at {TRUE_DAY}"
    )


def _t_bp_rejects_ramp():
    """Sharpness test must reject a smooth ramp even when its D_B is large."""
    n     = 500
    rng   = np.random.default_rng(13)
    days  = np.arange(n, dtype=int)
    pres  = rng.normal(P_REF, 8.0, n)
    areas = np.array([1e4, 5e3, 2e3, 8e2], dtype=float)
    ramp  = (days[:, None] / n) * _STEP_LOAD[None, :]   # same total Δ, spread linearly
    Y     = areas[None, :] * (1.0 + ramp) * rng.normal(1.0, 1e-3, (n, 4))
    s, m  = _minimal_model(Y, days, pres)
    bps   = detect_breakpoints(s, m)
    assert len(bps) == 0, f"Ramp produced {len(bps)} false bp(s): {bps}"


def _t_bp_clean_empty():
    """A clean sample with no injected shift must return no breakpoints."""
    m   = fit([_s1, _s2, _s3], _refs)
    bps = detect_breakpoints(_s1, m)
    assert len(bps) == 0, f"Clean sample produced {len(bps)} false bp(s): {bps}"


_run("detect_breakpoints finds injected step within ±14 days", _t_bp_finds_step)
_run("detect_breakpoints rejects smooth ramp (sharpness test)", _t_bp_rejects_ramp)
_run("detect_breakpoints returns empty for clean sample",       _t_bp_clean_empty)


# ─────────────────────────────────────────────────────────────────────────────
# 4. _pca_nan(standardize=False) — scale preservation
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== 4. _pca_nan(standardize=False) ===")


def _t_std_true_inflates():
    """standardize=True normalises O(1e-3) log-ratio data to unit variance."""
    rng = np.random.default_rng(42)
    X   = rng.normal(0, 1e-3, (200, 4))
    sc, _, _ = _pca_nan(X, n_components=1, standardize=True)
    assert sc[~np.isnan(sc[:, 0]), 0].std() > 0.1, \
        "standardize=True scores unexpectedly small"


def _t_std_false_preserves():
    """standardize=False keeps scores in the original O(1e-3) scale."""
    rng = np.random.default_rng(42)
    X   = rng.normal(0, 1e-3, (200, 4))
    sc, _, _ = _pca_nan(X, n_components=1, standardize=False)
    assert sc[~np.isnan(sc[:, 0]), 0].std() < 0.05, \
        f"standardize=False scores inflated: {sc[~np.isnan(sc[:,0]),0].std():.2e}"


def _t_std_false_amplitude_10x_smaller():
    """The historic bug: standardize=True inflated PC2 scores ~1000×, pushing
    the multiplier negative.  standardize=False must reduce amplitude ≥ 10×."""
    rng  = np.random.default_rng(7)
    n, k = 200, 4
    load = np.array([0.5, -0.3, 0.6, -0.4]); load /= np.linalg.norm(load)
    X = (rng.normal(0, 1e-3, n)[:, None] * load[None, :]
         + rng.normal(0, 1e-4, (n, k)))
    sc_t, _, _ = _pca_nan(X, n_components=1, standardize=True)
    sc_f, _, _ = _pca_nan(X, n_components=1, standardize=False)
    amp_t = float(np.nanstd(sc_t[:, 0]))
    amp_f = float(np.nanstd(sc_f[:, 0]))
    assert amp_f < amp_t / 10, (
        f"standardize=False did not reduce amplitude: True={amp_t:.2e} False={amp_f:.2e}"
    )


_run("standardize=True inflates log-ratio scores to O(1)",  _t_std_true_inflates)
_run("standardize=False preserves O(1e-3) scale",           _t_std_false_preserves)
_run("standardize=False amplitude ≥ 10× smaller than True", _t_std_false_amplitude_10x_smaller)


# ─────────────────────────────────────────────────────────────────────────────
# 5. _fit_polynomial_trend — polynomial degree selection
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== 5. _fit_polynomial_trend ===")

_NO_BPS = np.array([], dtype=int)


def _t_poly_flat():
    """Pure noise → degree 0 (practical min_drift_frac bound rejects any slope)."""
    rng = np.random.default_rng(0)
    dates = np.arange(200, dtype=int)
    y = rng.normal(0, 1e-4, 200)
    res = _fit_polynomial_trend(y, dates, _NO_BPS)
    assert res[0]['degree'] == 0, f"Flat signal → degree {res[0]['degree']}, expected 0"


def _t_poly_linear():
    """Strong linear trend → degree 1; slope recovered to 10 %."""
    TRUE_SLOPE = 1e-5   # 1e-5/day × 200 days = 2e-3 > min_drift_frac=1e-3
    rng   = np.random.default_rng(1)
    dates = np.arange(200, dtype=int)
    tc    = float(dates.mean())
    y     = TRUE_SLOPE * (dates - tc) + rng.normal(0, 1e-6, 200)
    res   = _fit_polynomial_trend(y, dates, _NO_BPS, min_pts_linear=30)
    assert res[0]['degree'] >= 1, \
        f"Linear trend → degree {res[0]['degree']}, expected ≥ 1"
    slope = float(res[0]['coeffs'][1])
    assert abs(slope - TRUE_SLOPE) / abs(TRUE_SLOPE) < 0.10, \
        f"Slope off: est={slope:.2e} true={TRUE_SLOPE:.2e}"


def _t_poly_quadratic():
    """Trend with both linear and quadratic components → degree 2."""
    rng   = np.random.default_rng(2)
    dates = np.arange(400, dtype=int)
    tc    = float(dates.mean())
    dt    = dates - tc
    # a1 × 400 = 2e-3 > min_drift_frac; a2 × 200² = 2e-3 > min_drift_frac
    y = 5e-6 * dt + 5e-8 * dt**2 + rng.normal(0, 1e-5, 400)
    res = _fit_polynomial_trend(y, dates, _NO_BPS,
                                min_pts_linear=30, min_pts_quadratic=90)
    assert res[0]['degree'] == 2, \
        f"Quadratic+linear trend → degree {res[0]['degree']}, expected 2"


def _t_poly_too_short():
    """Fewer than min_pts_linear days → degree 0 regardless of trend magnitude."""
    rng   = np.random.default_rng(3)
    dates = np.arange(20, dtype=int)   # 20 < min_pts_linear=30
    tc    = float(dates.mean())
    y     = 1e-5 * (dates - tc) + rng.normal(0, 1e-6, 20)
    res   = _fit_polynomial_trend(y, dates, _NO_BPS, min_pts_linear=30)
    assert res[0]['degree'] == 0, \
        f"Short episode → degree {res[0]['degree']}, expected 0"


def _t_poly_two_episodes():
    """Two episodes with opposite slopes must be fitted independently."""
    rng   = np.random.default_rng(4)
    n, bp = 300, 150
    dates = np.arange(n, dtype=int)
    tc0   = float(dates[:bp].mean())
    tc1   = float(dates[bp:].mean())
    SLOPE = 1e-5
    y     = np.empty(n)
    y[:bp] =  SLOPE * (dates[:bp] - tc0) + rng.normal(0, 1e-6, bp)
    y[bp:] = -SLOPE * (dates[bp:] - tc1) + rng.normal(0, 1e-6, n - bp)
    res = _fit_polynomial_trend(y, dates, np.array([bp], dtype=int),
                                min_pts_linear=30)
    assert len(res) == 2, f"Expected 2 episodes, got {len(res)}"
    assert res[0]['degree'] >= 1, "Episode 0 should detect linear trend"
    assert res[1]['degree'] >= 1, "Episode 1 should detect linear trend"
    assert np.sign(res[0]['coeffs'][1]) != np.sign(res[1]['coeffs'][1]), \
        "Episode slopes should have opposite signs"


_run("flat signal → degree 0",                          _t_poly_flat)
_run("linear trend → degree 1, slope to 10 %",         _t_poly_linear)
_run("linear + quadratic trend → degree 2",             _t_poly_quadratic)
_run("< min_pts_linear days → degree 0",                _t_poly_too_short)
_run("breakpoint → separate poly per episode",          _t_poly_two_episodes)


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'─' * 60}")
if _failures:
    print(f"{len(_failures)} test(s) FAILED:")
    for label, exc in _failures:
        print(f"  • {label}: {exc}")
    sys.exit(1)
else:
    print("All 25 unit tests passed.")
