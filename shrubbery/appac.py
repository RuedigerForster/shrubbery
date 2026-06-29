"""
APPAC — Atmospheric Pressure Peak Area Correction
JAX implementation with full covariance propagation.

Port of the R package in shrubbery/R/ by the same author.
R-specific extras (S4 classes, tidyverse, plotting) are omitted.

Environmental covariates
------------------------
The model supports any number of named environmental covariates, e.g.:
  "pressure"    — atmospheric pressure [mbar]

Pass only one covariate at a time during the exploration phase to rank their
individual explanatory power (kappa_rsq), then commit to the best model.

Pipeline
--------
1. Fit dispersion (sd ~ center + center²)
2. PCA decomposition → correlated (PC1), uncorrelated (PC2), noise (PC3+)
3. Estimate per-covariate sensitivity kappa via OLS on binned log-ratios
4. Covariate-correct raw areas
5. Second PCA pass on corrected data
6. Estimate uncorrelated model (trend + bias) via cross-sample PCA
7. Estimate correlated model (daily factor) via cross-sample PCA
8. Chi-square refinement of center values (essential for unbiased results)
9. Build correction multiplier; apply to raw areas

Multiplier
----------
    multiplier[t, j] = 1
        + episode_bias[episode(t), j]
        + (pc2_common[t] × pc2_amp[s] + poly_score(t, ep)) × pc2_load[j, s]
        + corr[t] × corr_scaling[j]
        + Σ_cv  kappa[cv] × (covariate[cv][t] − ref[cv])

    corrected = raw / multiplier

The PC2 term is only present after fit_uncorrelated_drift() has been called.

Covariance propagation
----------------------
GUM Eq. 13 summed over five independent uncertainty sources:

    Σ_out[i,i'] = Σ_Y[i,i'] / (M_i · M_i')                  [measurement]
                + Σ_cv  var(κ_cv) · s_κ[i] · s_κ[i']         [κ per covariate]
                + Σ_{e,j}  var(b_{e,j}) · s_b[i] · s_b[i']   [episode bias]
                + Σ_t  dcv[t] · s_c[i] · s_c[i']             [daily_corr between-sample]
                + Σ_{e,ci}  var(a_{e,ci}) · s_p[i] · s_p[i'] [PC2 polynomial coefficients]

Sensitivity vectors in the flattened (n_obs × n_peaks) index space:

    s_κ[t,j]        = −Y[t,j] · Δcv[t] / M[t,j]²
    s_b_{e,j}[t,k]  = −Y[t,j] / M[t,j]²  if episode(t)=e and k=j, else 0
    s_c[t,j]        = −Y[t,j] · sc[j]    / M[t,j]²   (all peaks on day t co-vary)
    s_p_{e,ci}[t,j] = −Y[t,j] · load[j] · (t−tc)^ci / M[t,j]²  within episode e

The Jacobian w.r.t. Y is diagonal (J = diag(1/M_flat)); the κ, daily_corr,
and PC2 terms are rank-1 outer products; the bias term updates only the
(n_e × n_e) block per (episode, peak).  No AD required.
"""

from __future__ import annotations

import warnings
from collections import Counter
import numpy as np
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SampleData:
    """Input data for one GC sample series."""
    name: str
    Y: np.ndarray                        # (n_obs, n_peaks)  raw peak areas
    dates: np.ndarray                    # (n_obs,)           integer day numbers
    covariates: dict[str, np.ndarray]    # {name: (n_obs,)}  environmental covariates
    peaks: list = field(default_factory=list)  # peak names, length n_peaks; empty = unknown


@dataclass
class AppacModel:
    """Estimated model parameters — output of fit()."""
    kappa: dict[str, float]                  # per-covariate sensitivity coefficient
    kappa_rsq: dict[str, float]              # per-covariate R² of kappa estimation
    covariate_refs: dict[str, float]         # reference value per covariate
    # Per-sample
    centers: dict[str, np.ndarray]           # (n_peaks,) column means of corrected Y
    episode_bias: dict[str, np.ndarray]      # (n_episodes, n_peaks) per-sample
    breakpoints: dict[str, np.ndarray]       # (n_breakpoints,) sorted dates per sample
    corr_scaling: dict[str, np.ndarray]      # (n_peaks,) correlated-feature scaling
    # Global daily correlated factor (sorted unique dates across all samples)
    dates_global: np.ndarray                 # (n_dates,)
    daily_corr: np.ndarray                   # (n_dates,)
    # Uncertainty fields (populated by fit())
    kappa_var: dict[str, float] = field(default_factory=dict)   # OLS variance of kappa
    kappa_resid: dict[str, np.ndarray] = field(default_factory=dict)  # WLS weighted residuals
    kappa_bins: dict[str, tuple] = field(default_factory=dict)  # (dp_b, w_b, y_b, y_pred) for diagnostics
    episode_bias_var: dict[str, np.ndarray] = field(default_factory=dict)  # (n_ep, n_peaks) per sample
    # PC2-based uncorrelated drift model (populated by fit_uncorrelated_drift)
    pc2_dates: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    pc2_common: np.ndarray = field(default_factory=lambda: np.array([]))
    pc2_common_amp: dict[str, float] = field(default_factory=dict)
    pc2_peak_load: dict[str, np.ndarray] = field(default_factory=dict)
    pc2_poly: dict[str, list] = field(default_factory=dict)
    # Samples excluded from cross-sample parameter estimation (κ, corr, PC2)
    excluded_samples: list = field(default_factory=list)
    # Per-day between-sample variance of daily_corr estimate (n_dates,)
    daily_corr_var: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class GeneralizedCorrectionModel:
    """
    Drift/bias model extracted from κ-corrected reference data.

    Residuals are referenced to the geometric mean of each sample's first
    episode (ep0_ref), so the correction traces every subsequent injection back
    to the instrument's initial baseline state.  PC1 captures the correlated
    daily factor; PC2 captures slow drift and inter-episode bias shifts.

    Apply to any unknown sample measured on a known instrument via
    build_multiplier_unknown().
    """
    dates_global: np.ndarray            # (n_days,)  union of all reference dates
    pc1_scores:   np.ndarray            # (n_days,)  NaN for unobserved days
    pc2_scores:   np.ndarray            # (n_days,)  NaN for unobserved days
    pc1_loading:  dict[str, float]      # per-sample PC1 amplitude
    pc2_loading:  dict[str, float]      # per-sample PC2 amplitude
    ep0_ref:      dict[str, np.ndarray] # (n_peaks,)  κ-corrected ep0 geometric means


@dataclass
class BreakpointResult:
    """
    Output of detect_breakpoints_global().

    Detected events are classified into two physically distinct categories:

    breakpoints : sorted array of dates for clean, isolated instrument-wide
        step changes (cylinder swap, detector service).  These are passed to
        fit() as episode boundaries.

    dirty_windows : list of (start, end) date pairs during which the
        instrument was being cleaned or reconditioned after contamination.
        Data within these windows should be excluded from all fitting.
        Typically produced by a cluster of close step changes — the instrument
        was "messed with" repeatedly over days or weeks while recovering, and
        the measurements during that period are not representative of any
        stable state.
    """
    breakpoints:   np.ndarray           # (n_bp,)  clean global step dates
    dirty_windows: list[tuple[int,int]] # [(start, end), ...]  contamination periods


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — dispersion normalization
# ──────────────────────────────────────────────────────────────────────────────

def _fit_dispersion(ct_all: np.ndarray, sc_all: np.ndarray) -> np.ndarray:
    """
    Fit  sd ~ 0 + center + center²  by OLS across all peaks and samples.

    Parameters
    ----------
    ct_all : (n_samples × n_peaks,)  concatenated column means from all samples
    sc_all : (n_samples × n_peaks,)  corresponding column standard deviations

    Returns
    -------
    ex : (2,)  coefficients [ex1, ex2] (no intercept)
    """
    x = ct_all.ravel()
    X = np.column_stack([x, x**2])
    ex, _, _, _ = np.linalg.lstsq(X, sc_all.ravel(), rcond=None)
    return ex


def _dispersion_scale(ct: np.ndarray, ex: np.ndarray) -> np.ndarray:
    """
    Compute per-peak dispersion scale from fitted dispersion coefficients.

    sc_new_j = ex[0] / (ex[0] + ex[1] × ct_j)

    Normalises heteroscedasticity so peaks with different mean areas
    contribute equally to PCA.

    Parameters
    ----------
    ct : (n_peaks,)  column means
    ex : (2,)        dispersion coefficients from ``_fit_dispersion``

    Returns
    -------
    (n_peaks,)  scale factors; multiply dispersion-normalised Z-scores by
    these before SVD and divide after to return to Z-score space
    """
    return ex[0] / (ex[0] + ex[1] * ct)


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — PCA decomposition (per sample)
# ──────────────────────────────────────────────────────────────────────────────

def _transform_features(
    Y_z: np.ndarray,
    sc_new: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA decompose dispersion-normalised Z-scored data (steps 1b + 2).

    Step 1a (Z-score) and 1b (dispersion normalisation) must be applied by
    the caller; pass Y_z = (Y − ct) / sc and sc_new = _dispersion_scale(ct, ex).

    Parameters
    ----------
    Y_z    : (n_obs, n_peaks)  Z-scored peak areas — (Y − ct) / sc
    sc_new : (n_peaks,)        dispersion scale — ex[0] / (ex[0] + ex[1]*ct)

    Returns (all in Z-score space, dispersion normalisation undone)
    -------
    Y_cor   — PC1 reconstruction: inter-peak correlated signal
    Y_unc   — PC2 reconstruction: uncorrelated residual
    Y_noiz  — PC3+: noise
    """
    Y_dn = Y_z * sc_new   # dispersion-normalised Z-scores

    U, S, Vt = np.linalg.svd(Y_dn, full_matrices=False)

    sc_safe = np.where(sc_new == 0, 1.0, sc_new)
    Y_cor = np.outer(U[:, 0] * S[0], Vt[0]) / sc_safe
    Y_unc = np.outer(U[:, 1] * S[1], Vt[1]) / sc_safe
    Y_noiz = Y_dn / sc_safe - Y_cor - Y_unc

    return Y_cor, Y_unc, Y_noiz


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — kappa estimation (chi-square / OLS on binned means)
# ──────────────────────────────────────────────────────────────────────────────

def _estimate_kappa(
    Y_cor_list: list[np.ndarray],
    covariate_lists: dict[str, list[np.ndarray]],
    ct_list: list[np.ndarray],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, np.ndarray], dict[str, tuple]]:
    """
    Estimate per-covariate sensitivity kappa via WLS on binned log-ratios.

    Forward model (exact for multiplicative / log-normal data):
        log(Y_j / ct_j)  ≈  κ · (covariate − ref)

    Working in log space makes residuals Gaussian when the underlying peak area
    distribution is log-normal (right-skewed in linear scale), which is typical
    for GC data.  OLS slope on binned means (0.1-unit bins) — binning averages
    out residuals without imposing sparsity.

    Parameters
    ----------
    covariate_lists : {name: [array_per_sample]}  values already referenced
                      (covariate − ref), so mean ≈ 0.

    Returns
    -------
    kappa, kappa_rsq, kappa_var, kappa_resid — dicts keyed by covariate name.
    kappa_var   : WLS variance σ²_w / Σ(w_i · dp_c_i²).
    kappa_resid : sqrt(w_i) · (y_b_i − ŷ_b_i) — WLS weighted residuals.
    kappa_bins  : (dp_b, w_b, y_b, y_pred) — binned data for scatter diagnostics.
    """
    kappa: dict[str, float] = {}
    rsq: dict[str, float] = {}
    kappa_var: dict[str, float] = {}
    kappa_resid: dict[str, np.ndarray] = {}
    kappa_bins: dict[str, tuple] = {}

    for cv_name, cv_per_sample in covariate_lists.items():
        # Mean log-ratio per observation, averaged over peaks.
        # log(Y/ct) is the correct dependent variable for the multiplicative
        # model; it symmetrises right-skewed (log-normal) peak area distributions.
        frac_all = np.concatenate([
            np.log(Y / ct).mean(axis=1)
            for Y, ct in zip(Y_cor_list, ct_list)
        ])
        dp_all = np.concatenate(cv_per_sample)

        # Bin by covariate at 0.1-unit resolution for outlier robustness
        bin_keys = np.round(dp_all, 1)
        unique_bins = np.unique(bin_keys)
        w_b  = np.array([float((bin_keys == b).sum()) for b in unique_bins])
        y_b  = np.array([frac_all[bin_keys == b].mean() for b in unique_bins])
        dp_b = unique_bins

        # WLS with weights = bin counts (recovers OLS on raw data; outer bins
        # have fewer obs → lower weight → homoscedastic weighted residuals)
        dp_w   = float(w_b @ dp_b) / w_b.sum()
        y_w    = float(w_b @ y_b)  / w_b.sum()
        dp_c   = dp_b - dp_w
        y_c    = y_b  - y_w
        ss_dp  = float(w_b @ (dp_c ** 2))
        kappa_est = float(w_b @ (dp_c * y_c)) / ss_dp if ss_dp > 0 else 0.0

        y_pred   = y_w + kappa_est * dp_c
        raw_res  = y_b - y_pred
        ss_res_w = float(w_b @ (raw_res ** 2))
        ss_tot_w = float(w_b @ (y_c ** 2))
        n_bins   = len(dp_b)

        sigma2_w = ss_res_w / max(n_bins - 2, 1)

        kappa[cv_name]       = kappa_est
        rsq[cv_name]         = 1.0 - ss_res_w / ss_tot_w if ss_tot_w > 0 else 0.0
        kappa_var[cv_name]   = sigma2_w / ss_dp if ss_dp > 0 else np.inf
        kappa_resid[cv_name] = np.sqrt(w_b) * raw_res
        kappa_bins[cv_name]  = (dp_b, w_b, y_b, y_pred)

    return kappa, rsq, kappa_var, kappa_resid, kappa_bins


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — covariate correction
# ──────────────────────────────────────────────────────────────────────────────

def _covariate_correct(
    Y: np.ndarray,
    covariates: dict[str, np.ndarray],
    kappa: dict[str, float],
    covariate_refs: dict[str, float],
) -> np.ndarray:
    """
    Divide raw areas by the summed covariate correction factor.

    Y_corrected = Y / (1 + Σ_cv  kappa[cv] × (covariate[cv] − ref[cv]))

    Parameters
    ----------
    Y              : (n_obs, n_peaks)  raw peak areas
    covariates     : {name: (n_obs,)}  covariate arrays (unreferenced)
    kappa          : {name: float}     fitted sensitivity coefficients
    covariate_refs : {name: float}     reference values

    Returns
    -------
    (n_obs, n_peaks)  covariate-corrected peak areas
    """
    denom = np.ones(len(Y))
    for cv, cov_array in covariates.items():
        denom = denom + kappa[cv] * (cov_array - covariate_refs[cv])
    return Y / denom[:, None]


# ──────────────────────────────────────────────────────────────────────────────
# Steps 6 & 7 — cross-sample PCA for daily factors
# ──────────────────────────────────────────────────────────────────────────────

def _fix_sign(y_scld: np.ndarray) -> np.ndarray:
    """
    Return a sign vector so all peaks co-vary positively with peak 0.
    cov(peak_0, peak_j) >= 0 → +1; anti-correlated peaks → −1.
    peak_0 always gets +1 since its self-covariance (variance) is non-negative.
    """
    cov_row = np.cov(y_scld, rowvar=False)[0]   # cov(peak_0, peak_j) for each j
    return np.where(cov_row >= 0, 1.0, -1.0)


def _daily_factors(
    Y_list: list[np.ndarray],
    dates_list: list[np.ndarray],
    ct_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    For each sample:
      1. Normalise Y by column means.
      2. Standardise and fix sign.
      3. Average first (sign-fixed) column per date → per-sample daily factor.

    Returns
    -------
    all_dates  : (n_dates,)  union of all dates
    daily_mat  : (n_dates, n_samples)  with NaN for uncovered dates
    sc_list    : per-sample signed scaling vectors (n_peaks,)
    """
    all_dates = np.unique(np.concatenate(dates_list))
    daily_mat = np.full((len(all_dates), len(Y_list)), np.nan)
    sc_list = []

    for i, (Y, dates, ct) in enumerate(zip(Y_list, dates_list, ct_list)):
        y_norm = Y / ct
        sc = y_norm.std(axis=0)
        mn = y_norm.mean(axis=0)
        y_std = (y_norm - mn) / np.where(sc == 0, 1.0, sc)
        sgn = _fix_sign(y_std)
        sc_list.append(sc * sgn)

        factor = y_std[:, 0] * sgn[0]
        for d in np.unique(dates):
            row = int(np.searchsorted(all_dates, d))
            daily_mat[row, i] = factor[dates == d].mean()

    return all_dates, daily_mat, sc_list


def _pca_nan(X: np.ndarray, n_components: int = 2, standardize: bool = True):
    """
    PCA on X (n_rows × n_cols) with NaN imputed to column means.

    Parameters
    ----------
    standardize : if True (default), divide each column by its std before SVD
                  so that columns with different scales contribute equally.
                  Set to False when all columns are already on the same scale
                  (e.g. log-ratio units) and absolute magnitudes must be
                  preserved in the returned scores.

    Returns scores (n_rows, n_components) with NaN for all-NaN rows,
    loadings (n_cols, n_components), and variance explained per component.
    """
    all_nan_rows = np.all(np.isnan(X), axis=1)
    valid = ~all_nan_rows
    X_v = X[valid].copy()
    col_means = np.nanmean(X_v, axis=0)
    nan_mask = np.isnan(X_v)
    nan_cols = np.where(nan_mask)[1]          # column index for each NaN entry
    X_v[nan_mask] = col_means[nan_cols]       # replace with that column's mean

    if standardize:
        col_std = X_v.std(axis=0)
        col_std[col_std == 0] = 1.0
        X_sc = (X_v - col_means) / col_std
    else:
        X_sc = X_v - col_means

    U, S, Vt = np.linalg.svd(X_sc, full_matrices=False)
    var_exp = S**2 / np.sum(S**2)

    scores = np.full((len(X), n_components), np.nan)
    scores[valid] = U[:, :n_components] * S[:n_components]
    loadings = Vt[:n_components].T   # (n_cols, n_components)

    return scores, loadings, var_exp[:n_components]


def _ppca_nan(
    X: np.ndarray,
    n_components: int = 2,
    standardize: bool = False,
    max_iter: int = 200,
    tol: float = 1e-6,
):
    """
    PCA with PPCA-style soft-impute for missing values.

    Unlike _pca_nan which fills NaN with the column mean (ignoring all
    cross-column correlation), this iterates:
        fill → low-rank SVD → reconstruct → refill
    until convergence.  The cross-sample correlation structure is used to
    impute rather than the marginal column mean, which preserves step changes
    that occur when only a subset of samples is observed.

    Returns scores (n_rows, n_components) with NaN for all-NaN rows,
    loadings (n_cols, n_components), and variance explained per component.
    """
    all_nan_rows = np.all(np.isnan(X), axis=1)
    valid        = ~all_nan_rows
    X_v          = X[valid].copy()
    nan_mask     = np.isnan(X_v)

    col_means = np.nanmean(X_v, axis=0)
    if standardize:
        col_std = np.nanstd(X_v, axis=0)
        col_std[col_std == 0] = 1.0
    else:
        col_std = np.ones(X_v.shape[1])

    # Initialise: fill NaN with column means
    X_imp = X_v.copy()
    if nan_mask.any():
        X_imp[nan_mask] = col_means[np.where(nan_mask)[1]]

    for _ in range(max_iter):
        X_c      = (X_imp - col_means) / col_std
        U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        recon    = (U[:, :n_components] * S[:n_components]) @ Vt[:n_components]
        new_fill = (recon * col_std + col_means)[nan_mask]
        if not nan_mask.any():
            break
        delta        = float(np.max(np.abs(new_fill - X_imp[nan_mask])))
        X_imp[nan_mask] = new_fill
        if delta < tol:
            break

    X_c      = (X_imp - col_means) / col_std
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    var_exp  = S**2 / np.sum(S**2)

    scores          = np.full((len(X), n_components), np.nan)
    scores[valid]   = U[:, :n_components] * S[:n_components]
    loadings        = Vt[:n_components].T

    return scores, loadings, var_exp[:n_components]


def _estimate_episode_bias(
    Y: np.ndarray,
    dates: np.ndarray,
    ct: np.ndarray,
    breakpoints: np.ndarray,
) -> np.ndarray:
    """
    Estimate a constant bias per episode from covariate-corrected data.

    Averaging over many observations within an episode cancels correlated
    variation and noise, leaving only the systematic episode offset.  A uniform
    shift across all peaks lives in PC1 (not PC2), so the full corrected data
    must be used rather than the PC2 reconstruction.

    Parameters
    ----------
    Y           : (n_obs, n_peaks)  covariate-corrected areas in original scale
    dates       : (n_obs,)          integer day numbers
    ct          : (n_peaks,)        column means (used for normalisation)
    breakpoints : (n_bp,)           sorted dates at which a new episode starts

    Returns
    -------
    bias     : (n_episodes, n_peaks)  fractional deviation per episode
               bias[e, j] = geometric_mean(Y[episode==e, j]) / ct[j] − 1
               Episode 0 is anchored to zero (reference episode).
    bias_var : (n_episodes, n_peaks)  GUM variance of bias estimate
               var(b[e,j]) ≈ sample_var(log(Y[e,:,j]/ct[j])) / n_e
               Episode 0 is the reference (var = 0); episodes ≥ 1 include the
               reference's estimation uncertainty (var[e] += var[0]).
               Set to np.inf when an episode has ≤ 1 observation.
    """
    episode_ids = np.digitize(dates, breakpoints)   # 0, 1, …, n_bp
    n_episodes = len(breakpoints) + 1
    bias     = np.zeros((n_episodes, len(ct)))
    bias_var = np.zeros((n_episodes, len(ct)))

    for e in range(n_episodes):
        mask = episode_ids == e
        n_e = int(mask.sum())
        if n_e == 0:
            continue
        log_ratios = np.log(Y[mask] / ct)               # (n_e, n_peaks)
        bias[e] = np.exp(log_ratios.mean(axis=0)) - 1.0
        if n_e > 1:
            bias_var[e] = log_ratios.var(axis=0, ddof=1) / n_e
        else:
            bias_var[e] = np.inf   # single observation: variance undefined

    bias -= bias[0]   # anchor episode 0 as reference (zero bias)

    # Anchoring propagates ep-0 uncertainty into all other episodes
    bias_var[1:] += bias_var[0]
    bias_var[0]   = 0.0   # ep 0 is definitionally the reference

    return bias, bias_var



def _estimate_correlated_model(
    Y_cor_list: list[np.ndarray],
    dates_list: list[np.ndarray],
    ct_list: list[np.ndarray],
    stable_mask: list[bool] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Cross-sample PCA on correlated (PC1) features.

    stable_mask : bool per sample — False samples are excluded from the PCA
                  but their sc_list entries are still computed so
                  build_multiplier can apply the shared daily factor.

    Returns all_dates, daily_corr (PC1 scores), sc_list, daily_corr_var.
    daily_corr_var : (n_dates,) between-sample variance of the daily_corr
        estimate — Var_i(daily_mat[t,i]) / n_t, the squared standard error of
        the mean across stable instruments on each day.  Zero when only one
        stable instrument is operating (cannot be estimated).  Days where no
        stable instrument operated are NaN.
    """
    all_dates, daily_mat, sc_list = _daily_factors(Y_cor_list, dates_list, ct_list)

    coverage = np.mean(~np.isnan(daily_mat), axis=0)
    stable = np.array(stable_mask if stable_mask is not None
                      else [True] * daily_mat.shape[1])
    pca_cols = np.where(stable & (coverage > 0.7))[0]
    if len(pca_cols) < 2:
        raise ValueError("Insufficient stable concurrent experiments for correlated model.")

    scores, loadings, _ = _pca_nan(daily_mat[:, pca_cols])
    valid = ~np.all(np.isnan(daily_mat[:, pca_cols]), axis=1)

    corr_vals = np.full(len(all_dates), np.nan)
    sv = scores[valid]
    corr_vals[valid] = (sv[:, 0:1] @ loadings[:, 0:1].T)[:, 0]

    # Between-sample variance of daily_corr — SEM² across stable instruments
    dcv = np.full(len(all_dates), np.nan)
    for di, obs_row in enumerate(daily_mat[:, pca_cols]):
        obs = obs_row[~np.isnan(obs_row)]
        n_t = len(obs)
        if n_t >= 2:
            dcv[di] = float(np.var(obs, ddof=1)) / n_t
        elif n_t == 1:
            dcv[di] = 0.0    # single instrument: between-sample variance unknown
    # Fill single-instrument days with global average of multi-instrument days
    global_mean = float(np.nanmean(dcv))
    dcv = np.where(dcv == 0.0, global_mean, dcv)
    dcv = np.where(np.isnan(dcv), 0.0, dcv)

    return all_dates, corr_vals, sc_list, dcv


# ──────────────────────────────────────────────────────────────────────────────
# Steps 8 & 9 — PC2 cross-sample model (uncorrelated drift)
# ──────────────────────────────────────────────────────────────────────────────

def _build_pc2_daily_matrix(
    samples: list[SampleData],
    model: AppacModel,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Build the cross-sample daily PC2-score matrix from corrected log-residuals.

    For each sample:
      1. Apply the full multiplier (episode_bias + daily_corr + kappa).
      2. Compute daily-averaged log-residuals log(Y_c / ct).
      3. Run PCA on those daily residuals (n_days_s × n_peaks) and extract
         PC2 scores (temporal, one scalar per day) and PC2 loadings (spatial,
         one vector per sample).

    Assemble the per-sample PC2 score series into a global
    (n_days_global × n_samples) matrix, NaN where a sample has no data on
    a given day.  This matrix is the input to cross-sample PCA(2).

    Returns
    -------
    all_dates  : (n_days_global,)           union of all unique sample dates
    daily_mat  : (n_days_global, n_samples) PC2 temporal scores, NaN-padded
    peak_loads : {sample_name: (n_peaks,)}  PCA(1).PC2 loading per sample
    """
    all_dates = np.unique(np.concatenate([s.dates for s in samples]))
    n_days = len(all_dates)
    daily_mat = np.full((n_days, len(samples)), np.nan)
    peak_loads: dict[str, np.ndarray] = {}

    for i, s in enumerate(samples):
        M   = build_multiplier(s, model)
        Y_c = correct(s.Y, M)
        ct  = model.centers[s.name]
        r   = np.log(Y_c / ct[None, :])                          # (n_obs, n_peaks)

        unique_d = np.unique(s.dates)
        r_daily  = np.array([r[s.dates == d].mean(axis=0) for d in unique_d])  # (n_d, n_peaks)

        n_d = r_daily.shape[0]
        n_pk = r_daily.shape[1]
        if n_d < 2 or not np.all(np.isfinite(r_daily)):
            peak_loads[s.name] = np.ones(n_pk) / np.sqrt(n_pk)
            continue

        r_c = r_daily - r_daily.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(r_c, full_matrices=False)
        except np.linalg.LinAlgError:
            peak_loads[s.name] = np.ones(n_pk) / np.sqrt(n_pk)
            continue
        scores   = U[:, 1] * S[1]    # (n_d,) PC2 temporal scores
        loadings = Vt[1]              # (n_peaks,) PC2 spatial loadings

        # Sign convention: peak with largest |loading| is positive
        pk = int(np.argmax(np.abs(loadings)))
        if loadings[pk] < 0:
            scores   = -scores
            loadings = -loadings

        peak_loads[s.name] = loadings
        rows = np.searchsorted(all_dates, unique_d)
        daily_mat[rows, i] = scores

    return all_dates, daily_mat, peak_loads


def _fit_polynomial_trend(
    daily_scores: np.ndarray,
    daily_dates: np.ndarray,
    breakpoints: np.ndarray,
    min_pts_linear: int = 30,
    min_pts_quadratic: int = 90,
    alpha: float = 0.05,
    min_drift_frac: float = 1e-3,
) -> list[dict]:
    """
    Fit polynomial trend (degree ≤ 2) within each episode via sequential F-tests.

    Parameters
    ----------
    daily_scores : (n_days,)  per-day PC2 residual scores (one per unique date)
    daily_dates  : (n_days,)  corresponding integer day numbers
    breakpoints  : sorted array of episode-start dates
    min_pts_linear    : min unique days in episode to consider a linear term
    min_pts_quadratic : min unique days to consider quadratic
    alpha             : F-test significance level
    min_drift_frac    : minimum total score change over episode span required
                        alongside p < alpha for linear/quadratic to be accepted
                        (practical lower bound in score units ≈ log-ratio units)

    Returns
    -------
    list of dicts, one per episode (episode id from np.digitize):
        degree     : int — selected polynomial degree (0, 1, or 2)
        coeffs     : (degree+1,) OLS coefficients [a0, a1?, a2?]
        t_center   : float — time origin (mean of episode dates)
        coeffs_var : (degree+1,) OLS variance per coefficient
    """
    from scipy.stats import f as _f

    ep_ids = np.digitize(daily_dates, breakpoints)
    n_episodes = len(breakpoints) + 1
    results = []

    def _ols(X: np.ndarray, y: np.ndarray):
        c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ c
        dof = max(len(y) - X.shape[1], 1)
        sigma2 = float(res @ res) / dof
        return c, float(res @ res), sigma2 * np.diag(np.linalg.pinv(X.T @ X))

    for ep in range(n_episodes):
        mask = ep_ids == ep
        t = daily_dates[mask].astype(float)
        y = daily_scores[mask]
        n = len(t)

        t_ctr = float(t.mean()) if n > 0 else 0.0
        tc    = t - t_ctr

        if n < 2:
            a0 = float(y[0]) if n == 1 else 0.0
            results.append({'degree': 0, 'coeffs': np.array([a0]),
                             't_center': t_ctr, 'coeffs_var': np.array([np.inf])})
            continue

        X0          = np.ones((n, 1))
        c0, rss0, v0 = _ols(X0, y)
        degree, coeffs, cvar = 0, c0, v0

        max_deg = (2 if n >= min_pts_quadratic else
                   1 if n >= min_pts_linear else 0)

        if max_deg >= 1:
            X1          = np.column_stack([np.ones(n), tc])
            c1, rss1, v1 = _ols(X1, y)
            f1 = (rss0 - rss1) / max(rss1 / max(n - 2, 1), 1e-30)
            p1 = 1.0 - _f.cdf(f1, 1, max(n - 2, 1))
            span = float(t[-1] - t[0])
            if p1 < alpha and abs(float(c1[1])) * span > min_drift_frac:
                degree, coeffs, cvar = 1, c1, v1

                if max_deg >= 2:
                    X2          = np.column_stack([np.ones(n), tc, tc ** 2])
                    c2, rss2, v2 = _ols(X2, y)
                    f2 = (rss1 - rss2) / max(rss2 / max(n - 3, 1), 1e-30)
                    p2 = 1.0 - _f.cdf(f2, 1, max(n - 3, 1))
                    # practical: quadratic adds |a2|*(span/2)^2 at episode mid
                    if p2 < alpha and abs(float(c2[2])) * (span / 2) ** 2 > min_drift_frac:
                        degree, coeffs, cvar = 2, c2, v2

        results.append({'degree': degree, 'coeffs': coeffs,
                         't_center': t_ctr, 'coeffs_var': cvar})

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Public API — input validation
# ──────────────────────────────────────────────────────────────────────────────

def check_inputs(
    samples: list[SampleData],
    covariate_refs: dict[str, float],
    covariate_names: list[str] | None = None,
    breakpoints: dict[str, np.ndarray] | None = None,
    min_obs_per_episode: int = 30,
) -> list[str]:
    """
    Validate inputs before fitting.  Called automatically by fit().

    Raises ValueError listing every critical problem found in one pass.
    Returns a list of warning strings for non-critical issues; an empty list
    means no warnings.

    Parameters
    ----------
    samples, covariate_refs, covariate_names, breakpoints
        Same semantics as fit().
    min_obs_per_episode
        Minimum observations per episode to avoid degenerate bias estimates.

    Returns
    -------
    warnings : list[str]
        Non-empty when data are valid but conditions for reliable estimation
        are not fully met.  Raise-worthy problems are never returned — they
        raise instead.
    """
    if covariate_names is None:
        covariate_names = list(covariate_refs.keys())
    if breakpoints is None:
        breakpoints = {s.name: np.array([], dtype=int) for s in samples}

    errors: list[str] = []
    warnings: list[str] = []

    # ── Top-level checks ──────────────────────────────────────────────────────

    if len(samples) == 0:
        raise ValueError("check_inputs: samples list is empty.")

    names = [s.name for s in samples]
    for n, count in Counter(names).items():
        if count > 1:
            errors.append(f"Duplicate sample name: '{n}' (appears {count} times)")

    if len(samples) < 3:
        warnings.append(
            f"Only {len(samples)} sample(s) provided. Cross-sample PCA requires "
            "≥ 3 samples with ≥ 70 % date overlap."
        )

    for cv in covariate_names:
        if cv not in covariate_refs:
            errors.append(
                f"Covariate '{cv}' is in covariate_names but has no entry in "
                "covariate_refs."
            )

    n_peaks_list = [s.Y.shape[1] if s.Y.ndim == 2 else None for s in samples]
    if len(set(n_peaks_list)) > 1:
        for s, n in zip(samples, n_peaks_list):
            all_peaks  = set(p for s2 in samples if s2.peaks for p in s2.peaks)
            own_peaks  = set(s.peaks) if s.peaks else set()
            missing    = sorted(all_peaks - own_peaks)
            if missing:
                warnings.append(
                    f"[{s.name}] has {n} peak(s); missing from global set: {missing}. "
                    f"This sample will be fitted with its own peak subset."
                )

    # ── Per-sample checks ─────────────────────────────────────────────────────

    for s in samples:
        p = f"[{s.name}]"

        # Array dimensions
        if s.Y.ndim != 2:
            errors.append(f"{p} Y must be 2-D, got shape {s.Y.shape}")
            continue                        # remaining shape checks would crash
        n_obs, n_peaks = s.Y.shape

        if s.dates.ndim != 1:
            errors.append(f"{p} dates must be 1-D, got shape {s.dates.shape}")
        elif len(s.dates) != n_obs:
            errors.append(
                f"{p} dates length {len(s.dates)} != Y rows {n_obs}"
            )

        for cv in covariate_names:
            if cv not in s.covariates:
                errors.append(f"{p} covariate '{cv}' is missing")
            elif s.covariates[cv].shape != (n_obs,):
                errors.append(
                    f"{p} covariates['{cv}'] shape {s.covariates[cv].shape} "
                    f"!= ({n_obs},)"
                )

        # Finite and positive Y
        n_nonfinite = int(np.sum(~np.isfinite(s.Y)))
        if n_nonfinite:
            errors.append(f"{p} Y contains {n_nonfinite} NaN/Inf value(s)")
        n_nonpos = int(np.sum(s.Y <= 0))
        if n_nonpos:
            errors.append(
                f"{p} Y contains {n_nonpos} non-positive value(s) "
                "(log(Y) is undefined)"
            )

        # Finite covariates
        for cv in covariate_names:
            if cv in s.covariates:
                arr = s.covariates[cv]
                n_bad = int(np.sum(~np.isfinite(arr)))
                if n_bad:
                    errors.append(
                        f"{p} covariates['{cv}'] contains {n_bad} NaN/Inf value(s)"
                    )

        # Sorted dates
        if len(s.dates) > 1 and not np.all(np.diff(s.dates) >= 0):
            errors.append(f"{p} dates are not sorted in ascending order")

        # Breakpoints
        bps = breakpoints.get(s.name, np.array([], dtype=int))
        if len(bps):
            if not np.all(np.diff(bps) > 0):
                errors.append(f"{p} breakpoints are not strictly increasing")
            if len(s.dates) and bps[0] <= s.dates[0]:
                errors.append(
                    f"{p} breakpoint {bps[0]} ≤ first date {s.dates[0]}: "
                    "episode 0 would be empty"
                )
            if len(s.dates) and bps[-1] > s.dates[-1]:
                errors.append(
                    f"{p} breakpoint {bps[-1]} > last date {s.dates[-1]}: "
                    "last episode would be empty"
                )

        # Episode observation counts (skip if a breakpoint-bound error already
        # explains why an episode is empty — avoids duplicate messages)
        bp_errors_present = any(f"{p} breakpoint" in e for e in errors)
        ep_ids = np.digitize(s.dates, bps)
        n_episodes = len(bps) + 1
        for e in range(n_episodes):
            n_e = int((ep_ids == e).sum())
            if n_e == 0:
                if not bp_errors_present:
                    errors.append(f"{p} episode {e} has zero observations")
            elif n_e < min_obs_per_episode:
                warnings.append(
                    f"{p} episode {e} has only {n_e} observation(s) "
                    f"(recommended ≥ {min_obs_per_episode})"
                )

        # Covariate variation and reference range
        for cv in covariate_names:
            if cv not in s.covariates:
                continue
            arr = s.covariates[cv]
            if not np.any(~np.isfinite(arr)):      # only if array is clean
                std = arr.std()
                if std == 0.0:
                    warnings.append(
                        f"{p} covariates['{cv}'] has zero variance — "
                        "κ cannot be estimated"
                    )
                ref = covariate_refs.get(cv)
                if ref is not None:
                    lo, hi = arr.min(), arr.max()
                    if not (lo <= ref <= hi):
                        warnings.append(
                            f"{p} reference {cv}={ref} is outside the observed "
                            f"range [{lo:.4g}, {hi:.4g}] — extrapolation"
                        )

        # Large date gaps (possible undetected breakpoint)
        if len(s.dates) > 1:
            unique_dates = np.unique(s.dates)
            if len(unique_dates) > 1:
                gaps = np.diff(unique_dates)
                max_gap = int(gaps.max())
                median_gap = float(np.median(gaps))
                if median_gap > 0 and max_gap > 30 * median_gap:
                    warnings.append(
                        f"{p} largest date gap is {max_gap} day(s) "
                        f"({max_gap / median_gap:.0f}× the median gap of "
                        f"{median_gap:.1f} day(s)) — possible undetected breakpoint"
                    )

        # Zero-variance peaks
        zero_var = np.where(s.Y.var(axis=0) == 0)[0]
        if len(zero_var):
            warnings.append(
                f"{p} peak(s) {zero_var.tolist()} have zero variance and will "
                "not contribute to PCA"
            )

    # ── Raise all errors together ─────────────────────────────────────────────

    if errors:
        bullet = "\n  • "
        raise ValueError(
            f"check_inputs found {len(errors)} error(s):{bullet}"
            + bullet.join(errors)
        )

    return warnings


# ──────────────────────────────────────────────────────────────────────────────
# Public API — fit
# ──────────────────────────────────────────────────────────────────────────────

def fit(
    samples: list[SampleData],
    covariate_refs: dict[str, float],
    covariate_names: list[str] | None = None,
    breakpoints: dict[str, np.ndarray] | None = None,
    ct_init: dict[str, np.ndarray] | None = None,
    exclude: list[str] | None = None,
    kappa_fixed: dict[str, float] | None = None,
) -> AppacModel:
    """
    Estimate all APPAC correction parameters.

    Parameters
    ----------
    samples         : list[SampleData]
        One series per GC sample (control cylinder).
    covariate_refs  : dict[str, float]
        Reference value per covariate, e.g. ``{"pressure": 1013.25}``.
        The correction term is zero when the covariate equals its reference.
    covariate_names : list[str], optional
        Subset of covariates to model; None uses all keys in
        ``covariate_refs``.  Pass a single-element list to study one
        covariate at a time during the exploration phase.
    breakpoints     : dict[str, ndarray], optional
        ``{sample_name: sorted int array of dates}`` at which a new episode
        begins (gas cylinder change, detector cleaning, recalibration).
        None treats every sample as a single episode.  Use
        ``detect_breakpoints()`` for automatic detection.
    ct_init         : dict[str, ndarray], optional
        Pre-specified center values per sample; used internally by
        ``chi_square_fit()`` to fix centers during the grid sweep.
    exclude         : list[str], optional
        Sample names to exclude from cross-sample parameter estimation
        (κ, correlated model, PC2 model).  Excluded samples are still
        corrected using the stable-sample parameters.  Use for instruments
        with known long-term instability (column deactivation, contamination).
    kappa_fixed     : dict[str, float], optional
        Pre-computed κ values (e.g. from a prior no-breakpoint run through
        ``chi_square_fit``).  When provided, ``_estimate_kappa`` is skipped
        entirely and the supplied values are used for covariate correction.
        ``kappa_var`` and ``kappa_resid`` are set to zero / empty so that
        downstream covariance propagation treats κ as a known constant.
        This is the correct choice after the first full pipeline pass has
        produced an unbiased κ estimate.

    Returns
    -------
    AppacModel
        Fitted model parameters.  The standard pipeline continues with
        ``fit_uncorrelated_drift()``, then ``chi_square_fit()`` (essential
        for unbiased centers), then ``build_multiplier()`` and
        ``propagate_covariance()``.  Use ``detect_breakpoints()`` to locate
        episode boundaries before the final fit.
    """
    if covariate_names is None:
        covariate_names = list(covariate_refs.keys())
    if breakpoints is None:
        breakpoints = {s.name: np.array([], dtype=int) for s in samples}
    exclude_set = set(exclude or [])
    stable_mask = [s.name not in exclude_set for s in samples]

    for name in exclude_set:
        warnings.warn(
            f"[{name}] excluded from cross-sample parameter estimation "
            f"(κ, correlated model, PC2 model). "
            f"This instrument is treated as unstable and will not contribute to "
            f"shared correction parameters. It is still corrected using the "
            f"model estimated from the remaining stable instruments.",
            stacklevel=2,
        )

    for w in check_inputs(samples, covariate_refs, covariate_names, breakpoints):
        warnings.warn(w, stacklevel=2)

    Y_list = [s.Y for s in samples]
    dates_list = [s.dates for s in samples]
    covariate_lists = {
        cv: [s.covariates[cv] for s in samples]
        for cv in covariate_names
    }

    # Initial centers and scales
    ct0_list = [ct_init[s.name] if ct_init else Y.mean(axis=0)
                for s, Y in zip(samples, Y_list)]
    sc0_list = [Y.std(axis=0) for Y in Y_list]

    # Dispersion normalization coefficients (shared across all samples)
    ex = _fit_dispersion(np.concatenate(ct0_list), np.concatenate(sc0_list))

    # ── Step 1 — Z-score normalisation + dispersion scale ─────────────────────
    # Replace zero std with 1 to avoid division by zero in Z-scoring
    sc0_safe = [np.where(sc == 0, 1.0, sc) for sc in sc0_list]

    Y_z0_list = []
    sc_new0_list = []
    for Y, ct, sc in zip(Y_list, ct0_list, sc0_safe):
        Y_z0_list.append((Y - ct) / sc)
        sc_new0_list.append(_dispersion_scale(ct, ex))

    # ── First PCA pass ────────────────────────────────────────────────────────
    Y_cor1_z = []   # PC1 in Z-score space
    Y_cor1   = []   # PC1 in original scale (for kappa estimation)
    for Y_z, sc_new, sc, ct in zip(Y_z0_list, sc_new0_list, sc0_safe, ct0_list):
        Y_cor_z, _, _ = _transform_features(Y_z, sc_new)
        Y_cor1_z.append(Y_cor_z)
        Y_cor1.append(Y_cor_z * sc + ct)   # back to original scale

    # ── Kappa estimation ──────────────────────────────────────────────────────
    # Shift each covariate array to be zero-centred at its reference value
    covariate_lists_ref: dict[str, list[np.ndarray]] = {}
    for cv, cv_list in covariate_lists.items():
        covariate_lists_ref[cv] = [arr - covariate_refs[cv] for arr in cv_list]

    if kappa_fixed is not None:
        kappa       = kappa_fixed
        kappa_rsq   = {cv: 1.0 for cv in kappa_fixed}
        kappa_var   = {cv: 0.0 for cv in kappa_fixed}
        kappa_resid = {cv: np.array([]) for cv in kappa_fixed}
        kappa_bins  = {cv: (np.array([]), np.array([]), np.array([]), np.array([]))
                       for cv in kappa_fixed}
    else:
        kappa, kappa_rsq, kappa_var, kappa_resid, kappa_bins = _estimate_kappa(
            [Y for Y, ok in zip(Y_cor1, stable_mask) if ok],
            {cv: [v for v, ok in zip(vals, stable_mask) if ok]
             for cv, vals in covariate_lists_ref.items()},
            [ct for ct, ok in zip(ct0_list, stable_mask) if ok],
        )

    # ── Covariate correction ──────────────────────────────────────────────────
    Y_corrected = []
    for Y, s in zip(Y_list, samples):
        cov_subset = {cv: s.covariates[cv] for cv in covariate_names}
        Y_corrected.append(_covariate_correct(Y, cov_subset, kappa, covariate_refs))

    # ── Step 1 on corrected data ──────────────────────────────────────────────
    # When ct_init is provided (chi_square_fit sweep), pin the centers so the
    # full second pass (bias, correlated model, model.centers) all use the same
    # trial center — matching the R behaviour where ct is fixed throughout.
    if ct_init is not None:
        ct2_list = [ct_init[s.name] for s in samples]
    else:
        ct2_list = [Y.mean(axis=0) for Y in Y_corrected]
    sc2_list = [Y.std(axis=0) for Y in Y_corrected]
    ex2 = _fit_dispersion(np.concatenate(ct2_list), np.concatenate(sc2_list))
    sc2_safe = [np.where(sc == 0, 1.0, sc) for sc in sc2_list]

    Y_z2_list = []
    sc_new2_list = []
    for Y, ct, sc in zip(Y_corrected, ct2_list, sc2_safe):
        Y_z2_list.append((Y - ct) / sc)
        sc_new2_list.append(_dispersion_scale(ct, ex2))

    # ── Second PCA pass on covariate-corrected data ───────────────────────────
    Y_cor2 = []   # PC1 in original scale (for correlated model)
    for Y_z, sc_new, sc, ct in zip(Y_z2_list, sc_new2_list, sc2_safe, ct2_list):
        Y_cor_z, _, _ = _transform_features(Y_z, sc_new)
        Y_cor2.append(Y_cor_z * sc + ct)

    # ── Per-sample episode bias (mean per episode from full corrected residuals)
    ep_bias: dict[str, np.ndarray] = {}
    ep_bias_var: dict[str, np.ndarray] = {}
    for i, s in enumerate(samples):
        b, bv = _estimate_episode_bias(
            Y_corrected[i], dates_list[i], ct2_list[i], breakpoints[s.name]
        )
        ep_bias[s.name]     = b
        ep_bias_var[s.name] = bv

    # ── Correlated model (daily factor, cross-sample PCA on PC1) ──────────────
    all_dates, daily_corr, corr_sc, daily_corr_var = _estimate_correlated_model(
        Y_cor2, dates_list, ct2_list, stable_mask=stable_mask
    )

    return AppacModel(
        kappa=kappa,
        kappa_rsq=kappa_rsq,
        covariate_refs=covariate_refs,
        centers={s.name: ct_i for s, ct_i in zip(samples, ct2_list)},
        episode_bias=ep_bias,
        breakpoints=breakpoints,
        corr_scaling={s.name: sc_i for s, sc_i in zip(samples, corr_sc)},
        dates_global=all_dates,
        daily_corr=daily_corr,
        kappa_var=kappa_var,
        kappa_resid=kappa_resid,
        kappa_bins=kappa_bins,
        episode_bias_var=ep_bias_var,
        excluded_samples=list(exclude_set),
        daily_corr_var=daily_corr_var,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API — build_multiplier and correct
# ──────────────────────────────────────────────────────────────────────────────

def build_multiplier(sample: SampleData, model: AppacModel) -> np.ndarray:
    """
    Compute the (n_obs, n_peaks) correction multiplier for one sample.

    multiplier[t, j] = 1
        + episode_bias[episode(t), j]
        + (pc2_common[t]*amp[s] + poly(t, ep)) × pc2_load[j]
        + corr[t] × corr_scaling[j]
        + Σ_cv  kappa[cv] × (covariate[cv][t] − ref[cv])

    The PC2 term is only present when ``fit_uncorrelated_drift()`` has been
    called on the model.  corrected = raw / multiplier.

    Parameters
    ----------
    sample : SampleData
        Must have been included in the ``fit()`` call that produced ``model``.
    model  : AppacModel
        Fitted model from ``fit()`` (and optionally ``fit_uncorrelated_drift()``).

    Returns
    -------
    ndarray, shape (n_obs, n_peaks)
        Multiplicative correction factor; divide raw areas by this to obtain
        corrected areas.
    """
    name = sample.name
    dates = sample.dates

    # Episode bias: map each observation to its episode
    bps = model.breakpoints[name]
    ep_ids = np.digitize(dates, bps)                         # (n_obs,)
    bt = model.episode_bias[name][ep_ids]                    # (n_obs, n_peaks)

    # PC2 uncorrelated drift (common instrument component + sample-specific poly)
    if model.pc2_poly and name in model.pc2_poly:
        # Common component: temporal score × per-sample amplitude
        pc2_idx = np.searchsorted(model.pc2_dates, dates)
        pc2_idx = np.clip(pc2_idx, 0, len(model.pc2_common) - 1)
        common_t = model.pc2_common[pc2_idx]                        # (n_obs,)
        common_t = np.where(np.isnan(common_t), 0.0, common_t)
        common_scores = model.pc2_common_amp.get(name, 0.0) * common_t

        # Sample-specific polynomial within each episode
        poly_scores = np.zeros(len(dates))
        poly_eps = model.pc2_poly[name]
        for ep in np.unique(ep_ids):
            ep_mask = ep_ids == ep
            info    = poly_eps[ep]
            dt      = dates[ep_mask].astype(float) - info['t_center']
            cs      = info['coeffs']
            s_ep    = np.full(int(ep_mask.sum()), float(cs[0]))
            if len(cs) > 1:
                s_ep = s_ep + float(cs[1]) * dt
            if len(cs) > 2:
                s_ep = s_ep + float(cs[2]) * dt ** 2
            poly_scores[ep_mask] = s_ep

        pc2_temporal = common_scores + poly_scores           # (n_obs,)
        pc2_term = pc2_temporal[:, None] * model.pc2_peak_load[name][None, :]
    else:
        pc2_term = 0.0

    # Cross-sample correlated daily factor
    if not np.all(np.isin(dates, model.dates_global)):
        missing = np.setdiff1d(dates, model.dates_global)
        raise ValueError(
            f"build_multiplier: {len(missing)} date(s) not seen during fitting "
            f"(e.g. {missing[:3]}). Refit the model including this sample."
        )
    idx = np.searchsorted(model.dates_global, dates)
    corr_t = model.daily_corr[idx]                           # (n_obs,)
    corr_t = np.where(np.isnan(corr_t), 0.0, corr_t)        # no correction on unobserved days
    sc_corr = model.corr_scaling[name]                       # (n_peaks,)
    cr = corr_t[:, None] * sc_corr[None, :]                  # (n_obs, n_peaks)

    # Environmental covariate contributions — iterate only fitted covariates
    cov_sum = np.zeros(len(dates))
    for cv, kap in model.kappa.items():
        cov_sum += kap * (sample.covariates[cv] - model.covariate_refs[cv])

    return 1.0 + bt + pc2_term + cr + cov_sum[:, None]


def correct(Y: np.ndarray, multiplier: np.ndarray) -> np.ndarray:
    """
    Apply APPAC correction.

    Parameters
    ----------
    Y          : (n_obs, n_peaks) raw peak areas
    multiplier : (n_obs, n_peaks) from build_multiplier()

    Returns
    -------
    np.ndarray, shape (n_obs, n_peaks)
        Corrected peak areas: raw / multiplier.
    """
    return Y / multiplier


# ──────────────────────────────────────────────────────────────────────────────
# Public API — covariance propagation
# ──────────────────────────────────────────────────────────────────────────────

def propagate_covariance(
    Y: np.ndarray,
    cov_Y: np.ndarray,
    sample: SampleData,
    model: AppacModel,
    outlier_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate input covariance through the APPAC correction (GUM Eq. 13).

    Three independent uncertainty sources are summed analytically — no AD.

    Parameters
    ----------
    Y            : (n_obs, n_peaks) raw peak areas
    cov_Y        : measurement covariance of Y.  Two shapes accepted:
                     (n_obs, n_peaks)                — diagonal variances only
                     (n_obs*n_peaks, n_obs*n_peaks)  — full covariance matrix
    sample       : SampleData for this sample
    model        : fitted AppacModel
    outlier_mask : optional bool array (n_obs,) from flag_outliers(); True =
                   flagged injection.  Flagged rows are excluded before
                   propagation.  Only supported for diagonal cov_Y; pass
                   pre-subsetted data for full covariance matrices.

    Returns
    -------
    Y_corrected  : (n_valid, n_peaks) corrected peak areas
    cov_corrected: (n_valid*n_peaks, n_valid*n_peaks) propagated covariance
    where n_valid = n_obs when outlier_mask is None, else n_obs - n_flagged.
    """
    if outlier_mask is not None:
        keep = ~outlier_mask
        if cov_Y.ndim == 2 and cov_Y.shape == Y.shape:
            cov_Y = cov_Y[keep]
        else:
            raise ValueError(
                "Full covariance matrix with outlier_mask is not supported. "
                "Pre-subset Y, cov_Y, and sample before calling."
            )
        s_red = SampleData(
            name=sample.name,
            Y=sample.Y[keep],
            dates=sample.dates[keep],
            covariates={cv: arr[keep] for cv, arr in sample.covariates.items()},
        )
        return propagate_covariance(Y[keep], cov_Y, s_red, model)

    M = build_multiplier(sample, model)   # (n_obs, n_peaks) — pure numpy
    Y_corr = Y / M
    n_obs, n_peaks = Y.shape
    M_flat = M.ravel()

    # ── Measurement term: Σ_Y / outer(M_flat, M_flat) ────────────────────────
    # J_Y = diag(1/M_flat) so J_Y @ Σ_Y @ J_Y.T = Σ_Y / outer(M, M)
    if cov_Y.shape == Y.shape:
        cov_out = np.diag(cov_Y.ravel() / M_flat ** 2)
    else:
        cov_out = cov_Y / np.outer(M_flat, M_flat)

    # ── κ uncertainty terms — one rank-1 outer product per covariate ──────────
    # s_κ[t,j] = ∂Y_corr[t,j]/∂κ = −Y[t,j]·ΔP[t] / M[t,j]²
    for cv, kap_var in model.kappa_var.items():
        if not (kap_var > 0):
            continue
        dp = sample.covariates[cv] - model.covariate_refs[cv]   # (n_obs,)
        s_kappa = -(Y * dp[:, None]) / M ** 2                    # (n_obs, n_peaks)
        s_flat  = s_kappa.ravel()
        cov_out += kap_var * np.outer(s_flat, s_flat)

    # ── Episode bias uncertainty — sub-block updates per (episode, peak) ──────
    # s_b_{e,j}[t,k] = −Y[t,j]/M[t,j]² for episode(t)=e and k=j, else 0.
    # Only the n_e×n_e block within episode e is non-zero, so we update that
    # sub-block directly rather than forming a full n×n outer product.
    bps    = model.breakpoints[sample.name]
    ep_ids = np.digitize(sample.dates, bps)        # (n_obs,) — reused below
    bias_var = model.episode_bias_var.get(sample.name)
    if bias_var is not None:
        n_ep = bias_var.shape[0]
        for e in range(n_ep):
            t_idx = np.where(ep_ids == e)[0]
            if len(t_idx) == 0:
                continue
            for j in range(n_peaks):
                var_bej = float(bias_var[e, j])
                if not np.isfinite(var_bej) or var_bej <= 0.0:
                    continue
                v = -Y[t_idx, j] / M[t_idx, j] ** 2   # (n_e,)
                flat_idx = t_idx * n_peaks + j
                cov_out[np.ix_(flat_idx, flat_idx)] += var_bej * np.outer(v, v)

    # ── Between-sample correlated-factor uncertainty — rank-1 per day ─────────
    # daily_corr[t] is estimated from n_t stable instruments; its variance is
    # the between-instrument SEM² stored in daily_corr_var.
    # Sensitivity: ∂(Y/M)/∂corr[t] = −Y[t,j]·sc[j] / M[t,j]²  for all j.
    # Different peaks on the same day are co-varied (one shared corr[t]).
    if len(model.daily_corr_var) > 0:
        dc_idx = np.searchsorted(model.dates_global, sample.dates)
        dcv    = model.daily_corr_var[dc_idx]           # (n_obs,)
        sc_c   = model.corr_scaling[sample.name]        # (n_peaks,)
        for t in range(n_obs):
            v_dcv = float(dcv[t])
            if not (v_dcv > 0):
                continue
            s_t = -(Y[t, :] * sc_c) / M[t, :] ** 2    # (n_peaks,)
            flat_t = np.arange(t * n_peaks, (t + 1) * n_peaks)
            cov_out[np.ix_(flat_t, flat_t)] += v_dcv * np.outer(s_t, s_t)

    # ── PC2 polynomial uncertainty — rank-1 per coefficient per episode ───────
    # poly_score(t) = Σ_ci coeffs[ci]·(t−tc)^ci; uncertainty in each coefficient
    # propagates across time within the episode and across peaks through the
    # PC2 peak loading vector.
    # Sensitivity: ∂(Y/M)/∂coeffs[ci] = −Y[t,j]·load[j]·(t−tc)^ci / M[t,j]²
    if model.pc2_poly and sample.name in model.pc2_poly:
        poly_eps  = model.pc2_poly[sample.name]
        pc2_load  = model.pc2_peak_load.get(sample.name)
        if pc2_load is not None and len(pc2_load) == n_peaks:
            for e, info in enumerate(poly_eps):
                cv = info.get('coeffs_var')
                if cv is None or not np.any(np.isfinite(cv) & (cv > 0)):
                    continue
                t_ep = np.where(ep_ids == e)[0]
                if len(t_ep) == 0:
                    continue
                dt  = sample.dates[t_ep].astype(float) - info['t_center']
                basis = [np.ones(len(dt)), dt, dt ** 2][: len(cv)]  # up to degree 2
                for ci, (b_ci, var_ci) in enumerate(zip(basis, cv)):
                    if not (var_ci > 0 and np.isfinite(var_ci)):
                        continue
                    for j in range(n_peaks):
                        s_j = -(Y[t_ep, j] * pc2_load[j] * b_ci) / M[t_ep, j] ** 2
                        flat_j = t_ep * n_peaks + j
                        cov_out[np.ix_(flat_j, flat_j)] += var_ci * np.outer(s_j, s_j)

    return Y_corr, cov_out


# ──────────────────────────────────────────────────────────────────────────────
# Public utility — outlier flagging
# ──────────────────────────────────────────────────────────────────────────────

def flag_outliers(
    samples: list[SampleData],
    model: AppacModel,
    k: float = 3.0,
) -> dict[str, np.ndarray]:
    """
    Flag injection-level outliers using a post-fit within-episode Hampel identifier.

    After full APPAC correction the log-residuals should be ~N(0, σ²).
    An injection is flagged when any of its peaks has a log-residual that
    deviates from the within-episode median by more than k × 1.4826 × MAD
    (where 1.4826 is the consistency factor making MAD a consistent
    estimator of σ for Gaussian data).

    Parameters
    ----------
    samples : list[SampleData]
        Same list passed to fit().
    model   : AppacModel
        Fitted model from fit() (and optionally fit_uncorrelated_drift()).
    k       : Hampel threshold in σ_MAD units (default 3.0).
              Injections whose log-residual exceeds k·MAD on any peak are flagged.

    Returns
    -------
    dict {sample_name: bool array (n_obs,)} — True = flagged outlier
    """
    result: dict[str, np.ndarray] = {}
    for s in samples:
        M   = build_multiplier(s, model)
        Y_c = correct(s.Y, M)
        ct  = model.centers[s.name]
        r   = np.log(Y_c / ct[None, :])          # (n_obs, n_peaks)

        bps    = model.breakpoints[s.name]
        ep_ids = np.digitize(s.dates, bps)        # episode index per obs

        flagged = np.zeros(len(s.dates), dtype=bool)
        for ep in np.unique(ep_ids):
            rows = np.where(ep_ids == ep)[0]
            if len(rows) < 4:
                continue
            r_ep     = r[rows]                                         # (n_ep, n_peaks)
            med      = np.median(r_ep, axis=0)                        # (n_peaks,)
            mad      = np.median(np.abs(r_ep - med[None, :]), axis=0) # (n_peaks,)
            sigma_mad = 1.4826 * mad
            exceed    = np.any(
                np.abs(r_ep - med[None, :]) > k * sigma_mad[None, :],
                axis=1,
            )
            flagged[rows[exceed]] = True

        result[s.name] = flagged
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Public utility — PC2 uncorrelated drift model
# ──────────────────────────────────────────────────────────────────────────────

def fit_uncorrelated_drift(
    samples: list[SampleData],
    model: AppacModel,
    min_pts_linear: int = 30,
    min_pts_quadratic: int = 90,
    alpha: float = 0.05,
    min_drift_frac: float = 1e-3,
) -> AppacModel:
    """
    Add a two-level PC2 drift model on top of an existing AppacModel.

    Level 1 — PCA(1) per sample
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Fully corrected log-residuals log(Y_c/ct) are daily-averaged and
    decomposed via SVD.  PC2 captures the *peak-specific* (uncorrelated)
    temporal variation:
      - PC2 scores   : one scalar per day — how much the drift occurred
      - PC2 loadings : one vector per sample — which peaks are affected

    Level 2 — cross-sample PCA(2) on the PC2 score matrix
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The (n_days_global × n_samples) matrix of PC2 scores is decomposed
    with NaN-robust PCA (_pca_nan):
      - PCA(2).PC1 : component *shared across all samples* — instrument
                     effect (injected-amount variation, thermal drift, etc.)
                     Its temporal scores (pc2_common) should be used for
                     breakpoint detection: step changes here indicate
                     maintenance events that affect all cylinders.
      - Residual    : per-sample deviation from the common component —
                     sample-specific chemistry drift (PLOT column
                     deactivation, CO→carbonyl, olefine polymerisation).
                     A polynomial (degree ≤ 2 selected by sequential F-test)
                     is fitted within each episode.

    Correction term added to build_multiplier for sample s, peak j, time t:
        Δ[t,j] = ( pc2_common[t] × pc2_common_amp[s]
                   + poly_score(t, episode(t), s) )
                  ×  pc2_peak_load[j, s]

    Parameters
    ----------
    min_pts_linear    : minimum unique days in episode for linear term
    min_pts_quadratic : minimum unique days for quadratic term
    alpha             : F-test significance level for polynomial degree selection
    min_drift_frac    : practical lower bound on |Δscore| over episode span;
                        linear/quadratic terms are only accepted when the
                        statistical test AND this bound are both satisfied

    Returns
    -------
    Updated AppacModel with pc2_dates, pc2_common, pc2_common_amp,
    pc2_peak_load, and pc2_poly populated.
    """
    from dataclasses import replace as _replace

    # ── Level 1: PCA(1) per sample ────────────────────────────────────────────
    pc2_dates, daily_mat, peak_loads = _build_pc2_daily_matrix(samples, model)

    # ── Level 2: cross-sample PCA(2) on stable samples only ──────────────────
    exclude_set = set(model.excluded_samples)
    stable = np.array([s.name not in exclude_set for s in samples])
    coverage = np.mean(~np.isnan(daily_mat), axis=0)
    use_cols = stable & (coverage > 0.3)
    if use_cols.sum() < 2:
        raise ValueError(
            "fit_uncorrelated_drift: fewer than 2 stable samples have >30% "
            "date coverage — cannot run cross-sample PCA(2)."
        )

    # standardize=False: columns are already in log-ratio units; preserving
    # absolute magnitudes is essential so that pc2_term stays in fractional units.
    scores_pca2, loadings_pca2, _ = _pca_nan(daily_mat[:, use_cols],
                                              n_components=1, standardize=False)
    # scores_pca2 : (n_days, 1), NaN for all-NaN rows
    # loadings_pca2: (n_valid_samples, 1)

    valid_names = [s.name for s, ok in zip(samples, use_cols) if ok]
    pc2_common = scores_pca2[:, 0]                        # (n_days,)
    pc2_common_amp: dict[str, float] = {
        name: float(loadings_pca2[j, 0])
        for j, name in enumerate(valid_names)
    }
    for s in samples:
        pc2_common_amp.setdefault(s.name, 0.0)

    # ── Per-sample residual polynomial ────────────────────────────────────────
    pc2_poly: dict[str, list] = {}

    for i, s in enumerate(samples):
        unique_d = np.unique(s.dates)
        rows     = np.searchsorted(pc2_dates, unique_d)
        samp_sc  = daily_mat[rows, i]                    # (n_d_s,), NaN if missing
        amp      = pc2_common_amp[s.name]
        comm_sc  = pc2_common[rows]                      # (n_d_s,)

        valid    = ~(np.isnan(samp_sc) | np.isnan(comm_sc))
        residual = np.where(valid, samp_sc - amp * comm_sc, 0.0)

        pc2_poly[s.name] = _fit_polynomial_trend(
            residual, unique_d, model.breakpoints[s.name],
            min_pts_linear=min_pts_linear,
            min_pts_quadratic=min_pts_quadratic,
            alpha=alpha,
            min_drift_frac=min_drift_frac,
        )

    return _replace(
        model,
        pc2_dates=pc2_dates,
        pc2_common=pc2_common,
        pc2_common_amp=pc2_common_amp,
        pc2_peak_load=peak_loads,
        pc2_poly=pc2_poly,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public utility — breakpoint detection
# ──────────────────────────────────────────────────────────────────────────────

def _bhattacharyya_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Bhattacharyya distance between two multivariate Gaussians estimated from samples.

    Used by ``detect_breakpoints`` as the split criterion in binary segmentation.
    A large D_B at a candidate split point indicates a genuine distributional
    shift between the two halves.  The *sharpness* of the D_B profile over all
    candidate splits distinguishes step changes (one sharp peak) from slow ramps
    (flat profile, because any split of a ramp produces a similar mean difference).

    Parameters
    ----------
    X : (n_x, n_features)  observations from the left segment
    Y : (n_y, n_features)  observations from the right segment

    Returns
    -------
    float  Bhattacharyya distance; 0.0 if either segment has fewer than 2 rows
    """
    if len(X) < 2 or len(Y) < 2:
        return 0.0
    mu_x, mu_y = X.mean(0), Y.mean(0)
    sx = np.cov(X, rowvar=False)
    sy = np.cov(Y, rowvar=False)
    # Scalar case
    if sx.ndim == 0:
        sx = np.array([[float(sx)]])
        sy = np.array([[float(sy)]])
        mu_x = mu_x.reshape(1)
        mu_y = mu_y.reshape(1)
    s_avg = 0.5 * (sx + sy)
    diff = mu_y - mu_x
    term1 = 0.125 * diff @ np.linalg.solve(s_avg, diff)
    _, ld_avg = np.linalg.slogdet(s_avg)
    _, ld_x   = np.linalg.slogdet(sx)
    _, ld_y   = np.linalg.slogdet(sy)
    term2 = 0.5 * (ld_avg - 0.5 * (ld_x + ld_y))
    return float(term1 + term2)


def detect_breakpoints(
    sample: SampleData,
    model: AppacModel,
    min_segment: int = 14,
    d_crit: float = 2.0,
    sharpness_thresh: float = 1.3,
) -> np.ndarray:
    """
    Detect step changes in the uncorrelated (bias) component of one sample.

    Uses Bhattacharyya distance between the left and right halves at every
    candidate split point as the segmentation criterion.  A split is accepted
    when D_B(best) > d_crit AND the profile is sharply peaked (sharpness =
    D_B(best) / mean(D_B_all) > sharpness_thresh).

    The sharpness test is the key innovation: a genuine step change produces a
    D_B profile with a sharp spike at the step location (sharpness ≈ 2.5–3).
    A slow ramp produces a nearly flat D_B profile (sharpness ≈ 1.0–1.1)
    because the mean difference between any two halves is always ≈ a·n/2
    regardless of where the split falls.  This rejects the false breakpoints
    that binary segmentation on mean residuals generates for a degrading sample.

    Parameters
    ----------
    min_segment       : minimum number of *days* on each side of a breakpoint
    d_crit            : minimum D_B at the best split to accept it
    sharpness_thresh  : minimum ratio D_B(best) / mean(D_B_all) to accept it

    Returns
    -------
    sorted array of breakpoint dates (integer day numbers)
    """
    M = build_multiplier(sample, model)
    Y_c = correct(sample.Y, M)
    ct = model.centers[sample.name]
    log_resid = np.log(Y_c / ct[None, :])   # (n_obs, n_peaks) — keep multivariate

    unique_dates = np.unique(sample.dates)
    # (n_days, n_peaks) — multivariate daily means
    r_daily = np.array([log_resid[sample.dates == d].mean(axis=0) for d in unique_dates])

    def _binseg(R: np.ndarray, dates: np.ndarray) -> list[int]:
        n = len(R)
        if n < 2 * min_segment:
            return []
        db_vals = np.zeros(n)
        for k in range(min_segment, n - min_segment):
            db_vals[k] = _bhattacharyya_distance(R[:k], R[k:])
        candidates = db_vals[min_segment : n - min_segment]
        if len(candidates) == 0:
            return []
        best_k   = int(np.argmax(candidates)) + min_segment
        best_db  = db_vals[best_k]
        mean_db  = candidates.mean()
        sharpness = best_db / max(mean_db, 1e-12)
        if best_db < d_crit or sharpness < sharpness_thresh:
            return []
        return (_binseg(R[:best_k], dates[:best_k])
                + [int(dates[best_k])]
                + _binseg(R[best_k:], dates[best_k:]))

    return np.array(sorted(_binseg(r_daily, unique_dates)), dtype=int)


def detect_breakpoints_global(
    gen_model: "GeneralizedCorrectionModel",
    min_segment: int = 30,
    d_crit: float = 0.15,
    sharpness_thresh: float = 1.3,
    guard_days: int = 60,
    cluster_window: int = 90,
) -> "BreakpointResult":
    """
    Detect instrument-wide breakpoints from the PPCA-imputed PC2 signal and
    classify them as clean steps or contamination periods.

    Operates on ``gen_model.pc2_scores`` from ``fit_generalized_correction()``.
    The PPCA imputation propagates step changes that appear in ≥ 2
    contemporaneous instruments into the cross-sample PC2 score; events
    confined to a single instrument average toward zero and do not pass the
    Bhattacharyya sharpness filter.

    Classification of detected dates
    ---------------------------------
    After binary segmentation, the detected dates are grouped into clusters
    (consecutive dates within ``cluster_window`` days).

    - **Isolated event** (cluster of 1): a single clean step — cylinder swap
      or detector service.  Returned in ``BreakpointResult.breakpoints``.
    - **Cluster of 2+**: the instrument was contaminated and underwent
      repeated adjustments during recovery.  Data in the window
      [first_date − min_segment, last_date + min_segment] is unreliable.
      Returned in ``BreakpointResult.dirty_windows``; all observations within
      the window should be excluded from fitting.

    Guard zone
    ----------
    Binary segmentation tends to produce boundary artefacts: the split nearest
    the end (or start) of the series always achieves a high D_B simply because
    the mean of the bulk differs from the mean of the last few days.
    ``guard_days`` trims the signal before segmentation so the last and first
    ``guard_days`` days are not candidates for splitting.

    Parameters
    ----------
    gen_model        : from fit_generalized_correction() with PPCA imputation
    min_segment      : minimum days on each side of a candidate breakpoint;
                       also used as the buffer added around dirty windows
    d_crit           : minimum Bhattacharyya distance (1-D; ≈0.15 for this signal)
    sharpness_thresh : minimum D_B(best) / mean(D_B_all)
    guard_days       : days trimmed from each end of the signal before detection
    cluster_window   : maximum gap (days) for two detections to be considered
                       part of the same contamination event

    Returns
    -------
    BreakpointResult with .breakpoints and .dirty_windows
    """
    valid  = np.isfinite(gen_model.pc2_scores)
    dates  = gen_model.dates_global[valid]
    signal = gen_model.pc2_scores[valid].reshape(-1, 1)

    # Trim guard zone from both ends
    d_lo, d_hi = dates[0] + guard_days, dates[-1] - guard_days
    keep   = (dates >= d_lo) & (dates <= d_hi)
    dates  = dates[keep]
    signal = signal[keep]

    def _binseg(R: np.ndarray, ds: np.ndarray) -> list[int]:
        n = len(R)
        if n < 2 * min_segment:
            return []
        db_vals = np.zeros(n)
        for k in range(min_segment, n - min_segment):
            db_vals[k] = _bhattacharyya_distance(R[:k], R[k:])
        candidates = db_vals[min_segment : n - min_segment]
        if len(candidates) == 0:
            return []
        best_k    = int(np.argmax(candidates)) + min_segment
        best_db   = db_vals[best_k]
        mean_db   = candidates.mean()
        sharpness = best_db / max(mean_db, 1e-12)
        if best_db < d_crit or sharpness < sharpness_thresh:
            return []
        return (_binseg(R[:best_k], ds[:best_k])
                + [int(ds[best_k])]
                + _binseg(R[best_k:], ds[best_k:]))

    all_dates = sorted(_binseg(signal, dates))
    if not all_dates:
        return BreakpointResult(breakpoints=np.array([], dtype=int),
                                dirty_windows=[])

    # Cluster consecutive detections within cluster_window days
    clusters: list[list[int]] = []
    current = [all_dates[0]]
    for d in all_dates[1:]:
        if d - current[0] <= cluster_window:
            current.append(d)
        else:
            clusters.append(current)
            current = [d]
    clusters.append(current)

    breakpoints: list[int]         = []
    dirty_windows: list[tuple[int,int]] = []
    for cluster in clusters:
        if len(cluster) == 1:
            breakpoints.append(cluster[0])
        else:
            # Contamination period: buffer by min_segment on each side
            dirty_windows.append((cluster[0]  - min_segment,
                                  cluster[-1] + min_segment))

    return BreakpointResult(
        breakpoints=np.array(sorted(breakpoints), dtype=int),
        dirty_windows=dirty_windows,
    )


def apply_global_breakpoints(
    samples: list["SampleData"],
    result: "BreakpointResult",
) -> dict[str, np.ndarray]:
    """
    Filter the clean breakpoints from a BreakpointResult to each sample's
    observed date range.

    Parameters
    ----------
    samples : reference SampleData list
    result  : BreakpointResult from detect_breakpoints_global()

    Returns
    -------
    dict mapping sample name → array of breakpoint dates within that sample's range
    """
    out: dict[str, np.ndarray] = {}
    for s in samples:
        lo, hi = int(s.dates.min()), int(s.dates.max())
        mask = (result.breakpoints > lo) & (result.breakpoints < hi)
        out[s.name] = result.breakpoints[mask]
    return out


def flag_dirty_windows(
    samples: list["SampleData"],
    result: "BreakpointResult",
    existing_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """
    Build per-sample boolean outlier masks that exclude all observations
    falling inside the contamination windows from detect_breakpoints_global().

    Parameters
    ----------
    samples        : reference SampleData list
    result         : BreakpointResult from detect_breakpoints_global()
    existing_masks : existing outlier masks to combine with (logical OR);
                     if None, starts from all-False masks

    Returns
    -------
    dict mapping sample name → bool array (n_obs,); True = excluded
    """
    masks: dict[str, np.ndarray] = {}
    for s in samples:
        base = (existing_masks[s.name].copy()
                if existing_masks is not None
                else np.zeros(len(s.dates), dtype=bool))
        for lo, hi in result.dirty_windows:
            base |= (s.dates >= lo) & (s.dates <= hi)
        masks[s.name] = base
    return masks


# ──────────────────────────────────────────────────────────────────────────────
# Chi-square refinement of center values
# ──────────────────────────────────────────────────────────────────────────────

def chi_square_fit(
    samples: list[SampleData],
    model: AppacModel,
    covariate_refs: dict[str, float] | None = None,
) -> AppacModel:
    """
    Refine center values by minimising χ² of corrected residuals.

    This step is essential for unbiased results.  The arithmetic mean of
    log-normal peak areas sits above the geometric mean (mode of the
    multiplicative model) by exp(σ²/2).  Without this refinement, the
    arithmetic-mean centers introduce a systematic positive bias into the
    corrected areas.  χ² minimisation finds the cf* that symmetrises the
    residuals — equivalent to finding the geometric mean of the corrected
    distribution — and must be run after fit() and fit_uncorrelated_drift().

    Algorithm
    ---------
    χ²(cf) is quadratic in cf, so three evaluation points determine it exactly.
    For cf ∈ {0.99, 1.0, 1.01}:
      1. Set trial centers ct_trial = cf × ct₀ for all samples.
      2. Re-fit the full model with ct_init = ct_trial so that PCA, κ, bias,
         and the correlated model all respond to the center shift (matching the
         R behaviour where ct is fixed throughout the pipeline).
      3. Record  χ²(cf, j) = Σ_t (Y_corrected[t,j] − ct_trial[j])²  per peak j.
    Fit a quadratic to the three points → analytical minimum cf*[j] = −b₁/(2b₂).

    A final re-fit at the refined centers returns a fully consistent model.
    Total cost: 4 full fit() calls (3 grid + 1 final).
    """
    if covariate_refs is None:
        covariate_refs = model.covariate_refs
    covariate_names = list(covariate_refs.keys())

    # Three points suffice for an exact quadratic fit
    cfl = np.array([0.99, 1.0, 1.01])
    ct0 = {s.name: model.centers[s.name].copy() for s in samples}
    chisq: dict[str, list[np.ndarray]] = {s.name: [] for s in samples}

    for cf in cfl:
        ct_scaled = {s.name: cf * ct0[s.name] for s in samples}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_cf = fit(samples, covariate_refs=covariate_refs,
                       covariate_names=covariate_names,
                       breakpoints=model.breakpoints,
                       ct_init=ct_scaled,
                       exclude=model.excluded_samples or None)
        for s in samples:
            M = build_multiplier(s, m_cf)
            Y_c = correct(s.Y, M)
            chisq[s.name].append(
                np.sum((Y_c - ct_scaled[s.name][None, :]) ** 2, axis=0)
            )

    # Find optimal cf* per peak from quadratic fit to χ²(cf)
    refined_centers = {}
    for s in samples:
        curves = np.array(chisq[s.name])   # (n_points, n_peaks)
        ct_new = ct0[s.name].copy()
        for j in range(curves.shape[1]):
            b2, b1, _ = np.polyfit(cfl, curves[:, j], 2)
            if abs(b2) > 1e-12:
                ct_new[j] *= float(np.clip(-b1 / (2 * b2), 0.99, 1.01))
        refined_centers[s.name] = ct_new

    # Re-fit once at the refined centers so all model fields are consistent.
    # The grid fits above are used only to locate the χ² minimum; this final
    # fit is the one the caller uses.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(
            samples,
            covariate_refs=covariate_refs,
            covariate_names=covariate_names,
            breakpoints=model.breakpoints,
            ct_init=refined_centers,
            exclude=model.excluded_samples or None,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Generalised drift/bias correction for unknown samples
# ──────────────────────────────────────────────────────────────────────────────

def fit_generalized_correction(
    samples: list[SampleData],
    model: AppacModel,
    outlier_masks: dict[str, np.ndarray] | None = None,
    kappa_fixed: dict[str, float] | None = None,
) -> GeneralizedCorrectionModel:
    """
    Build a generalised drift/bias correction from κ-only-corrected reference data.

    Unlike the full APPAC model, this correction uses only the physically
    generalisable κ term and is therefore valid for unknown samples whose
    composition need not be known.

    Architecture
    ------------
    This function implements steps 2–4 of the four-step pipeline:
      1. (caller) Unbiased κ — full no-breakpoint pass through chi_square_fit.
      2. Pressure correction — divide by 1 + κ·ΔP using kappa_fixed (or
         model.kappa if kappa_fixed is not supplied).  κ is the only parameter
         that generalises to unknown samples.
      3. Missing value imputation — PPCA soft-impute on the (n_days × n_samples)
         daily log-ratio matrix.  Uses cross-sample correlations rather than
         column means so that step changes visible in only a subset of
         contemporaneous instruments are faithfully propagated.
      4. Breakpoint and drift detection — binary segmentation on the
         PPCA-imputed PC2 score (cross-sample PCA, standardize=False).

    For each reference sample (step 2 / 3):
      1. Apply κ correction; remove flagged injections.
      2. Compute ep0_ref: geometric mean of κ-corrected areas in episode 0.
      3. Compute log-ratios log(Y_κ / ep0_ref) and average across peaks
         (equal weights — consistent with the Z-score / dispersion-normalised
         representation used throughout the pipeline).
      4. Daily-average the peak-mean log-ratios → one scalar per (sample, day).

    PPCA cross-sample PCA (step 3 / 4, standardize=False):
      PC1 — shared correlated daily variation
      PC2 — slow drift and inter-episode bias shifts

    Stable samples (not in model.excluded_samples) drive the PCA; excluded
    samples receive a loading of 0 and are not used for unknown correction.

    Parameters
    ----------
    samples       : reference SampleData list (same as passed to fit())
    model         : fitted AppacModel (supplies breakpoints, excluded_samples,
                    covariate_refs; kappa is taken from kappa_fixed if given)
    outlier_masks : {sample_name: bool (n_obs,)} from flag_outliers(); True = excluded
    kappa_fixed   : unbiased κ from the no-breakpoint chi_square_fit pass.
                    When supplied, model.kappa is ignored for pressure correction.
    """
    if outlier_masks is None:
        outlier_masks = {}

    kappa           = kappa_fixed if kappa_fixed is not None else model.kappa
    covariate_names = list(kappa.keys())
    exclude_set     = set(model.excluded_samples)

    all_dates = np.unique(np.concatenate([s.dates for s in samples]))
    n_days    = len(all_dates)
    daily_mat = np.full((n_days, len(samples)), np.nan)
    ep0_ref: dict[str, np.ndarray] = {}

    for i, s in enumerate(samples):
        cov_sub = {cv: s.covariates[cv] for cv in covariate_names if cv in s.covariates}
        Y_kappa = _covariate_correct(s.Y, cov_sub, kappa, model.covariate_refs)

        valid   = ~outlier_masks.get(s.name, np.zeros(len(s.dates), dtype=bool))
        Y_v     = Y_kappa[valid]
        dates_v = s.dates[valid]

        bps      = model.breakpoints.get(s.name, np.array([], dtype=int))
        ep_ids   = np.digitize(dates_v, bps)
        ep0_mask = ep_ids == 0
        if not ep0_mask.any():
            raise ValueError(
                f"[{s.name}] no valid observations in episode 0 after outlier removal."
            )
        ct_ep0          = np.exp(np.log(Y_v[ep0_mask]).mean(axis=0))
        ep0_ref[s.name] = ct_ep0

        # Peak-mean log-ratio (equal weights across peaks)
        r = np.log(Y_v / ct_ep0[None, :]).mean(axis=1)   # (n_valid,)
        for d in np.unique(dates_v):
            row = int(np.searchsorted(all_dates, d))
            daily_mat[row, i] = float(r[dates_v == d].mean())

    # PPCA on stable samples only; standardize=False preserves fractional units.
    # Soft-impute uses cross-sample correlations for missing days rather than
    # column means, so step changes survive in the PC2 score even when only a
    # subset of instruments was running.
    stable   = np.array([s.name not in exclude_set for s in samples])
    pca_cols = np.where(stable)[0]
    if len(pca_cols) < 2:
        raise ValueError(
            "fit_generalized_correction: fewer than 2 stable samples — cannot run PCA."
        )

    scores, loadings, _ = _ppca_nan(daily_mat[:, pca_cols], n_components=2, standardize=False)

    # Sign convention: largest-|loading| sample is positive for each component
    for comp in range(2):
        pk = int(np.argmax(np.abs(loadings[:, comp])))
        if loadings[pk, comp] < 0:
            loadings[:, comp] *= -1
            scores[:, comp]   *= -1

    stable_names     = [s.name for s, ok in zip(samples, stable) if ok]
    name_to_load_idx = {name: j for j, name in enumerate(stable_names)}

    pc1_loading: dict[str, float] = {}
    pc2_loading: dict[str, float] = {}
    for s in samples:
        j = name_to_load_idx.get(s.name)
        if j is not None:
            pc1_loading[s.name] = float(loadings[j, 0])
            pc2_loading[s.name] = float(loadings[j, 1])
        else:
            pc1_loading[s.name] = 0.0
            pc2_loading[s.name] = 0.0

    return GeneralizedCorrectionModel(
        dates_global=all_dates,
        pc1_scores=scores[:, 0],
        pc2_scores=scores[:, 1],
        pc1_loading=pc1_loading,
        pc2_loading=pc2_loading,
        ep0_ref=ep0_ref,
    )


def build_multiplier_unknown(
    unknown: SampleData,
    gen_model: GeneralizedCorrectionModel,
    model: AppacModel,
    instrument: str,
) -> np.ndarray:
    """
    Build a correction multiplier for an unknown sample on a known instrument.

    The unknown sample's composition need not be known.  The correction is a
    uniform fractional factor across all peaks (linear with peak area), comprising:
      - κ covariate correction (pressure)
      - PC1 correlated daily factor
      - PC2 drift/bias

    Consistent with build_multiplier(): returns a (n_obs, n_peaks) array M
    such that  corrected = raw / M.

    Parameters
    ----------
    unknown    : SampleData — name need not be in model; covariates must be present
    gen_model  : from fit_generalized_correction()
    model      : fitted AppacModel (for kappa parameters)
    instrument : name of the reference sample (known cylinder) whose instrument
                 carried out the unknown measurement

    Returns
    -------
    (n_obs, n_peaks)  multiplier
    """
    if instrument not in gen_model.pc2_loading:
        raise ValueError(
            f"instrument '{instrument}' not found in generalised model. "
            f"Available: {sorted(gen_model.pc2_loading)}"
        )

    missing = np.setdiff1d(unknown.dates, gen_model.dates_global)
    if len(missing):
        raise ValueError(
            f"build_multiplier_unknown: {len(missing)} date(s) not covered by the "
            f"generalised model (e.g. {missing[:3]}). "
            "Dates beyond the training range are not yet supported."
        )

    idx   = np.searchsorted(gen_model.dates_global, unknown.dates)
    pc1_t = np.where(np.isnan(gen_model.pc1_scores[idx]), 0.0, gen_model.pc1_scores[idx])
    pc2_t = np.where(np.isnan(gen_model.pc2_scores[idx]), 0.0, gen_model.pc2_scores[idx])

    drift = (pc1_t * gen_model.pc1_loading[instrument]
             + pc2_t * gen_model.pc2_loading[instrument])   # (n_obs,) — uniform across peaks

    cov_sum = np.zeros(len(unknown.dates))
    for cv, kap in model.kappa.items():
        if cv in unknown.covariates:
            cov_sum += kap * (unknown.covariates[cv] - model.covariate_refs[cv])

    n_peaks = unknown.Y.shape[1]
    return 1.0 + cov_sum[:, None] + drift[:, None] * np.ones(n_peaks)
