"""
pipeline.py — the canonical end-to-end APPAC pipeline.

Both ``run_appac_on_data.py`` (text report) and ``plot_real_data.py`` (plots)
drive the *same* validated sequence through :func:`run_pipeline`, so the report
and the reference plots can never disagree about κ, centers, outliers, or the
breakpoint set.

Stages
------
1. Unbiased κ — ``fit`` (no breakpoints) → ``fit_uncorrelated_drift`` →
   ``chi_square_fit`` (arithmetic-mean centers, χ² refined).  κ is then held
   fixed for every subsequent step.
2. Instrument-wide breakpoints — detected on the cross-sample PPCA PC2 drift
   score.
3. Refit with breakpoints (κ fixed) → drift.
4. Flag outliers ∪ dirty windows, then the final generalised correction.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass


from .appac import (
    AppacModel,
    GeneralizedCorrectionModel,
    BreakpointResult,
    SampleData,
    fit,
    fit_uncorrelated_drift,
    chi_square_fit,
    flag_outliers,
    fit_generalized_correction,
    detect_breakpoints_global,
    apply_global_breakpoints,
    flag_dirty_windows,
    build_multiplier,
    correct,
)


def _recenter_on_corrected(samples, model, masks):
    """Re-center each cylinder on the mean of its FULLY-corrected areas.

    ``fit`` sets ``model.centers`` to the mean of the *κ-corrected* areas, but the
    final correction also applies episode-bias and daily/PC2 drift terms.  For the
    minor (heavy) peaks those terms carry a small per-cylinder mean shift, so the
    corrected-area mean no longer equals the center — leaving a per-cylinder
    residual *offset* that dominates the heavy-peak residual variance.  Setting the
    center to the corrected-area mean (over the kept injections) is the corrected
    "true value" — matching appac_v3's ``the_true_value`` — and removes the offset.

    Centers are only a reference for residuals; they do not enter ``build_multiplier``
    or ``propagate_covariance``, so this does not change the correction or the GUM
    uncertainty.
    """
    for s in samples:
        Yc   = correct(s.Y, build_multiplier(s, model))
        keep = ~masks[s.name]
        if keep.any():
            model.centers[s.name] = Yc[keep].mean(axis=0)
    return model


@dataclass
class PipelineResult:
    """Everything the report/plot scripts need from one pipeline run."""
    kappa:     dict                       # common κ per covariate (fixed downstream)
    m0:        AppacModel                 # no-breakpoint χ²-refined model (κ diagnostics)
    model:     AppacModel                 # final model: breakpoints + κ fixed + drift
    masks:     dict                       # per-sample outlier ∪ dirty-window mask (True = excluded)
    bpr:       BreakpointResult           # detected breakpoints + dirty windows
    gen_model: GeneralizedCorrectionModel # final generalised correction


def run_pipeline(
    samples: list[SampleData],
    covariate_refs: dict[str, float],
    exclude: list[str] | None = None,
) -> PipelineResult:
    """Run the validated APPAC pipeline end-to-end and return all artifacts.

    Single source of truth for the report and the reference plots, so the two
    never diverge.  κ is estimated once (χ²-refined, unbiased) and then held
    fixed; injections inside detected dirty windows are excluded from the final
    flagging alongside the Hampel outliers.
    """
    exclude = exclude or None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 1. unbiased κ: fit (no breakpoints) → drift → χ² center refinement
        m0 = fit(samples, covariate_refs=covariate_refs, exclude=exclude)
        m0 = fit_uncorrelated_drift(samples, m0)
        m0 = chi_square_fit(samples, m0, covariate_refs=covariate_refs)
        kappa = m0.kappa

        # 2. instrument-wide breakpoints from the cross-sample PC2 drift score
        masks0 = flag_outliers(samples, m0)
        gen0   = fit_generalized_correction(samples, m0, outlier_masks=masks0,
                                            kappa_fixed=kappa)
        bpr    = detect_breakpoints_global(gen0)
        breakpoints = apply_global_breakpoints(samples, bpr)

        # 3. refit with breakpoints (κ fixed) → drift
        model = fit(samples, covariate_refs=covariate_refs,
                    breakpoints=breakpoints, exclude=exclude, kappa_fixed=kappa)
        model = fit_uncorrelated_drift(samples, model)

        # 4. flag outliers ∪ dirty windows, then the final generalised correction
        masks = flag_outliers(samples, model)
        masks = flag_dirty_windows(samples, bpr, existing_masks=masks)
        gen_model = fit_generalized_correction(samples, model, outlier_masks=masks,
                                               kappa_fixed=kappa)
        # Re-center on the fully-corrected areas (true value) — removes the
        # per-cylinder offset on the minor peaks.  Done last: nothing above
        # consumes model.centers by value, so it only fixes the residual reference.
        model = _recenter_on_corrected(samples, model, masks)

    return PipelineResult(kappa=kappa, m0=m0, model=model, masks=masks,
                          bpr=bpr, gen_model=gen_model)
