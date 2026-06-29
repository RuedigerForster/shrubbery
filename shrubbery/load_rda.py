"""
load_rda.py — Load APPAC dataset files into SampleData objects.

Accepted formats
----------------
.parquet   — preferred; read directly with pandas (no extra dependencies)
.rda       — legacy R data format; requires the ``rdata`` package

Input shape (long): one row per (injection × peak).
Output: list[SampleData] ready for appac.fit().

Expected columns (names are matched case-insensitively):
  sample.name     — sample identifier
  file.name       — unique injection identifier (used to group peaks)
  injection.date  — integer days since 1970-01-01, or parseable date string
  air.pressure    — ambient pressure [mbar = hPa]
  raw.area        — raw GC peak area
  peak.name       — compound name
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

try:
    import rdata
    _READER = "rdata"
except ImportError:
    _READER = None


def _load_df(path: str) -> pd.DataFrame:
    """Read a .parquet or .rda file and return a flat DataFrame."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        df.columns = [str(c).lower() for c in df.columns]
        return df

    if _READER == "rdata":
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            store = rdata.read_rda(path)
        df = next(iter(store.values()))
    else:
        raise ImportError(
            "The 'rdata' package is required to read .rda files. "
            "Install it with: pip install rdata"
        )

    df.columns = [str(c).lower() for c in df.columns]
    return df


def _to_float(series: pd.Series) -> np.ndarray:
    """Convert a pandas Series of any numeric type to a plain float64 array."""
    return np.array(series.to_list(), dtype=float)


def _to_int_dates(series: pd.Series) -> np.ndarray:
    """
    Convert injection.date to integer day numbers.

    R IDate / Date integers are days since 1970-01-01.  String dates
    ('YYYY-MM-DD') are also handled via pandas.
    """
    sample = series.iloc[0]
    if isinstance(sample, (int, np.integer)) or pd.api.types.is_integer_dtype(series):
        return np.array(series.to_list(), dtype=int)
    # nullable Int32 (pandas)
    try:
        return np.array(series.to_list(), dtype=int)
    except (TypeError, ValueError):
        pass
    # String dates
    dt = pd.to_datetime(series, errors="coerce")
    origin = pd.Timestamp("1970-01-01")
    return ((dt - origin).dt.days).to_numpy(dtype=int)


def load_samples(
    path: str,
    sample_col:   str = "sample.name",
    file_col:     str = "file.name",
    date_col:     str = "injection.date",
    pressure_col: str = "air.pressure",
    area_col:     str = "raw.area",
    peak_col:     str = "peak.name",
    min_injections: int = 10,
    pressure_ref: float | None = None,
) -> tuple[list, dict, float]:
    """
    Load an RData file and return (samples, breakpoints_stub, pressure_ref).

    Parameters
    ----------
    path            : path to the .rda file
    min_injections  : skip samples with fewer injections than this
    pressure_ref    : reference pressure [hPa]; if None, uses the overall median

    Returns
    -------
    samples         : list[SampleData]
    breakpoints     : dict {name: empty array} — no breakpoints assumed initially
    pressure_ref    : float  (median pressure across all valid observations)
    """
    from .appac import SampleData

    df = _load_df(path)

    # Check required columns exist
    required = {sample_col, file_col, date_col, pressure_col, area_col, peak_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")

    # Pivot to wide: rows = (sample, file/injection), columns = peaks
    peak_names = sorted(df[peak_col].dropna().unique().astype(str))
    print(f"  Peaks: {peak_names}")

    samples_out = []

    for sname in sorted(df[sample_col].dropna().unique().astype(str)):
        sub = df[df[sample_col].astype(str) == sname].copy()

        # Pivot: one row per injection (file), one column per peak
        sub_pivot = sub.pivot_table(
            index=file_col,
            columns=peak_col,
            values=area_col,
            aggfunc="mean",     # average if multiple rows per (file, peak)
        )
        # Reindex to global peak order, then drop peaks absent from this sample.
        # A peak is considered absent when fewer than half the injections have data.
        sub_pivot = sub_pivot.reindex(columns=peak_names)
        col_coverage  = sub_pivot.notna().mean()
        present_peaks = col_coverage[col_coverage >= 0.5].index.tolist()
        absent_peaks  = [p for p in peak_names if p not in present_peaks]
        if absent_peaks:
            warnings.warn(
                f"Sample '{sname}': peak(s) {absent_peaks} have no data (<50 % "
                f"coverage) and are excluded from this sample. "
                f"Proceeding with {len(present_peaks)} peak(s): {present_peaks}."
            )
            sub_pivot = sub_pivot[present_peaks]
        sub_pivot.dropna(how="any", inplace=True)

        n_inj = len(sub_pivot)
        if n_inj < min_injections:
            warnings.warn(
                f"Sample '{sname}': only {n_inj} complete injections — skipped "
                f"(min_injections={min_injections})"
            )
            continue

        Y = sub_pivot.to_numpy(dtype=float)

        # Dates: one per injection (file), in the same order as pivot rows
        date_lookup = (
            sub.groupby(file_col)[date_col].first()
        )
        dates_raw = date_lookup.reindex(sub_pivot.index)
        dates = _to_int_dates(dates_raw)

        # Sort chronologically
        sort_idx = np.argsort(dates)
        Y     = Y[sort_idx]
        dates = dates[sort_idx]

        # Pressure: one per injection
        pressure_lookup = sub.groupby(file_col)[pressure_col].mean()
        pressure_raw = pressure_lookup.reindex(sub_pivot.index)
        pressure = _to_float(pressure_raw)[sort_idx]

        n_nan_p = int(np.sum(~np.isfinite(pressure)))
        if n_nan_p > 0:
            warnings.warn(
                f"Sample '{sname}': {n_nan_p}/{n_inj} injections have NaN "
                "pressure — those rows are dropped."
            )
            valid = np.isfinite(pressure) & np.all(np.isfinite(Y), axis=1)
            Y, dates, pressure = Y[valid], dates[valid], pressure[valid]

        if len(Y) < min_injections:
            warnings.warn(
                f"Sample '{sname}': fewer than {min_injections} valid rows "
                "after pressure NaN removal — skipped."
            )
            continue

        samples_out.append(SampleData(
            name       = sname,
            Y          = Y,
            dates      = dates,
            covariates = {"pressure": pressure},
            peaks      = present_peaks,
        ))
        print(f"  {sname:40s}  n={len(Y):5d}  peaks={Y.shape[1]}"
              f"  P={pressure.min():.1f}–{pressure.max():.1f} hPa"
              f"  dates={dates.min()}–{dates.max()}")

    if not samples_out:
        raise ValueError("No usable samples found in the file.")

    # Pressure reference: median across all samples
    all_p = np.concatenate([s.covariates["pressure"] for s in samples_out])
    p_ref = float(np.nanmedian(all_p)) if pressure_ref is None else pressure_ref
    print(f"\n  Pressure reference: {p_ref:.2f} hPa (median of all observations)")

    breakpoints = {s.name: np.array([], dtype=int) for s in samples_out}
    return samples_out, breakpoints, p_ref
