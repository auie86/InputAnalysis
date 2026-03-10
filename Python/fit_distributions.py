"""
fit_distributions.py
--------------------
Fits numerical observations to five parametric distributions and reports
descriptive statistics, MLE parameters, and GoF p-values.

Distributions fitted
--------------------
  normal      – N(mu, sigma)
  uniform     – U(loc, loc+scale)
  exponential – Exp(loc, scale=1/lambda)
  triangular  – Tri(c, loc, scale)   [c is the shape: mode=(loc + c*scale)]
  lognormal   – LogN(s, loc, scale)  [s=sigma of the underlying normal]

Goodness-of-fit
---------------
  Kolmogorov-Smirnov (KS) test is used for all distributions.
  p-value > 0.05 → insufficient evidence to reject the fit at the 5 % level.

CSV input
---------
  The CSV file must contain numeric data.  Two layouts are supported:

  • Single-column (no header):
        1.23
        4.56
        ...

  • Single-column (with header):
        value
        1.23
        4.56
        ...

  • Multi-column: pass --column <name|0-based-index> to select the column.
        id,value,flag
        1,1.23,True
        2,4.56,False
        ...

CLI usage
---------
  python fit_distributions.py data.csv
  python fit_distributions.py data.csv --column value
  python fit_distributions.py data.csv --column 1
  python fit_distributions.py data.csv --delimiter ";"

Dependencies
------------
  numpy, scipy  (both ship with most scientific Python environments)
  Install: pip install numpy scipy
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_csv(
    filepath: str | Path,
    *,
    column: str | int | None = None,
    delimiter: str = ",",
) -> np.ndarray:
    """
    Read numeric observations from a CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    column : str, int, or None
        Which column to read.
        - None (default): expects a single data column (header optional).
        - str: column name matching a header row.
        - int: 0-based column index.
    delimiter : str, default ","
        Field delimiter character.

    Returns
    -------
    np.ndarray
        1-D float array of observations with blank rows dropped.

    Raises
    ------
    FileNotFoundError  - file does not exist.
    ValueError         - column not found, no numeric data, or parse errors.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        has_header = csv.Sniffer().has_header(sample)
        reader = csv.reader(fh, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty.")

    # Determine the target column index
    if isinstance(column, str):
        if not has_header:
            raise ValueError(
                f"column={column!r} given but no header row detected."
            )
        header = [c.strip() for c in rows[0]]
        if column not in header:
            raise ValueError(
                f"Column {column!r} not found. Available columns: {header}"
            )
        col_idx = header.index(column)
        data_rows = rows[1:]
    elif isinstance(column, int):
        col_idx = column
        data_rows = rows[1:] if has_header else rows
    else:
        # Auto-detect: single data column
        col_idx = 0
        data_rows = rows[1:] if has_header else rows
        n_cols = max(len(r) for r in data_rows) if data_rows else 1
        if n_cols > 1:
            raise ValueError(
                f"CSV has {n_cols} columns but --column was not specified. "
                "Use --column <name|index> to select one."
            )

    values: list[float] = []
    skipped = 0
    for i, row in enumerate(data_rows, start=2 if has_header else 1):
        if col_idx >= len(row):
            skipped += 1
            continue
        cell = row[col_idx].strip()
        if cell == "":
            skipped += 1
            continue
        try:
            values.append(float(cell))
        except ValueError:
            raise ValueError(
                f"Non-numeric value {cell!r} at row {i}, column index {col_idx}."
            )

    if skipped:
        print(f"  [info] {skipped} blank/missing row(s) skipped.")

    if not values:
        raise ValueError("No numeric data found in the selected column.")

    return np.asarray(values, dtype=float)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DescriptiveStats:
    n: int
    mean: float
    std: float          # sample std (ddof=1)
    variance: float
    median: float
    skewness: float
    kurtosis: float     # excess kurtosis
    minimum: float
    maximum: float

    def __str__(self) -> str:
        lines = [
            "── Descriptive Statistics ───────────────────────────────────",
            f"  n          : {self.n}",
            f"  mean       : {self.mean:.6g}",
            f"  std (s)    : {self.std:.6g}",
            f"  variance   : {self.variance:.6g}",
            f"  median     : {self.median:.6g}",
            f"  skewness   : {self.skewness:.6g}",
            f"  kurtosis   : {self.kurtosis:.6g}  (excess)",
            f"  min        : {self.minimum:.6g}",
            f"  max        : {self.maximum:.6g}",
        ]
        return "\n".join(lines)


@dataclass
class FitResult:
    distribution: str
    params: dict[str, float]
    ks_statistic: float
    p_value: float
    success: bool
    error: str | None = None

    def __str__(self) -> str:
        if not self.success:
            return (
                f"  {self.distribution:<14}: FIT FAILED – {self.error}"
            )
        param_str = ",  ".join(f"{k}={v:.6g}" for k, v in self.params.items())
        return (
            f"  {self.distribution:<14}: {param_str}\n"
            f"  {'':14}  KS={self.ks_statistic:.4f},  p={self.p_value:.4f}"
            + ("  ✓ good fit" if self.p_value >= 0.05 else "  ✗ poor fit")
        )


@dataclass
class FitReport:
    descriptive: DescriptiveStats
    fits: list[FitResult] = field(default_factory=list)

    def __str__(self) -> str:
        sep = "─" * 63
        lines = [sep, str(self.descriptive), sep, "── Distribution Fits ────────────────────────────────────────"]
        for fr in self.fits:
            lines.append(str(fr))
        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core fitting logic
# ---------------------------------------------------------------------------

def _compute_descriptive(data: np.ndarray) -> DescriptiveStats:
    return DescriptiveStats(
        n=len(data),
        mean=float(np.mean(data)),
        std=float(np.std(data, ddof=1)),
        variance=float(np.var(data, ddof=1)),
        median=float(np.median(data)),
        skewness=float(stats.skew(data, bias=False)),
        kurtosis=float(stats.kurtosis(data, bias=False)),  # excess
        minimum=float(np.min(data)),
        maximum=float(np.max(data)),
    )


def _fit_normal(data: np.ndarray) -> FitResult:
    mu, sigma = stats.norm.fit(data)
    ks, p = stats.kstest(data, "norm", args=(mu, sigma))
    return FitResult(
        distribution="normal",
        params={"mu": mu, "sigma": sigma},
        ks_statistic=ks,
        p_value=p,
        success=True,
    )


def _fit_uniform(data: np.ndarray) -> FitResult:
    loc, scale = stats.uniform.fit(data)
    ks, p = stats.kstest(data, "uniform", args=(loc, scale))
    return FitResult(
        distribution="uniform",
        params={"a (loc)": loc, "b (loc+scale)": loc + scale},
        ks_statistic=ks,
        p_value=p,
        success=True,
    )


def _fit_exponential(data: np.ndarray) -> FitResult:
    # scipy's expon: CDF = 1 - exp(-(x-loc)/scale), where scale = 1/lambda
    loc, scale = stats.expon.fit(data, floc=np.min(data))
    lam = 1.0 / scale
    ks, p = stats.kstest(data, "expon", args=(loc, scale))
    return FitResult(
        distribution="exponential",
        params={"loc": loc, "scale (1/λ)": scale, "λ": lam},
        ks_statistic=ks,
        p_value=p,
        success=True,
    )


def _fit_triangular(data: np.ndarray) -> FitResult:
    # scipy's triang: shape c ∈ (0,1), mode = loc + c*scale
    try:
        c, loc, scale = stats.triang.fit(data)
        mode = loc + c * scale
        ks, p = stats.kstest(data, "triang", args=(c, loc, scale))
        return FitResult(
            distribution="triangular",
            params={"a (min)": loc, "b (max)": loc + scale, "c (mode)": mode},
            ks_statistic=ks,
            p_value=p,
            success=True,
        )
    except Exception as exc:
        return FitResult(
            distribution="triangular",
            params={},
            ks_statistic=float("nan"),
            p_value=float("nan"),
            success=False,
            error=str(exc),
        )


def _fit_lognormal(data: np.ndarray) -> FitResult:
    if np.any(data <= 0):
        return FitResult(
            distribution="lognormal",
            params={},
            ks_statistic=float("nan"),
            p_value=float("nan"),
            success=False,
            error="lognormal requires strictly positive data",
        )
    try:
        s, loc, scale = stats.lognorm.fit(data, floc=0)
        mu_ln = np.log(scale)   # mean of underlying normal
        sigma_ln = s            # std of underlying normal
        ks, p = stats.kstest(data, "lognorm", args=(s, loc, scale))
        return FitResult(
            distribution="lognormal",
            params={"mu_ln": mu_ln, "sigma_ln": sigma_ln, "loc": loc},
            ks_statistic=ks,
            p_value=p,
            success=True,
        )
    except Exception as exc:
        return FitResult(
            distribution="lognormal",
            params={},
            ks_statistic=float("nan"),
            p_value=float("nan"),
            success=False,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_distributions(
    data: list[float] | np.ndarray,
    *,
    verbose: bool = True,
) -> FitReport:
    """
    Fit five parametric distributions to *data* and return a FitReport.

    Parameters
    ----------
    data : array-like of floats
        The observations to fit (must contain at least 3 values).
    verbose : bool, default True
        If True, print the formatted report to stdout.

    Returns
    -------
    FitReport
        Contains DescriptiveStats and a list of FitResult objects.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError("data must be a 1-D array-like.")
    if len(arr) < 3:
        raise ValueError("At least 3 observations are required.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("data contains NaN or infinite values.")

    descriptive = _compute_descriptive(arr)

    fitters = [
        _fit_normal,
        _fit_uniform,
        _fit_exponential,
        _fit_triangular,
        _fit_lognormal,
    ]

    fits: list[FitResult] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fitter in fitters:
            fits.append(fitter(arr))

    report = FitReport(descriptive=descriptive, fits=fits)

    if verbose:
        print(report)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        description=(
            "Fit numerical observations from a CSV file to five parametric "
            "distributions and report descriptive statistics and GoF results."
        )
    )
    p.add_argument(
        "csv_file",
        metavar="CSV_FILE",
        nargs="?",
        help="Path to the input CSV file. Omit to run the built-in demo.",
    )
    p.add_argument(
        "--column", "-c",
        default=None,
        metavar="NAME_OR_INDEX",
        help=(
            "Column to read. Accepts a header name (str) or a 0-based index "
            "(int). Required when the CSV has more than one column."
        ),
    )
    p.add_argument(
        "--delimiter", "-d",
        default=",",
        metavar="CHAR",
        help="Field delimiter (default: comma).",
    )
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.csv_file is None:
        # Built-in demo
        rng = np.random.default_rng(42)
        values = rng.lognormal(mean=1.5, sigma=0.4, size=200).tolist()
        print("No CSV file supplied - running demo with 200 lognormal samples.\n")
        fit_distributions(values)
    else:
        # Coerce --column to int if it looks like a digit
        col = args.column
        if col is not None and col.lstrip("-").isdigit():
            col = int(col)

        print(f"Reading: {args.csv_file}\n")
        data = load_csv(args.csv_file, column=col, delimiter=args.delimiter)
        fit_distributions(data)
