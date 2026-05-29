"""
data_loader.py — Load and preprocess SJV precipitation and ET data.

Expected data sources:
  - Precipitation: PRISM gridded daily precipitation (mm/day)
      URL: https://prism.oregonstate.edu/
      OR:  GRIDMET via Google Earth Engine
  - ET: SSEBop actual ET (mm/day) or MODIS ET
      URL: https://earlywarning.usgs.gov/ssebop/

For the prototype, a CSV with columns [date, precip_mm, et_mm] works fine.
One row = one day, averaged across the SJV bounding box.

SJV approximate bounding box:
  lat: 35.5 N to 38.0 N
  lon: -121.5 W to -118.0 W
"""

import numpy as np
import pandas as pd
import xarray as xr


def load_netcdf(precip_path, et_path):
    """
    Load monthly SJV observations as anomalies from the seasonal climatology.

    Working at monthly resolution (252 months, 2000-01 to 2020-12) removes the
    problem of fitting a seasonal sine wave: the model sees departures from the
    long-term monthly mean, so it can distinguish drought years from wet years
    rather than just "summer vs winter".

    Features returned:
      precip_anom  — monthly precip total minus that calendar-month's 21-yr mean (mm)
      et_anom      — monthly ET minus that calendar-month's 21-yr mean (mm)

    Returns
    -------
    X     : (252, 2) float array
    dates : pandas DatetimeIndex (monthly, period start)
    """
    gpm   = xr.open_dataset(precip_path)
    et_ds = xr.open_dataset(et_path)

    # ── GPM: aggregate daily precip to monthly totals ──────────────────────
    precip_daily = gpm["precip_mean"].to_series()
    precip_daily.index = pd.DatetimeIndex(precip_daily.index).normalize()
    precip_monthly = (
        precip_daily
        .resample("MS")           # month-start
        .sum()
        .rename("precip_mm")
    )

    # ── OpenET: already monthly, align index to month-start ────────────────
    et_monthly = et_ds["et_mean"].to_series().rename("et_mm")
    et_monthly.index = (
        pd.DatetimeIndex(et_monthly.index)
        .to_period("M").to_timestamp("M") - pd.offsets.MonthBegin(1)
    )

    df = pd.DataFrame({"precip_mm": precip_monthly, "et_mm": et_monthly})
    df = df[(df.index.year >= 2000) & (df.index.year <= 2020)].dropna()

    # ── Remove seasonal climatology → anomalies ─────────────────────────────
    # Climatological mean for each calendar month (Jan=1 … Dec=12)
    clim = df.groupby(df.index.month).mean()          # shape (12, 2)
    df["precip_anom"] = df["precip_mm"] - df.index.map(lambda d: clim.loc[d.month, "precip_mm"])
    df["et_anom"]     = df["et_mm"]     - df.index.map(lambda d: clim.loc[d.month, "et_mm"])

    X     = df[["precip_anom", "et_anom"]].values.astype(float)
    dates = df.index
    return X, dates


def load_csv(path):
    """
    Load daily SJV observations from a CSV file.

    Expected columns: date, precip_mm, et_mm
    Returns X: (T, 2) numpy array, and dates index.

    Parameters
    ----------
    path : str, path to CSV file

    Returns
    -------
    X     : (T, 2) float array [precip, ET]
    dates : pandas DatetimeIndex
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[(df["date"].dt.year >= 2000) & (df["date"].dt.year <= 2020)]
    df = df.dropna(subset=["precip_mm", "et_mm"])

    X = df[["precip_mm", "et_mm"]].values.astype(float)
    dates = pd.DatetimeIndex(df["date"])
    return X, dates


def normalize(X):
    """
    Standardize observations to zero mean, unit variance per feature.
    Important for numerical stability of Gaussian emissions.

    Returns X_norm, mean, std (save mean/std to invert later).
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


def make_synthetic_data(T=7670, K=4, D=2, random_state=42):
    """
    Generate synthetic HMM data for testing the implementation.
    T = ~21 years of daily data (365 * 21 ≈ 7665)

    Use this to verify your Baum-Welch implementation recovers
    the true parameters before running on real data.
    """
    rng = np.random.default_rng(random_state)

    # True parameters
    pi_true = np.array([0.25, 0.25, 0.25, 0.25])
    A_true = np.array([
        [0.95, 0.03, 0.01, 0.01],
        [0.02, 0.93, 0.03, 0.02],
        [0.01, 0.02, 0.94, 0.03],
        [0.02, 0.01, 0.02, 0.95],
    ])
    # Regimes: [wet, moderate, dry, hot-dry]
    mus_true = np.array([
        [8.0, 2.0],   # wet:     high precip, low ET
        [3.0, 3.5],   # moderate
        [0.5, 4.5],   # dry:     low precip, high ET
        [0.2, 6.0],   # hot-dry: very low precip, very high ET
    ])
    sigmas_true = np.array([
        [[4.0, 0.0], [0.0, 0.5]],
        [[1.0, 0.0], [0.0, 0.5]],
        [[0.2, 0.0], [0.0, 0.8]],
        [[0.1, 0.0], [0.0, 1.0]],
    ])

    # Sample state sequence
    states = np.zeros(T, dtype=int)
    states[0] = rng.choice(K, p=pi_true)
    for t in range(1, T):
        states[t] = rng.choice(K, p=A_true[states[t-1]])

    # Sample observations
    X = np.zeros((T, D))
    for t in range(T):
        k = states[t]
        X[t] = rng.multivariate_normal(mus_true[k], sigmas_true[k])
    X = np.clip(X, 0, None)   # precip and ET are non-negative

    return X, states, {"pi": pi_true, "A": A_true, "mus": mus_true, "sigmas": sigmas_true}
