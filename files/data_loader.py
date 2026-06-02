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

    # ── 12-month rolling sums → SPI-12 / ETI-12 ──────────────────────────────
    # Multi-year droughts (e.g., California 2012–2017) are by definition
    # cumulative water deficits that build over many months. SPI-3 captures
    # seasonal anomalies but misses the multi-year buildup; SPI-12 is the
    # standard index for long-term drought used by the U.S. Drought Monitor.
    # We z-score by calendar month so each month's anomaly is comparable.
    WIN = 12
    df["precip_win"] = df["precip_mm"].rolling(WIN, min_periods=WIN).sum()
    df["et_win"]     = df["et_mm"].rolling(WIN, min_periods=WIN).sum()
    df = df.dropna(subset=["precip_win", "et_win"])    # drops first WIN-1 months

    clim_mean = df[["precip_win", "et_win"]].groupby(df.index.month).mean()
    clim_std  = df[["precip_win", "et_win"]].groupby(df.index.month).std()

    months = df.index.month
    df["precip_anom"] = (
        (df["precip_win"].values - clim_mean.loc[months, "precip_win"].values)
        / clim_std.loc[months, "precip_win"].values
    )
    df["et_anom"] = (
        (df["et_win"].values - clim_mean.loc[months, "et_win"].values)
        / clim_std.loc[months, "et_win"].values
    )
    df = df.dropna(subset=["precip_anom", "et_anom"])

    X     = df[["precip_anom", "et_anom"]].values.astype(float)
    dates = df.index
    return X, dates


def load_spatial_percentile(precip_path, window=12):
    """
    Compute percentile-rank drought severity per GPM grid cell, per month.

    For each cell, take the 12-month rolling precip total, then rank each
    month against all other Januaries (or Februaries…) in the 21-year record.
    The output is a percentile in [0, 1] where 0 = driest such month on
    record and 1 = wettest.

    This is the same approach the U.S. Drought Monitor uses to assign
    categories, and is distribution-free — it doesn't depend on a Gaussian
    assumption or get deflated when the baseline includes the drought itself.

    USDM category mapping (driest p% on record):
      p < 0.02:  D4 Exceptional Drought
      p < 0.05:  D3 Extreme
      p < 0.10:  D2 Severe
      p < 0.20:  D1 Moderate
      p < 0.30:  D0 Abnormally Dry
      0.30–0.70: Normal
      p > 0.70:  Wet → Exceptionally Wet

    Returns
    -------
    lat   : (Nlat,) latitudes
    lon   : (Nlon,) longitudes
    dates : (T,) pandas DatetimeIndex
    pct   : (T, Nlat, Nlon) float32  percentile in [0, 1]
    """
    gpm = xr.open_dataset(precip_path)
    precip = gpm["precipitation"]

    # The source data has NaN cells outside the SJV — those should stay NaN
    # through the percentile calculation so downstream code can render them
    # as background (white) and draw an outline only around real SJV cells.
    sjv_mask = ~np.all(np.isnan(precip.values), axis=0)         # (lat, lon)

    monthly = precip.resample(time="MS").sum()
    rolling = monthly.rolling(time=window, min_periods=window).sum()
    rolling = rolling.dropna("time", how="all")

    arr  = rolling.values.copy()             # (T, lat, lon)
    arr[:, ~sjv_mask] = np.nan               # restore SJV mask after the resample
    dts  = pd.DatetimeIndex(rolling["time"].values)
    months = dts.month.values

    pct = np.full_like(arr, np.nan, dtype="float32")
    sjv_cells = np.argwhere(sjv_mask)
    for m in range(1, 13):
        idx_m = np.where(months == m)[0]
        if len(idx_m) < 2:
            continue
        sub = arr[idx_m]
        Nyears = sub.shape[0]
        for i, j in sjv_cells:
            vals = sub[:, i, j]
            ranks = vals.argsort().argsort()
            pct[idx_m, i, j] = (ranks + 0.5) / Nyears

    return rolling["lat"].values, rolling["lon"].values, dts, pct


def load_spatial_spi(precip_path, window=12):
    """
    Compute SPI-`window` (z-scored 12-month rolling precipitation anomaly)
    at every GPM grid cell over the SJV.

    Returns
    -------
    lat   : (Nlat,) array of latitudes (cell centres, south→north)
    lon   : (Nlon,) array of longitudes (cell centres, west→east)
    dates : (T,) pandas DatetimeIndex of valid month-starts
    spi   : (T, Nlat, Nlon) float32 array — SPI z-score per cell per month
    """
    gpm = xr.open_dataset(precip_path)
    precip = gpm["precipitation"]           # (time, lat, lon), daily mm

    # Aggregate daily → monthly totals per cell
    monthly = precip.resample(time="MS").sum()
    # 12-month rolling sum at each cell
    rolling = monthly.rolling(time=window, min_periods=window).sum()

    # Per-cell climatology (mean and std by calendar month) — vectorised
    by_month = rolling.groupby("time.month")
    clim_mean = by_month.mean("time")        # (12, lat, lon)
    clim_std  = by_month.std("time")         # (12, lat, lon)

    # Avoid division by zero for cells with degenerate climatology
    clim_std_safe = clim_std.where(clim_std > 1.0, 1.0)

    months = rolling["time"].dt.month
    mean_t = clim_mean.sel(month=months)
    std_t  = clim_std_safe.sel(month=months)
    spi = ((rolling - mean_t) / std_t).astype("float32")

    # Drop the leading NaN window
    valid = ~spi.isnull().all(dim=("lat", "lon"))
    spi   = spi.where(valid, drop=True)

    return (
        spi["lat"].values,
        spi["lon"].values,
        pd.DatetimeIndex(spi["time"].values),
        spi.values,
    )


def load_raw_monthly(precip_path, et_path):
    """
    Return raw monthly totals (no z-scoring) for the SJV mean time series.

    Returns
    -------
    precip_monthly : pd.Series  mm/month, index = month-start DatetimeIndex
    et_monthly     : pd.Series  mm/month, same index
    dates          : pd.DatetimeIndex
    """
    gpm   = xr.open_dataset(precip_path)
    et_ds = xr.open_dataset(et_path)

    precip_daily = gpm["precip_mean"].to_series()
    precip_daily.index = pd.DatetimeIndex(precip_daily.index).normalize()
    precip_monthly = precip_daily.resample("MS").sum().rename("precip_mm")

    et_monthly = et_ds["et_mean"].to_series().rename("et_mm")
    et_monthly.index = (
        pd.DatetimeIndex(et_monthly.index)
        .to_period("M").to_timestamp("M") - pd.offsets.MonthBegin(1)
    )

    df = pd.DataFrame({"precip_mm": precip_monthly, "et_mm": et_monthly})
    df = df[(df.index.year >= 2000) & (df.index.year <= 2020)].dropna()
    return df["precip_mm"], df["et_mm"], df.index


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
