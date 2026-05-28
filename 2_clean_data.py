import pandas as pd
import numpy as np
import glob
import os


def clean_well(path):
    name = os.path.basename(path).replace(".csv", "")
    df = pd.read_csv(path)

    # Parse datetime (USGS timestamps may include timezone offset)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Depth to float; drop "Ice", "Eqp", and other non-numeric sentinels
    df["depth_ft"] = pd.to_numeric(df["depth_ft"], errors="coerce")
    df = df.dropna(subset=["depth_ft"])
    df = df[df["depth_ft"] > 0]  # negative depths are instrument errors

    # Sort and resample to monthly (mean of all readings that month)
    df = df.set_index("datetime").sort_index()
    monthly = df["depth_ft"].resample("MS").mean()

    # Forward-fill gaps of 1–2 months (small seasonal gaps are common)
    monthly = monthly.ffill(limit=2)

    # Require at least 10 years (120 months) of data
    n_valid = monthly.notna().sum()
    if n_valid < 120:
        print(f"  {name}: only {n_valid} months, skipping (need ≥ 120)")
        return None

    # Check for large interior gaps (> 3 consecutive NaN after forward-fill)
    if monthly.isna().any():
        # Find longest NaN run
        is_nan = monthly.isna().astype(int)
        run_id = (is_nan != is_nan.shift()).cumsum()
        max_gap = is_nan.groupby(run_id).sum().max()
        if max_gap > 3:
            print(f"  {name}: gap > 3 months detected, skipping")
            return None

    monthly = monthly.dropna()

    result = monthly.reset_index()
    result.columns = ["date", "depth_ft"]

    # Month-over-month change Δt = X_t - X_{t-1}
    result["delta"] = result["depth_ft"].diff()

    # Train/test split: train = 2000–2021, test (replay) = 2022+
    result["split"] = "train"
    result.loc[result["date"] >= "2022-01-01", "split"] = "test"

    result["well"] = name
    return result


def main():
    raw_files = sorted(glob.glob("data/raw/*.csv"))
    if not raw_files:
        print("No raw files found. Run 1_fetch_data.py first.")
        return {}

    kept = {}
    for path in raw_files:
        name = os.path.basename(path).replace(".csv", "")
        print(f"Cleaning {name}...")
        df = clean_well(path)
        if df is not None:
            out_path = f"data/{name}_clean.csv"
            df.to_csv(out_path, index=False)
            n_train = (df["split"] == "train").sum()
            n_test = (df["split"] == "test").sum()
            print(f"  Saved → {out_path}  ({n_train} train, {n_test} test months)")
            kept[name] = df

    print(f"\nKept {len(kept)}/{len(raw_files)} wells: {list(kept.keys())}")
    return kept


if __name__ == "__main__":
    main()
