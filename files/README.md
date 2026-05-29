# Data

## Expected Format

A CSV file with columns:
```
date,precip_mm,et_mm
2000-01-01,3.2,1.1
2000-01-02,0.0,1.4
...
```

One row per day, covering 2000-01-01 through 2020-12-31 (7670 rows).
Values are daily averages across the San Joaquin Valley bounding box
(lat: 35.5–38.0 N, lon: -121.5–118.0 W).

---

## Data Sources

### Precipitation — PRISM
- URL: https://prism.oregonstate.edu/
- Product: AN81d (4km gridded daily precipitation)
- Variable: ppt (mm/day)
- Download via the PRISM Explorer or the `prism` R package

### ET — SSEBop (recommended)
- URL: https://earlywarning.usgs.gov/ssebop/modis/daily
- Product: SSEBop actual ET (mm/day)
- OR: MODIS MOD16A2 (8-day ET, needs temporal interpolation)

### Alternative: GRIDMET (easiest, everything in one place)
- URL: https://www.climatologylab.org/gridmet.html
- Variables: pr (precip), pet (reference ET)
- Accessible via Google Earth Engine or the `climateR` R package

---

## Quick Download via Python (GRIDMET/OpenDAP)

```python
# Install: pip install xarray netcdf4 requests

import xarray as xr
import numpy as np
import pandas as pd

# SJV bounding box
lat_min, lat_max = 35.5, 38.0
lon_min, lon_max = -121.5, -118.0

# GRIDMET via Thredds (example for precip)
url = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_pr_1979_CurrentYear_CONUS.nc"
ds = xr.open_dataset(url)

# Subset to SJV and 2000-2020
sjv_precip = ds["precipitation_amount"].sel(
    lat=slice(lat_min, lat_max),
    lon=slice(lon_min, lon_max),
    day=slice("2000-01-01", "2020-12-31")
).mean(dim=["lat", "lon"])

df_precip = sjv_precip.to_dataframe().reset_index()[["day", "precipitation_amount"]]
df_precip.columns = ["date", "precip_mm"]

# Repeat for ET, then merge:
# df = pd.merge(df_precip, df_et, on="date")
# df.to_csv("data/sjv_daily.csv", index=False)
```

---

## Prototype / Testing

Until you have real data, use the synthetic generator:
```python
from src.data_loader import make_synthetic_data
X, true_states, true_params = make_synthetic_data()
```
This generates 7670 days of synthetic HMM data from known parameters,
so you can verify your Baum-Welch recovers the true values.
