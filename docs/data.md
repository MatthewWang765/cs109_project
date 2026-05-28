# data.md — USGS Data Source and Well Selection

---

## Data Source

**USGS National Groundwater Monitoring Network**
- URL: https://waterdata.usgs.gov
- API: https://waterservices.usgs.gov/nwis/gwlevels/
- Parameter code for depth to water: `72019`
- Format: JSON or RDB (tab-separated)

---

## API Call Structure

```
GET https://waterservices.usgs.gov/nwis/gwlevels/
  ?format=json
  &sites={site_no}
  &startDT=2000-01-01
  &parameterCd=72019
```

Returns a JSON timeseries of depth-to-water readings in feet below land surface.
Positive values = deeper water = worse.

---

## Well Selection Strategy

Pick 3–5 wells across a range of stress levels for contrast in the demo:

### Recommended: California Central Valley (heavily stressed)
This is the most depleted major aquifer in the US — dramatic signals.

Browse wells at:
https://waterdata.usgs.gov/ca/nwis/gwlevels

Good candidate site numbers to try (verify they have long records):
- 343453119300801 (Tulare County area)
- Search by county: Kings, Fresno, Tulare, Kern

### Also Consider
- High Plains Aquifer (Kansas, Texas) — another heavily stressed system
- Compare against a healthy well in a wetter region for contrast

### What to Look For in a Good Well
- At least 10 years of monthly (or more frequent) readings
- Clear trend (not just noise)
- Known casing depth (X_fail) — look in well construction metadata
- Minimal large gaps in the record

---

## Data Latency Reality

- Real-time wells: small subset, daily readings, provisional/noisy
- Most wells: monthly or irregular (manual measurement)
- Consequence: in a one-week project window, you get 0–1 new real readings

**Solution:** Train on 2000–2021, replay 2022–2024 as sequential "new data"
for the Bayesian updating demo. This is standard time series methodology.

---

## Data Cleaning Steps (2_clean_data.py)

1. Parse datetime column
2. Convert depth values to float, coerce errors to NaN
3. Drop provisional or flagged readings if needed
4. Resample to monthly frequency (take mean if multiple readings per month)
5. Forward-fill gaps of 1–2 months (small gaps)
6. Drop wells with gaps > 3 months or < 10 years of data
7. Compute Δt = X_t - X_{t-1} column

---

## Failure Threshold (X_fail)

X_fail = the depth at which the well pump intake is no longer submerged.
This is a physical property of each well.

Options for getting X_fail:
1. USGS well construction records (sometimes in the metadata)
2. Use a fixed offset below the historical maximum depth observed
3. For demo purposes: set X_fail = historical_max_depth + some buffer

Be explicit in the writeup about how X_fail is defined for each well.
