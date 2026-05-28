import requests
import os
import pandas as pd

# California Central Valley wells (heavily stressed) + 1 contrast well
WELLS = {
    "tulare_ca":     "343453119300801",
    "kings_ca":      "362401119454701",
    "fresno_ca":     "354501119464001",
    "kern_ca":       "351500119181001",
    "vermont_vt":    "444158072440001",  # wetter region for contrast
}

BASE_URL = "https://waterservices.usgs.gov/nwis/gwlevels/"


def fetch_well(site_no):
    params = {
        "format": "json",
        "sites": site_no,
        "startDT": "2000-01-01",
        "endDT": "2024-12-31",
        "parameterCd": "72019",
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_timeseries(data, name):
    ts_list = data.get("value", {}).get("timeSeries", [])
    if not ts_list:
        print(f"  No timeseries returned for {name}")
        return None

    ts = ts_list[0]
    values = ts["values"][0]["value"]
    if not values:
        print(f"  Empty values for {name}")
        return None

    records = []
    for v in values:
        records.append({
            "datetime": v["dateTime"],
            "depth_ft": v["value"],
            "qualifier": v.get("qualifiers", [""])[0] if v.get("qualifiers") else "",
        })

    df = pd.DataFrame(records)
    df["well"] = name
    df["site_no"] = ts["sourceInfo"]["siteCode"][0]["value"]
    return df


def main():
    os.makedirs("data/raw", exist_ok=True)
    fetched = []

    for name, site_no in WELLS.items():
        print(f"Fetching {name} (site {site_no})...")
        try:
            data = fetch_well(site_no)
            df = parse_timeseries(data, name)
            if df is not None and len(df) > 0:
                path = f"data/raw/{name}.csv"
                df.to_csv(path, index=False)
                print(f"  Saved {len(df)} records → {path}")
                fetched.append(name)
            else:
                print(f"  No data for {name}, skipping")
        except Exception as e:
            print(f"  Error fetching {name}: {e}")

    print(f"\nDone. Fetched {len(fetched)} wells: {fetched}")


if __name__ == "__main__":
    main()
