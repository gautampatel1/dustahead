"""NCEI Storm Events Database: the second, independent cross-check for station dust labels.

Files (verified Oct 2026):
  https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d2025_c20260819.csv.gz
The creation date in the name changes with each update, so we list the directory and pick the file per year.
We keep EVENT_TYPE == "Dust Storm" in AZ/NM/TX/CA with begin/end times converted to UTC.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
import requests

NCEI = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
TZ_OFFSET = {"MST-7": -7, "MST": -7, "CST-6": -6, "CST": -6, "PST-8": -8, "PST": -8, "EST-5": -5, "EST": -5}


def list_detail_files() -> dict[int, str]:
    html = requests.get(NCEI, timeout=120).text
    out = {}
    for m in re.finditer(r"(StormEvents_details-ftp_v1\.0_d(\d{4})_c\d{8}\.csv\.gz)", html):
        out[int(m.group(2))] = NCEI + m.group(1)
    return out


def fetch_dust_storm_events(year: int, states=("ARIZONA", "NEW MEXICO", "TEXAS", "CALIFORNIA"),
                            cache_dir: str | Path | None = None) -> pd.DataFrame:
    url = list_detail_files()[year]
    if cache_dir:
        p = Path(cache_dir) / Path(url).name
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(requests.get(url, timeout=600).content)
        raw = p.read_bytes()
    else:
        raw = requests.get(url, timeout=600).content
    df = pd.read_csv(io.BytesIO(raw), compression="gzip", low_memory=False)
    df = df[(df.EVENT_TYPE == "Dust Storm") & (df.STATE.isin(states))].copy()
    off = df.CZ_TIMEZONE.map(TZ_OFFSET).fillna(-7)
    begin = pd.to_datetime(df.BEGIN_DATE_TIME, format="%d-%b-%y %H:%M:%S")
    end = pd.to_datetime(df.END_DATE_TIME, format="%d-%b-%y %H:%M:%S")
    df["begin"] = (begin - pd.to_timedelta(off, unit="h")).dt.tz_localize("UTC")
    df["end"] = (end - pd.to_timedelta(off, unit="h")).dt.tz_localize("UTC")
    df["cz_name"] = df.CZ_NAME.astype(str)
    cols = ["EVENT_ID", "STATE", "cz_name", "begin", "end", "BEGIN_LAT", "BEGIN_LON", "DEATHS_DIRECT", "INJURIES_DIRECT",
            "DAMAGE_PROPERTY", "EVENT_NARRATIVE"]
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)
