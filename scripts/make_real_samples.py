"""
DustAhead: build real sample images + labels from GOES-18 and ASOS.

Run on your own PC (needs internet access to AWS and Iowa Environmental Mesonet):
    pip install s3fs satpy pyspectral pyresample netCDF4 h5netcdf pandas requests pillow
    python make_real_samples.py

Output: samples/<name>.png (Dust RGB, Arizona + New Mexico) and samples/labels.csv
Not yet tested end to end - check the first image before relying on it.
"""
import os, io, datetime as dt
import pandas as pd, requests, s3fs
from satpy import Scene

# --- 1. Pick 5 sample times (UTC). Fill in dates from the NWS warning archive:
#     https://mesonet.agron.iastate.edu/vtec/search.php  (phenomena DS = Dust Storm Warning)
SAMPLES = [
    # name,                 UTC time,                       ASOS station, kind
    ("1_dust_storm_day",    dt.datetime(2025, 8, 26, 2, 0), "PHX", "dust"),   # Phoenix haboob, 25 Aug 2025 ~7 pm MST
    ("2_storm_no_dust",     None,                           "TUS", "storm, rain, no dust"),
    ("3_clear_desert",      None,                           "DMN", "clear"),
    ("4_dust_night",        None,                           "DMN", "dust at night"),
    ("5_dust_under_cloud",  None,                           "LSB", "dust below anvil"),
]
BANDS = ["C11", "C13", "C14", "C15"]          # 8.4, 10.3, 11.2, 12.3 µm -> Dust RGB
BBOX = (-115.0, 30.5, -103.0, 36.5)           # lon_min, lat_min, lon_max, lat_max (AZ + NM)
OUT = "samples"
fs = s3fs.S3FileSystem(anon=True)


def goes_files(t):
    """Nearest GOES-18 CONUS L1b file per band for time t."""
    doy = t.timetuple().tm_yday
    prefix = f"noaa-goes19/ABI-L1b-RadC/{t:%Y}/{doy:03d}/{t:%H}/"
    files = fs.ls(prefix)
    picked = []
    for b in BANDS:
        cands = [f for f in files if f"M6{b}_" in f]
        # file names carry scan start as sYYYYDDDHHMMSSs; pick the closest
        def start(f):
            s = f.split("_s")[1][:13]
            return dt.datetime.strptime(s, "%Y%j%H%M%S")
        picked.append(min(cands, key=lambda f: abs((start(f) - t).total_seconds())))
    local = []
    for f in picked:
        dst = os.path.join(OUT, "raw", os.path.basename(f))
        if not os.path.exists(dst):
            fs.get(f, dst)
        local.append(dst)
    return local


def dust_png(t, name):
    scn = Scene(reader="abi_l1b", filenames=goes_files(t))
    scn.load(["dust"])
    scn = scn.crop(ll_bbox=BBOX).resample(resampler="native")
    path = os.path.join(OUT, f"{name}.png")
    scn.save_dataset("dust", filename=path)
    return path


def asos_label(station, t):
    """Nearest ASOS report within 30 min and the DustAhead station label."""
    url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
           f"?station={station}&data=vsby&data=wxcodes&data=gust&tz=Etc/UTC&format=onlycomma"
           f"&latlon=no&missing=empty&trace=empty&report_type=2&report_type=3"
           f"&sts={(t - dt.timedelta(minutes=30)):%Y-%m-%dT%H:%MZ}"
           f"&ets={(t + dt.timedelta(minutes=30)):%Y-%m-%dT%H:%MZ}")
    df = pd.read_csv(io.StringIO(requests.get(url, timeout=60).text))
    if df.empty:
        return {"vsby": None, "wxcodes": None, "label": "no report"}
    df["valid"] = pd.to_datetime(df["valid"])
    r = df.iloc[(df["valid"] - pd.Timestamp(t)).abs().argsort().iloc[0]]
    vis = pd.to_numeric(r["vsby"], errors="coerce")
    wx = str(r["wxcodes"]) if pd.notna(r["wxcodes"]) else ""
    dust = any(c in wx for c in ("DU", "DS", "SS", "BLDU", "HZ"))
    rain = any(c in wx for c in ("RA", "DZ", "SN"))
    # Warning / Storm Events cross-check is done separately (see proposal, Labels section)
    label = 1 if (vis == vis and vis <= 1.0 and dust and not rain) else 0
    return {"vsby": vis, "wxcodes": wx, "label": label}


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    rows = []
    for name, t, stn, kind in SAMPLES:
        if t is None:
            print(f"skip {name}: add a UTC time first")
            continue
        png = dust_png(t, name)
        lab = asos_label(stn, t)
        rows.append({"sample": name, "utc": t, "station": stn, "kind": kind, "image": png, **lab})
        print(name, lab)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "labels.csv"), index=False)
    print("done ->", OUT)
