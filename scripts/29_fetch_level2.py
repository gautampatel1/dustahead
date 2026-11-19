"""Step 29: download raw NEXRAD Level II radar volumes for a few storms (standard library only).

    python -u scripts/29_fetch_level2.py            # the five 2025 check storms below
    python -u scripts/29_fetch_level2.py --radar KIWA --start 2025-08-25T22:00 --end 2025-08-26T01:30 --every 10

Why: the radar mosaic used so far (IEM N0Q) holds reflectivity only, which mostly shows rain. The raw volumes of
the WSR-88D radars (Weather Surveillance Radar, 1988 Doppler) also hold radial velocity and the dual-polarization
moments, among them the correlation coefficient (how alike the particles in the beam are). NWS forecasters use
exactly these for dust (NOAA Technical Memorandum NWS-WR 290: inbound velocities of 40-45 kt and non-uniform
correlation coefficient). This step only fetches the files (public bucket unidata-nexrad-level2 on AWS) into
data/raw/level2/<radar>/; they are decoded elsewhere because the decoders need libraries blocked on this PC.
One volume nearest to every `--every` minutes is kept.
"""
import argparse
import datetime as dt
import re
import sys
import urllib.request
from pathlib import Path

BUCKET = "https://unidata-nexrad-level2.s3.amazonaws.com"
STORMS = [  # radar, start, end (UTC), minutes between kept volumes, note
    ("KIWA", "2025-08-25T22:00", "2025-08-26T01:30", 10, "Phoenix haboob (monsoon)"),
    ("KIWA", "2025-08-22T01:30", "2025-08-22T05:00", 15, "Casa Grande - Marana (monsoon)"),
    ("KIWA", "2025-09-04T03:00", "2025-09-04T04:30", 10, "Phoenix (monsoon)"),
    ("KEPZ", "2025-03-18T17:00", "2025-03-18T20:30", 15, "El Paso - Deming (synoptic wind)"),
    ("KEPZ", "2025-04-27T20:00", "2025-04-27T23:30", 15, "El Paso (frontal)"),
]


def list_keys(radar: str, day: dt.date) -> list[str]:
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={day:%Y/%m/%d}/{radar}/" + (f"&continuation-token={urllib.request.quote(token, safe='')}" if token else "")
        xml = urllib.request.urlopen(url, timeout=60).read().decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            return [k for k in keys if not k.endswith("_MDM")]
        token = m.group(1)


def fetch(radar: str, start: dt.datetime, end: dt.datetime, every: int, root: Path) -> int:
    keys = []
    d = start.date()
    while d <= end.date():
        keys += list_keys(radar, d); d += dt.timedelta(days=1)
    times = {}
    for k in keys:
        m = re.search(rf"{radar}(\d{{8}})_(\d{{6}})", k)
        if m:
            times[dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")] = k
    want, t = [], start
    while t <= end:
        near = min(times, key=lambda q: abs((q - t).total_seconds()), default=None)
        if near is not None and abs((near - t).total_seconds()) <= every * 30 + 240 and times[near] not in want:
            want.append(times[near])
        t += dt.timedelta(minutes=every)
    out = root / radar
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for k in want:
        f = out / k.rsplit("/", 1)[1]
        if not f.exists() or f.stat().st_size == 0:
            tmp = f.with_suffix(".part")
            urllib.request.urlretrieve(f"{BUCKET}/{k}", tmp)
            tmp.replace(f)
        n += 1
    print(f"{radar} {start:%Y-%m-%d %H:%M}-{end:%H:%M}: {n} volumes of {len(times)} listed, "
          f"{sum(p.stat().st_size for p in out.iterdir()) / 1e6:.0f} MB in {out}", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radar"); ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--root", default="data/raw/level2")
    a = ap.parse_args()
    storms = [(a.radar, a.start, a.end, a.every, "")] if a.radar else STORMS
    total = 0
    for radar, s, e, every, note in storms:
        total += fetch(radar, dt.datetime.fromisoformat(s), dt.datetime.fromisoformat(e), every, Path(a.root))
    print("done", total, "volumes")
    sys.exit(0 if total else 1)


if __name__ == "__main__":
    main()
