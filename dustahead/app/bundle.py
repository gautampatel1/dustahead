"""Bundle writer: one forecast -> the small set of files the web app and the phone app read.

Terms used here: P = probability of dust at a grid cell; lead = minutes after the forecast time t0;
ASOS = Automated Surface Observing System (airport weather stations); NWS = National Weather Service;
DS.W = NWS Dust Storm Warning; MP = milepost; R2 = Cloudflare's S3-compatible object store; FCM = Firebase
Cloud Messaging (the push topics are named after the segment ids written here).

Layout of one bundle (relative to the publish root, which is what gets uploaded):

    latest.json                       the pointer both apps poll; rewritten last, so it never points at missing files
    runs/YYYYMMDDHHMM/bundle.json     the same JSON, kept for replay and for the record
    runs/YYYYMMDDHHMM/p15.png ...     one RGBA map per lead, already in Web Mercator rows, north up

Why PNG maps and not a tile server: the domain is 250 x 200 cells (~4 km), so a whole lead is a ~20 kB image that
MapLibre drapes with an `image` source. Grid row 0 is the SOUTH edge (dustahead/grid.py), so rows are flipped, and
because the map is drawn in Web Mercator the rows are resampled to equal Mercator spacing; without that the dust
would sit up to ~5 km south of where the model put it at the northern edge of the domain.

The road state of every segment is decided here, from the same `road_alerts` + `AlertTracker` output the dashboard
shows. The apps only draw it; they never re-threshold P themselves, so the web map, the phone and the pushes agree.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from .predict import I10, I10_EXITS, I10_NAMES

SCHEMA = 1
# Milepost of each waypoint = its I-10 exit number (predict.I10_EXITS, from OpenStreetMap).
I10_MP = [(state, exit_no) for _, state, exit_no, _, _ in I10_EXITS]
assert len(I10_MP) == len(I10) == len(I10_NAMES)

# Colour ramp for the P layer on the dusk basemap: (P, hex, alpha). Below the first stop the cell is transparent,
# so faint noise never tints the map. The road colours (watch / severe) are deliberately NOT on this ramp: the
# raster says "how likely", the road line says "what to do".
RAMP = [(0.20, "#f0c069", 0.0), (0.40, "#f0c069", 0.38), (0.60, "#d98a3a", 0.62), (0.80, "#b5502e", 0.80),
        (1.00, "#8f3320", 0.90)]


def segment_id(k: int) -> str:
    """Stable id of the segment between waypoints k and k+1, e.g. 'i10-az-208-219' or 'i10-az378-nm22'.
    Also the FCM topic name ('seg-' is not prefixed here; topic names allow letters, digits and '-')."""
    (s0, m0), (s1, m1) = I10_MP[k], I10_MP[k + 1]
    return f"i10-{s0.lower()}-{m0}-{m1}" if s0 == s1 else f"i10-{s0.lower()}{m0}-{s1.lower()}{m1}"


def segment_table() -> list[dict]:
    """Static description of every segment, in the order of DustAhead.road_cells()."""
    segs = []
    for k in range(len(I10) - 1):
        (s0, m0), (s1, m1) = I10_MP[k], I10_MP[k + 1]
        segs.append(dict(id=segment_id(k), name=f"{I10_NAMES[k]} → {I10_NAMES[k + 1]}", from_=I10_NAMES[k],
                         to=I10_NAMES[k + 1], state=s0 if s0 == s1 else f"{s0}/{s1}", mp=[m0, m1],
                         line=[list(I10[k]), list(I10[k + 1])]))
    return segs


# ---------------------------------------------------------------------------------------------------- the P maps
def _hex(c: str) -> tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def colorize(p: np.ndarray) -> np.ndarray:
    """(ny, nx) P in [0, 1] -> (ny, nx, 4) uint8 RGBA along RAMP (linear between stops)."""
    p = np.clip(np.nan_to_num(p.astype(np.float32), nan=0.0), 0, 1)
    xs = [s[0] for s in RAMP]
    rgba = np.zeros(p.shape + (4,), np.float32)
    for ch in range(3):
        rgba[..., ch] = np.interp(p, xs, [_hex(s[1])[ch] for s in RAMP])
    rgba[..., 3] = np.interp(p, xs, [s[2] * 255 for s in RAMP])
    rgba[p < xs[0]] = 0
    return rgba.round().astype(np.uint8)


def _merc_y(lat_deg):
    return np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))


def mercator_rows(grid, height: int | None = None) -> np.ndarray:
    """Source row (south-up grid index) for each output image row (north-up, equal Web Mercator spacing)."""
    h = height or grid.ny
    y0, y1 = _merc_y(grid.lat_min), _merc_y(grid.lat_max)
    y = y1 - (np.arange(h) + 0.5) / h * (y1 - y0)                  # output row 0 = north edge
    lat = np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)
    return np.clip(np.floor((lat - grid.lat_min) / grid.dlat).astype(int), 0, grid.ny - 1)


def prob_png(p: np.ndarray, grid, path: str | Path) -> Path:
    """Write one lead's P map as a north-up, Web-Mercator-row RGBA PNG."""
    from PIL import Image
    img = colorize(p)[mercator_rows(grid)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, "RGBA").save(path, optimize=True)
    return path


# ---------------------------------------------------------------------------------------------------- the JSON
def _iso(t) -> str:
    """Any datetime-like (naive = UTC, as everywhere in this repo) -> '2026-08-25T23:10:00Z'."""
    if isinstance(t, np.ndarray):                  # stored samples keep t0 as a 0-d datetime64 array
        t = t[()]
    if isinstance(t, np.datetime64):
        t = t.astype("datetime64[s]").astype(dt.datetime)
    if isinstance(t, str):
        t = dt.datetime.fromisoformat(t.replace("Z", "+00:00"))
    if t.tzinfo is not None:
        t = t.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return t.replace(microsecond=0).isoformat() + "Z"


def stations_json(st_ij, st_now, grid, names: dict | None = None) -> list[dict]:
    """Stations with a report in the window (+1 dust, -1 clear); 'no report' stations are left out.
    `names` maps (i, j) -> ASOS id when the station table is available (live mode, or labels/stations.csv)."""
    if st_ij is None or st_now is None:
        return []
    lons, lats = grid.lons(), grid.lats()
    out = []
    for (i, j), n in zip(np.asarray(st_ij), np.asarray(st_now)):
        if n == 0:
            continue
        sid = (names or {}).get((int(i), int(j)))
        out.append(dict(id=sid or f"cell-{int(i)}-{int(j)}", lon=round(float(lons[j]), 3), lat=round(float(lats[i]), 3),
                        dust=bool(n == 1)))
    return out


def tiers_json(P: dict) -> dict | None:
    """The two alert tiers (product.tiers) as the apps need them, or None when tiers are off.
    `rated` is True only when the product model is the one the hit rates were measured for (as in road_alerts);
    otherwise the apps must not show the rates."""
    T = P.get("tiers") or {}
    if not T.get("enabled"):
        return None
    ck = [str(c).replace("\\", "/") for c in P.get("checkpoints", [])]
    rated = ck == [str(c) for c in T.get("validated_for", [])]
    out = dict(rated=rated)
    for k in ("nearby", "new_storm"):
        t = T[k]
        out[k] = dict(thresholds=list(t["thresholds"]), hit_rate=t.get("hit_rate") if rated else None,
                      coverage=t.get("coverage") if rated else None)
    return out


def build_bundle(fc: dict, engine, *, source: str, issued=None, nws: list | None = None,
                 station_names: dict | None = None, layers: dict | None = None) -> dict:
    """The latest.json document for one forecast.

    fc       output of DustAhead.predict()/predict_array(): probs (L, ny, nx), genesis map, alerts (already passed
             through AlertTracker when the caller has one), t0, station_ij/station_now.
    source   'live' | 'replay' | 'fixture'. The apps treat only 'live' bundles as real-time; the others show a badge.
    nws      active NWS dust products, [] for none, None when the feed could not be read (the apps say "unknown").
    """
    cfg = engine.cfg
    P = cfg.get("product", {})
    probs = np.asarray(fc["probs"], np.float32)
    gmap = fc.get("genesis")
    table = segment_table()
    name_to_id = {}
    segs = []
    for s, (name, i, j) in zip(table, engine.road_cells()):
        name_to_id[name] = s["id"]
        seg = {k.rstrip("_"): v for k, v in s.items()}
        seg["p"] = [round(float(probs[l][i, j].max()), 3) for l in range(len(engine.leads))]
        seg["genesis"] = bool(gmap[i, j].all()) if gmap is not None else False
        segs.append(seg)
    alerts = []
    for a in fc.get("alerts", []):
        sid = name_to_id.get(a["segment"])
        if sid is None:
            continue
        alerts.append(dict(segment=sid, name=a["segment"].removeprefix("I-10 "), lead_min=int(a["lead_min"]),
                           p_max=round(float(a["p_max"]), 3), level=a["level"], genesis=bool(a.get("genesis")),
                           held=bool(a.get("held", False)), tier=a.get("tier"),
                           hit_rate=(round(float(a["hit_rate"]), 3) if a.get("hit_rate") is not None else None)))
    order = {s["id"]: k for k, s in enumerate(segs)}
    alerts.sort(key=lambda a: order[a["segment"]])
    stations = stations_json(fc.get("station_ij"), fc.get("station_now"), engine.grid, station_names)
    g = engine.grid
    names = getattr(engine, "member_names", []) or []
    return dict(
        schema=SCHEMA,
        t0=_iso(fc["t0"]),
        issued=_iso(issued or dt.datetime.now(dt.timezone.utc)),
        source=source,
        model=(f"max({', '.join(names)})" if len(names) > 1 else (names[0] if names else source)),
        leads=list(engine.leads),
        bounds=[g.lon_min, g.lat_min, g.lon_max, g.lat_max],
        layers=layers or {},
        policy=dict(thresholds=P.get("thresholds"), severe_margin=P.get("severe_margin", 0.1),
                    release=P.get("release_threshold", 0.5), min_hold_min=10 * int(P.get("min_hold_steps", 3)),
                    stale_after_min=int(cfg.get("publish", {}).get("stale_after_min", 25)),
                    genesis_km=50, tiers=tiers_json(P)),
        segments=segs,
        alerts=alerts,
        stations=stations,
        dust_now=[s["id"] for s in stations if s["dust"]],
        nws=nws,
    )


def write_bundle(fc: dict, engine, root: str | Path, *, source: str, issued=None, nws=None,
                 station_names: dict | None = None, update_latest: bool = True) -> tuple[Path, list[Path]]:
    """Write runs/<t0>/{p*.png, bundle.json} and then latest.json under `root`.
    Returns (bundle.json path, every file written in upload order: maps first, pointer last)."""
    root = Path(root)
    t0 = _iso(fc["t0"])
    stamp = t0[:16].replace("-", "").replace(":", "").replace("T", "")
    run = Path("runs") / stamp
    written, layers = [], {}
    for l, lead in enumerate(engine.leads):
        rel = run / f"p{lead}.png"
        written.append(prob_png(np.asarray(fc["probs"])[l], engine.grid, root / rel))
        layers[str(lead)] = rel.as_posix()
    doc = build_bundle(fc, engine, source=source, issued=issued, nws=nws, station_names=station_names, layers=layers)
    text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    bpath = root / run / "bundle.json"
    bpath.write_text(text, encoding="utf-8")
    written.append(bpath)
    if update_latest:
        tmp = root / "latest.json.tmp"
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(root / "latest.json")          # atomic on one filesystem: a reader never sees half a file
        written.append(root / "latest.json")
    return bpath, written


def write_replay_index(root: str | Path, bundles: list[Path], *, title: str, source: str) -> Path:
    """replay.json: the ordered list of bundle.json files the web app's replay mode steps through."""
    root = Path(root)
    frames = [Path(b).relative_to(root).as_posix() for b in bundles]
    path = root / "replay.json"
    path.write_text(json.dumps(dict(schema=SCHEMA, title=title, source=source, frames=frames), ensure_ascii=False,
                               indent=1), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------------- upload
def upload_r2(files: list[Path], root: str | Path, env=None) -> int:
    """Upload `files` (paths under `root`) to Cloudflare R2 in the given order. Credentials come from the
    environment only: R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET. Returns files uploaded.
    latest.json is sent with a 30 s cache so a phone never sits on an old pointer; run files never change."""
    import os

    import boto3
    env = env or os.environ
    s3 = boto3.client("s3", endpoint_url=env["R2_ENDPOINT"], aws_access_key_id=env["R2_ACCESS_KEY_ID"],
                      aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"], region_name="auto")
    root = Path(root)
    for f in files:
        key = Path(f).relative_to(root).as_posix()
        ctype = "application/json" if key.endswith(".json") else "image/png"
        cache = "public, max-age=30" if key in ("latest.json", "replay.json") else "public, max-age=31536000, immutable"
        s3.upload_file(str(f), env["R2_BUCKET"], key, ExtraArgs={"ContentType": ctype, "CacheControl": cache})
    return len(files)

