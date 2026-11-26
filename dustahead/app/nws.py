"""Active NWS (National Weather Service) dust products over the I-10 domain, for the apps' warning banner.

The apps show an official warning whatever the model says (blueprint rule "Official warnings first"), so the
worker attaches the active products to every bundle. Source: the api.weather.gov active-alerts feed, which carries
the same VTEC (Valid Time Event Code) products IEM (Iowa Environmental Mesonet) archives for training labels:
DS.W Dust Storm Warning, DU.Y Blowing Dust Advisory, and the newer Blowing Dust Warning.

Returns [] when nothing is active and None when the feed could not be read, and the two must not be confused:
None makes the apps say "NWS warnings unknown" instead of implying there are none.
"""
from __future__ import annotations

DUST_EVENTS = {"Dust Storm Warning", "Blowing Dust Warning", "Blowing Dust Advisory", "Dust Advisory"}
# Texas is in the domain only around El Paso; zone alerts elsewhere in Texas are dropped by name.
TX_COUNTIES = ("El Paso", "Hudspeth")
URL = "https://api.weather.gov/alerts/active"
HEADERS = {"User-Agent": "DustAhead-prototype (research nowcast)", "Accept": "application/geo+json"}


def _keep(props: dict) -> bool:
    if props.get("event") not in DUST_EVENTS:
        return False
    area = props.get("areaDesc", "")
    states = {c[:2] for c in props.get("geocode", {}).get("UGC", [])}       # UGC codes: AZZ540, NMC029, ...
    return bool(states & {"AZ", "NM"}) or any(c in area for c in TX_COUNTIES)


def rings(geom: dict | None, max_points: int = 150) -> list[list[list[float]]]:
    """Outer rings of a GeoJSON Polygon or MultiPolygon, rounded to 3 decimals (~100 m) and thinned to at most
    `max_points` vertices each, so a county-sized zone stays a few kB in every bundle."""
    if not geom:
        return []
    polys = [geom["coordinates"]] if geom.get("type") == "Polygon" else geom.get("coordinates", []) if geom.get("type") == "MultiPolygon" else []
    out = []
    for poly in polys:
        ring = poly[0]
        step = max(1, -(-len(ring) // max_points))       # ceil, so the cap holds
        thin = ring[::step]
        if thin[-1] != ring[-1]:
            thin.append(ring[-1])
        out.append([[round(x, 3), round(y, 3)] for x, y in thin])
    return out


def parse(feature_collection: dict) -> list[dict]:
    """GeoJSON from api.weather.gov -> the bundle's nws list. Storm-based warnings (Dust Storm Warnings) carry their
    own polygon. Zone-based products (most Blowing Dust Advisories) carry none: their `zones` URLs are kept so
    fetch_active can look the zone shapes up; until then `polygons` is []."""
    out = []
    for f in feature_collection.get("features", []):
        p = f.get("properties", {})
        if not _keep(p):
            continue
        vtec = (p.get("parameters", {}).get("VTEC") or [""])[0]
        out.append(dict(type=p.get("event"), vtec=vtec, area=p.get("areaDesc", ""),
                        onset=p.get("onset") or p.get("effective"), expires=p.get("ends") or p.get("expires"),
                        polygons=rings(f.get("geometry")), zones=list(p.get("affectedZones", []))))
    return out


def fetch_active(timeout: float = 10.0, max_zones: int = 15) -> list[dict] | None:
    """Active dust products. Zone-based ones get their zone outlines from api.weather.gov/zones (only when such a
    product is active, so normally no extra requests); a zone that can't be fetched is skipped and the apps then
    show the product by name without a shape."""
    import requests
    try:
        r = requests.get(URL, params={"area": "AZ,NM,TX", "status": "actual"}, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        out = parse(r.json())
    except Exception as e:  # noqa: BLE001  - the bundle still goes out, marked "warnings unknown"
        print(f"[nws] active alerts unavailable ({type(e).__name__}): bundle nws = null")
        return None
    cache: dict[str, list] = {}
    for w in out:
        if not w["polygons"]:
            for z in w["zones"][:max_zones]:
                if z not in cache:
                    try:
                        zr = requests.get(z, headers=HEADERS, timeout=timeout)
                        zr.raise_for_status()
                        cache[z] = rings(zr.json().get("geometry"))
                    except Exception as e:  # noqa: BLE001
                        print(f"[nws] zone shape unavailable ({type(e).__name__}): {z.rsplit('/', 1)[-1]}")
                        cache[z] = []
                w["polygons"].extend(cache[z])
        w.pop("zones", None)
    return out
