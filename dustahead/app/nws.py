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


def parse(feature_collection: dict) -> list[dict]:
    """GeoJSON from api.weather.gov -> the bundle's nws list (polygon kept when the product has one)."""
    out = []
    for f in feature_collection.get("features", []):
        p = f.get("properties", {})
        if not _keep(p):
            continue
        vtec = (p.get("parameters", {}).get("VTEC") or [""])[0]
        geom = f.get("geometry")
        out.append(dict(type=p.get("event"), vtec=vtec, area=p.get("areaDesc", ""),
                        onset=p.get("onset") or p.get("effective"), expires=p.get("ends") or p.get("expires"),
                        polygon=geom["coordinates"][0] if geom and geom.get("type") == "Polygon" else None))
    return out


def fetch_active(timeout: float = 10.0) -> list[dict] | None:
    import requests
    try:
        r = requests.get(URL, params={"area": "AZ,NM,TX", "status": "actual"}, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return parse(r.json())
    except Exception as e:  # noqa: BLE001  - the bundle still goes out, marked "warnings unknown"
        print(f"[nws] active alerts unavailable ({type(e).__name__}): bundle nws = null")
        return None
