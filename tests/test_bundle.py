"""Bundle writer (dustahead/app/bundle.py): what the web and phone apps read must match the product policy."""
import json

import numpy as np
import pytest

from dustahead import load_config
from dustahead.app import bundle as B
from dustahead.app import nws
from dustahead.app.predict import AlertTracker, DustAhead


@pytest.fixture(scope="module")
def engine():
    return DustAhead.policy_only(load_config())


def test_segment_ids_are_unique_and_follow_mileposts(engine):
    segs = B.segment_table()
    ids = [s["id"] for s in segs]
    assert len(ids) == len(engine.road_cells()) == 24 and len(set(ids)) == 24
    assert "i10-az-208-219" in ids                      # Eloy -> Picacho Peak
    assert "i10-az378-nm22" in ids                      # San Simon -> Lordsburg crosses the state line
    assert all(len(i) <= 40 and i.replace("-", "").isalnum() for i in ids)    # valid FCM topic names


def test_waypoints_are_the_named_exits():
    """Each waypoint sits at its own town's exit: a guard against names drifting off their positions again."""
    from dustahead.app.predict import I10_EXITS

    def km_between(lon0, lat0, lon1, lat1):
        return float(np.hypot((lon1 - lon0) * 111.32 * np.cos(np.radians(lat0)), (lat1 - lat0) * 110.57))

    towns = {"Phoenix": (-112.07, 33.45), "Casa Grande": (-111.76, 32.88), "Eloy": (-111.55, 32.76),
             "Marana": (-111.22, 32.44), "Tucson": (-110.97, 32.22), "Willcox": (-109.83, 32.25),
             "Lordsburg": (-108.71, 32.35), "Deming": (-107.76, 32.27), "Las Cruces": (-106.78, 32.32)}
    for name, _, _, lon, lat in I10_EXITS:
        if name in towns:
            assert km_between(lon, lat, *towns[name]) < 10, name
    lons = [e[3] for e in I10_EXITS]
    assert lons == sorted(lons)                                    # west -> east
    assert [e[2] for e in I10_EXITS if e[1] == "AZ"] == sorted(e[2] for e in I10_EXITS if e[1] == "AZ")


def test_png_rows_are_north_up_and_mercator(engine):
    g = engine.grid
    rows = B.mercator_rows(g)
    assert rows[0] == g.ny - 1 and rows[-1] == 0 and np.all(np.diff(rows) <= 0)
    # Web Mercator stretches the north: the northern half of the image covers fewer degrees than the southern half
    top, bottom = rows[: g.ny // 2], rows[g.ny // 2:]
    assert top[0] - top[-1] < bottom[0] - bottom[-1]


def test_colorize_hides_low_p():
    rgba = B.colorize(np.array([[0.0, 0.1, 0.5, 1.0]]))
    assert rgba[0, 0, 3] == 0 and rgba[0, 1, 3] == 0
    assert 0 < rgba[0, 2, 3] < rgba[0, 3, 3]


def _storm(engine, seg_index: int, p: float):
    """Probability maps with dust only on the cells of one road segment."""
    probs = np.zeros((len(engine.leads), engine.grid.ny, engine.grid.nx), np.float32)
    _, i, j = engine.road_cells()[seg_index]
    probs[:, i, j] = p
    return probs


def test_bundle_carries_the_policy_decision(engine, tmp_path):
    probs = _storm(engine, 8, 0.95)                      # Eloy -> Picacho Peak, P above threshold + severe margin
    st_ij = np.array([[10, 10]])
    st_now = np.array([-1], np.int8)                     # one clear report far away: the segment is genesis
    gmap = engine.genesis_map(st_ij, st_now, probs.shape[1:])
    fc = dict(probs=probs, genesis=gmap, t0=np.datetime64("2026-08-25T23:10"), station_ij=st_ij, station_now=st_now,
              alerts=engine.road_alerts(probs, genesis_map=gmap))
    fc["alerts"] = AlertTracker(engine.cfg).update(fc["alerts"], probs, engine)
    bpath, files = B.write_bundle(fc, engine, tmp_path, source="fixture", nws=[])
    doc = json.loads((tmp_path / "latest.json").read_text())
    assert files[-1].name == "latest.json"                         # pointer is written (and uploaded) last
    assert doc["t0"] == "2026-08-25T23:10:00Z" and doc["source"] == "fixture"
    assert doc["layers"] == {"15": "runs/202608252310/p15.png", "30": "runs/202608252310/p30.png",
                             "60": "runs/202608252310/p60.png"}
    assert all((tmp_path / v).exists() for v in doc["layers"].values())
    # neighbours share the waypoint cell with segment 8, so they alert too (as road_alerts always has)
    assert [x["segment"] for x in doc["alerts"]] == ["i10-az-194-208", "i10-az-208-219", "i10-az-219-236"]
    a = doc["alerts"][1]
    assert a["segment"] == "i10-az-208-219" and a["level"] == "severe" and a["lead_min"] == 15 and a["genesis"]
    seg = next(s for s in doc["segments"] if s["id"] == a["segment"])
    assert seg["p"] == [0.95, 0.95, 0.95] and seg["mp"] == [208, 219] and seg["genesis"]
    assert doc["stations"][0]["dust"] is False and doc["dust_now"] == []
    assert doc["policy"]["stale_after_min"] == 25 and doc["nws"] == []
    # two alert tiers (product.tiers): no station reports dust within 50 km, so this is the "new storm" tier, and the
    # default config's product model is the one the hit rates were measured for, so the rate is published
    T = doc["policy"]["tiers"]
    if T is not None:
        assert a["tier"] == "new_storm" and T["rated"] and a["hit_rate"] == T["new_storm"]["hit_rate"]


def test_png_puts_dust_where_the_model_put_it(engine, tmp_path):
    from PIL import Image
    probs = np.zeros((3, engine.grid.ny, engine.grid.nx), np.float32)
    probs[:, -20:, :] = 1.0                              # northernmost 20 rows (row 0 = south in the grid)
    B.prob_png(probs[0], engine.grid, tmp_path / "p.png")
    a = np.asarray(Image.open(tmp_path / "p.png"))
    assert a.shape == (engine.grid.ny, engine.grid.nx, 4)
    assert a[0, :, 3].min() > 0 and a[-1, :, 3].max() == 0          # image top = north


def test_nws_parse_keeps_dust_products_in_the_domain():
    fc = {"features": [
        {"properties": {"event": "Dust Storm Warning", "areaDesc": "Pinal, AZ", "geocode": {"UGC": ["AZC021"]},
                        "parameters": {"VTEC": ["/O.NEW.KPSR.DS.W.0007.260825T2300Z-260826T0000Z/"]},
                        "ends": "2026-08-26T00:00:00Z"},
         "geometry": {"type": "Polygon", "coordinates": [[[-111.9, 32.5], [-111.0, 32.5], [-111.0, 33.2], [-111.9, 32.5]]]}},
        {"properties": {"event": "Flood Warning", "areaDesc": "Pima, AZ", "geocode": {"UGC": ["AZC019"]}}, "geometry": None},
        {"properties": {"event": "Blowing Dust Advisory", "areaDesc": "Lubbock, TX", "geocode": {"UGC": ["TXZ035"]}},
         "geometry": None},
    ]}
    [w] = nws.parse(fc)
    assert w["type"] == "Dust Storm Warning" and ".DS.W." in w["vtec"] and w["polygon"] is not None
