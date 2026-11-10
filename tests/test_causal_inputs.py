"""Hard rule 3 - only data available at forecast time goes into a sample: station channel, GOES frames, HRRR analysis."""
import datetime as dt
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from dustahead import load_config
from dustahead.data import goes
from dustahead.data.hrrr import analysis_run
from dustahead.data.sampler import paint_station_now
from dustahead.events.stations import station_now, station_targets

CFG = load_config()
UTC = dt.timezone.utc


def test_station_now_uses_only_reports_at_or_before_t0_and_no_warning():
    t0 = pd.Timestamp("2025-08-26T02:00Z")
    m = pd.Timedelta(minutes=1)
    obs = pd.DataFrame(dict(
        station=["PHX", "TUS", "TUS", "CGZ", "LUF", "ELP", "DMN", "DMN", "DMN", "DMN"],
        valid=[t0 - 5 * m, t0 - 10 * m, t0 + 5 * m, t0 - 30 * m, t0 - 5 * m, t0 - 5 * m,
               t0 - 19 * m, t0 - 15 * m, t0 - 10 * m, t0 - 5 * m],
        vsby=[0.5, 10.0, 0.1, 0.25, 0.5, np.nan, 0.5, 10.0, 10.0, 10.0],
        wxcodes=["+DS", "", "BLDU", "BLDU", "HZ", "BLDU", "BLDU", "", "", ""]))
    now = dict(zip(["PHX", "TUS", "CGZ", "LUF", "ELP", "DMN", "YUM"],
                   station_now(obs, ["PHX", "TUS", "CGZ", "LUF", "ELP", "DMN", "YUM"], t0, CFG)))
    # stored encoding: 1 dust, 0 clear report, -1 no report
    assert now["PHX"] == 1          # dust code + low visibility, no warning needed (a warning may come later)
    assert now["TUS"] == 0          # the dust report 5 min AFTER t0 is not visible at t0
    assert now["CGZ"] == -1         # only report is older than the 20-min window
    assert now["LUF"] == 0          # haze alone is not dust without a warning
    assert now["ELP"] == 0          # missing visibility counts as clear (as in the live feed)
    assert now["DMN"] == 0          # dust 19 min ago, but not among the last 3 reports
    assert now["YUM"] == -1         # no report at all
    # a naive t0 means UTC, exactly as stored in the samples
    assert station_now(obs, ["PHX"], t0.tz_localize(None), CFG).tolist() == [1]
    # the painted channel: dust disc +1, clear disc -1, unreported station absent
    ch = paint_station_now(np.array([[5, 5], [5, 20], [20, 5]]), station_now(obs, ["PHX", "TUS", "YUM"], t0, CFG), (30, 30))
    assert ch[5, 5] == 1 and ch[5, 20] == -1 and ch[20, 5] == 0


def test_station_targets_no_longer_carry_an_input():
    t0 = pd.Timestamp("2025-08-26T02:00Z")
    labels = pd.DataFrame(dict(station=["PHX"] * 2, valid=[t0, t0 + pd.Timedelta(minutes=30)], label=[1, 1]))
    st = station_targets(labels, pd.DataFrame(dict(sid=["PHX"])), t0, [15, 30, 60])
    assert "dust_now" not in st.columns and st.y_30.tolist() == [1] and st.y_15.tolist() == [-1]


def _name(s: dt.datetime, c: dt.datetime) -> str:
    stamp = lambda t: f"{t:%Y}{t.timetuple().tm_yday:03d}{t:%H%M%S}0"
    return f"noaa-goes19/ABI-L2-MCMIPC/OR_ABI-L2-MCMIPC-M6_G19_s{stamp(s)}_e{stamp(s)}_c{stamp(c)}.nc"


def test_available_file_never_uses_a_scan_after_t(monkeypatch):
    t = dt.datetime(2025, 8, 26, 2, 0, tzinfo=UTC)
    # CONUS scans start at :x1 and :x6 and are created ~3.5 min later
    starts = [t + dt.timedelta(minutes=k) for k in (-19, -14, -9, -4, 1, 6)]
    files = [_name(s, s + dt.timedelta(minutes=3, seconds=30)) for s in starts]
    monkeypatch.setattr(goes, "list_files", lambda bucket, product, tt: files if tt.hour == t.hour else [])
    assert goes.scan_start(goes.nearest_file(None, "ABI-L2-MCMIPC", t)) > t          # the old rule looked ahead
    best = goes.available_file(None, "ABI-L2-MCMIPC", t, latency_min=2.0)
    assert goes.scan_start(best) == t - dt.timedelta(minutes=9)    # -4 scan is created at -0.5, +2 min latency > t
    assert goes.available_file(None, "ABI-L2-MCMIPC", t, latency_min=0.0) == files[3]
    assert goes.available_file(None, "ABI-L2-MCMIPC", t, max_age_min=5.0) is None   # too old: report missing data


def test_hrrr_analysis_is_one_that_was_published_by_t0():
    for hh, mm, want in ((14, 0, 13), (14, 10, 13), (14, 59, 13), (15, 0, 14)):
        t0 = dt.datetime(2025, 8, 26, hh, mm, tzinfo=UTC)
        assert analysis_run(t0, 60) == dt.datetime(2025, 8, 26, want, tzinfo=UTC)


def test_in_place_fix_rebuilds_the_sampler_station_order():
    spec = importlib.util.spec_from_file_location("fix14", Path(__file__).parents[1] / "scripts" / "20_causal_station_now.py")
    fix14 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fix14)
    t0 = pd.Timestamp("2025-08-26T02:00Z")
    stations = pd.DataFrame(dict(sid=["TUS", "A39", "PHX", "ELP"], lon=0.0, lat=0.0, i=[1, 2, 3, 4], j=[5, 6, 7, 8]))
    labels = pd.DataFrame(dict(station=["PHX", "TUS", "XXX", "A39", "PHX"], valid=t0, label=0))
    st = station_targets(labels, stations, t0, [15, 30, 60])                     # what build_sample does ...
    st = st.merge(stations[["sid", "lon", "lat", "i", "j"]], left_on="station", right_on="sid")
    assert fix14.station_order(labels, stations) == st.station.tolist()           # ... and what the fix assumes
