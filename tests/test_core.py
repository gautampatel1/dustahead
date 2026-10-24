import datetime as dt

import numpy as np
import pandas as pd
import pytest

from dustahead import load_config
from dustahead.data import physics
from dustahead.data.sampler import N_CH, assemble, paint_targets
from dustahead.data.synthetic import synthetic_sample
from dustahead.events.stations import has_code, label_reports
from dustahead.events.warnings import build_event_list, point_in_rings
from dustahead.eval.metrics import ece, scores_from_counts, station_scores, fss_np
from dustahead.eval.bootstrap import holm
from dustahead.grid import Grid

CFG = load_config()


def test_grid_roundtrip_and_size():
    g = Grid.from_config(CFG)
    assert g.shape == (200, 250)
    ew, ns = g.cell_km()
    assert 3 < ew < 5 and 3 < ns < 4
    i, j = g.to_ij(-112.01, 33.43)            # Phoenix
    lon, lat = g.lons()[j], g.lats()[i]
    assert abs(lon + 112.01) < g.dlon and abs(lat - 33.43) < g.dlat
    assert g.disk_mask(-112.01, 33.43, 12).sum() > 20
    dens = g.bin_points([-112.0, -112.0, -120.0], [33.4, 33.4, 33.4])
    assert dens.sum() == 2


def test_physics_worked_example():
    # clay 10 %: w' = 1.84 %; soil moisture 5 % -> f_w = 1.91 (proposal section 5)
    f = physics.moisture_factor(5.0, 10.0)
    assert abs(f - 1.91) < 0.02
    assert physics.moisture_factor(0.5, 10.0) == 1.0           # drier than w' -> no increase
    e_dry = physics.erodibility_prior(8.0, 0.5, 10.0, 1.0)    # u* = 0.46 m/s vs dry threshold 0.25
    e_wet = physics.erodibility_prior(8.0, 20.0, 10.0, 1.0)   # wet: threshold rises to 0.78 m/s
    assert e_dry > 0.9 and e_wet < 0.1


def test_wx_codes():
    assert has_code("BLDU", CFG["label"]["dust_codes"])
    assert has_code("VCBLDU", CFG["label"]["dust_codes"])
    assert has_code("TSRA BLDU", CFG["label"]["rain_codes"])
    assert not has_code("HZ", CFG["label"]["rain_codes"])


def test_label_rule_requires_warning_and_no_rain():
    stations = pd.DataFrame(dict(sid=["PHX"], lon=[-112.01], lat=[33.43], county=["Maricopa"]))
    t = pd.Timestamp("2025-08-26T02:00Z")
    obs = pd.DataFrame(dict(station=["PHX"] * 4, valid=[t] * 4, vsby=[0.25, 0.25, 0.25, 5.0],
                            wxcodes=["BLDU", "TSRA BLDU", "BLDU", "BLDU"]))
    w = pd.DataFrame(dict(phenomena=["DS"], wfo=["PSR"], year=[2025], eventid=[12],
                          issue=[t - pd.Timedelta(hours=1)], expire=[t + pd.Timedelta(minutes=30)], locations=["Maricopa, Pinal [AZ]"]))
    out = label_reports(obs, w, stations, CFG)
    assert out.label.tolist() == [1, 1, 1, 0]         # "-TSRA +DS" is still dust (Phoenix 2025-08-26); visibility kills row 4
    assert out.rain_too.tolist() == [False, True, False, False]
    assert out.severe.tolist() == [1, 1, 1, 0]
    # rain-only low visibility is NOT dust (no dust code)
    obs_rain = obs.iloc[:1].assign(wxcodes="+TSRA", vsby=0.5)
    assert label_reports(obs_rain, w, stations, CFG).label.tolist() == [0]
    # Storm Events entry can confirm instead of a warning
    se = pd.DataFrame(dict(begin=[t - pd.Timedelta(hours=1)], end=[t + pd.Timedelta(hours=1)], cz_name=["MARICOPA"]))
    far = w.assign(issue=t + pd.Timedelta(hours=5), expire=t + pd.Timedelta(hours=6))
    assert label_reports(obs.iloc[:1], far, stations, CFG, storm_events=se).label.tolist() == [1]
    # no warning -> no label even with dust code
    out2 = label_reports(obs.iloc[:1], w.assign(issue=t + pd.Timedelta(hours=5), expire=t + pd.Timedelta(hours=6)), stations, CFG)
    assert out2.label.tolist() == [0]
    # polygon check
    ring = np.array([[-113, 33], [-111, 33], [-111, 34], [-113, 34], [-113, 33]], float)
    assert point_in_rings(-112.0, 33.5, [ring]) and not point_in_rings(-110.0, 33.5, [ring])


def test_event_clustering_and_null_days():
    t = pd.Timestamp("2025-08-25T22:00Z")
    rows = [dict(state="AZ", year=2025, wfo="PSR", eventid=k, phenomena="DS", significance="W",
                 issue=t + pd.Timedelta(minutes=40 * k), expire=t + pd.Timedelta(minutes=40 * k + 60), locations="Maricopa", area=1)
            for k in range(3)]
    rows.append(dict(state="AZ", year=2025, wfo="PSR", eventid=9, phenomena="DS", significance="W",
                     issue=t + pd.Timedelta(days=3), expire=t + pd.Timedelta(days=3, hours=1), locations="Pinal", area=1))
    rows.append(dict(state="AZ", year=2025, wfo="PSR", eventid=50, phenomena="SV", significance="W",
                     issue=t + pd.Timedelta(days=10), expire=t + pd.Timedelta(days=10, hours=1), locations="Pinal", area=1))
    rows.append(dict(state="AZ", year=2025, wfo="PSR", eventid=51, phenomena="SV", significance="W",
                     issue=t + pd.Timedelta(hours=2), expire=t + pd.Timedelta(hours=3), locations="Pinal", area=1))
    ev = build_event_list(pd.DataFrame(rows))
    assert (ev.kind == "dust").sum() == 2            # three warnings 40 min apart = one event, + one 3 days later
    assert ev[ev.kind == "dust"].iloc[0].n_warnings == 3
    assert (ev.kind == "null").sum() == 1            # SV on the dust day is not a null day


def test_sample_times_window():
    from dustahead.data.sampler import sample_times
    ev = pd.Series(dict(t_start=pd.Timestamp("2025-08-25T22:36Z"), t_end=pd.Timestamp("2025-08-26T01:45Z")))
    ts = sample_times(ev, CFG)
    assert ts[0] == pd.Timestamp("2025-08-25T19:30Z").to_pydatetime()     # 3 h before, floored to 10 min
    assert ts[-1] <= pd.Timestamp("2025-08-26T03:50Z").to_pydatetime()    # 2 h after, ceiled
    assert 40 <= len(ts) <= 72 and (ts[1] - ts[0]).total_seconds() == 600


def test_synthetic_sample_shapes():
    g = Grid(nx=80, ny=64)
    s = synthetic_sample(g, np.random.default_rng(0))
    assert s["x"].shape == (N_CH, 64, 80) and N_CH == 62
    assert s["y"].shape == (3, 64, 80) and s["y"].max() == 1
    assert np.isfinite(s["x"].astype(np.float32)).all()


def test_paint_targets():
    g = Grid(nx=80, ny=64)
    st = pd.DataFrame(dict(lon=[-112.0, -106.0], lat=[33.0, 33.0], y_15=[1, 0], y_30=[1, -1], y_60=[-1, 0]))
    y, w = paint_targets(g, st, (15, 30, 60), 30.0)
    assert y[0].sum() > 0 and w[0].sum() > y[0].sum()      # positive disk + negative disk
    assert w[1].sum() == y[1].sum()                         # only the positive station at +30
    assert y[2].sum() == 0 and w[2].sum() > 0


def test_metrics():
    s = scores_from_counts(8, 2, 2, 88)
    assert abs(s["pod"] - 0.8) < 1e-9 and abs(s["far"] - 0.2) < 1e-9 and abs(s["csi"] - 8 / 12) < 1e-9
    p = np.array([0.9, 0.8, 0.2, 0.1, 0.6]); y = np.array([1, 1, 0, 0, -1])
    r = station_scores(p, y, 0.5)
    assert r["csi"] == 1.0 and r["n"] == 4 and r["bss"] > 0
    assert 0 <= ece(p, y) <= 1
    a = np.zeros((20, 20)); a[5:10, 5:10] = 1
    assert fss_np(a, a) == 1.0 and fss_np(a, np.roll(a, 8, 0)) < 0.5
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}
