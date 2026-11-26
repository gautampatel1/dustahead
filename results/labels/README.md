# Extended ground truth from Synoptic Data — calibration against the ASOS rule (4 Nov 2026)

Inventory inside the domain (data/labels/synoptic_stations.csv): 200 stations with visibility sensors (87 ASOS/AWOS,
37 ADOT/NMDOT road-weather = RWIS, 58 White Sands / Yuma range stations, misc.) and 63 PM10 monitors (AirNow, NMAQ).
Historical access 2018–present was granted on request (academic tier).

## Do the new sources mean the same thing as the ASOS dust label?
Co-located check (`scripts/17_calibrate_pm10.py`, `_label_agreement.py`): extended observations paired with ASOS
reports within 12 km and ±30 min (±5 min for visibility), over all 542 event windows.

| source | pairs | extended says dust → ASOS agrees | ASOS says dust → extended agrees | best CSI of the raw value vs the ASOS label |
|---|---|---|---|---|
| PM10 (hourly, warning-filtered, ≥ 250 µg/m³) | 128k | **9 %** (47 of 511) | 46 % (47 of 103) | 0.05 at ≥ 500–1,250 µg/m³ (FAR ≥ 0.92 at every threshold) |
| range / mesonet visibility ≤ 1 mi (warning-filtered) | 544k | **0 %** (0 of 251) | — (no ASOS dust in those pairs) | 0 |
| road-weather (RWIS) visibility ≤ 1 mi, no rain, warning-filtered | 99k | no co-located ASOS dust either way | — | untestable this way |

Median PM10 when a co-located ASOS station reports a dust storm: 198 µg/m³; when it reports clear: 21. The hourly
PM10 average rises with any dusty air (local sources, agriculture, the hours after a haboob) and is too coarse in
time to mark a visibility collapse; the range visibility sensors read low visibility ASOS never calls dust.
**Neither is the quantity the product forecasts (road visibility ≤ 1 mile in a dust storm). They are not used as labels.**

RWIS (887 positive rows in 61 events, up to 8 stations per event) measures the right quantity on the road itself,
but only 19 of its 61 dust events coincide with any ASOS dust report within an hour (nearest concurrent ASOS dust
report: median 96 km away). Either RWIS sees dust where airports have no station — the reason to add it — or part
of it is sensor noise. The test is empirical: v13 / v13a train on ASOS + RWIS labels and are scored on the
ASOS-only validation stations; if ASOS-only skill holds or improves, the RWIS labels carry signal.

Numbers from the first (too permissive) extension, kept for the record: with PM10 ≥ 250 and the range sensors as
labels, validation positives rose 208 → 1,585 at +30, but station persistence at the ASOS stations fell from CSI
0.443 to 0.239 — the new "dust now" reports contradicted the ASOS truth. That is what prompted the calibration.
