# DustAhead data sources — verified inventory

Every source below was checked against the live server on **23 Oct 2026** from the development workspace, except the two
marked *PC-only* (they refuse automated readers from that network and are tested by `scripts/00_check_sources.py`).
Nothing in Phase 1 needs an account.

## Phase 1 (now): Arizona + New Mexico, 2023–2026

| # | Data set | What we take | Where | Cadence / size | Verified |
|---|---|---|---|---|---|
| 1 | **GOES-18 ABI Cloud & Moisture Imagery, CONUS** (`ABI-L2-MCMIPC`) | Bands 2, 7, 11, 13, 14, 15 (0.64, 3.9, 8.4, 10.3, 11.2, 12.3 µm) + quality flags | `s3://noaa-goes18/ABI-L2-MCMIPC/YYYY/DDD/HH/` (anonymous) | every 5 min; 47–52 MB per file, all 16 bands at 2 km; we read ~18 MB remotely and cache 0.6 MB | ✅ 12 files listed for 2025-238 02Z; file-name pattern `_sYYYYDDDHHMMSSt_` confirmed |
| 2 | **GOES-18 Aerosol Detection Product** (`ABI-L2-ADPC`) | `Dust` flag + DQF → weak pixel labels (weight 0.3) | `s3://noaa-goes18/ABI-L2-ADPC/YYYY/DDD/HH/` | every 10 min; ~0.75 MB | ✅ 6 files per hour |
| 3 | **GOES-18 Geostationary Lightning Mapper** (`GLM-L2-LCFA`) | flash lat/lon → flashes per cell per 10 min | `s3://noaa-goes18/GLM-L2-LCFA/YYYY/DDD/HH/` | every 20 s; ~0.3 MB; 30 files per window | ✅ |
| 4 | **HRRR surface analysis** (High-Resolution Rapid Refresh) | u10, v10, gust, CAPE, RH 2 m, soil moisture, T 2 m, dewpoint 2 m (f00); 0–6 h precipitation (f06) for 7-day rain | `s3://noaa-hrrr-bdp-pds/hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcfFF.grib2` via **Herbie** (byte-range by `.idx`) | hourly; 138 MB per file, ~1–2 MB per field fetched | ✅ file + .idx present |
| 5 | **NWS Dust Storm Warnings** (DS.W) and **Severe Thunderstorm Warnings** (SV.W) | issue/expire, WFO, event id, polygon → event list, null days, label cross-check | IEM `json/vtec_events_bystate.py`, polygons `geojson/vtec_event.py?...&sbw=1` | per state-year JSON | ✅ AZ 2025: 56 DS.W; polygon PSR #10 returns MultiPolygon |
| 6 | **ASOS/AWOS station reports** | visibility, present-weather codes, wind, gust, RH, 1-h precip | IEM `cgi-bin/request/asos.py` with `report_type=1,2,3` (5-min, routine, specials) | 5 min at ASOS airports; CSV | ✅ PHX 2025-08-26 00:35Z: `-TSRA +DS`, vis 0.00 mi, gust 61 kt |
| 7 | **Station metadata** | lon/lat, county, WFO, online flag | IEM `geojson/network/{AZ,NM,TX,CA}_ASOS.geojson` | static | ✅ AZ_ASOS 50 stations |
| 8 | **NCEI Storm Events Database** | `EVENT_TYPE == "Dust Storm"` begin/end, county, deaths, injuries → second label cross-check and casualty stats | `https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_dYYYY_c*.csv.gz` | yearly; 7–13 MB | ✅ 2023, 2024, 2025, 2026 files listed |
| 9 | **USGS 3DEP elevation** | elevation → slope | `elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer` via **py3dep** | static; 90 m request | ✅ ImageServer JSON |
| 10 | **SoilGrids 2.0** | sand %, clay % (0–5 cm) → soil texture + Fécan threshold | `https://maps.isric.org/mapserv?map=/map/{sand,clay}.map` WCS 2.0.1, coverage `clay_0-5cm_mean` | static; 250 m | ⚠️ *PC-only* (robots block) |
| 11 | **USDA Cropland Data Layer** | fallow/idle, barren, harvested crops → bare fraction | CropScape `GetCDLFile?year=&bbox=` (EPSG:5070) | yearly; 30 m | ⚠️ *PC-only* (robots block) |
| 12 | **Annual NLCD land cover** (optional) | class map, bare/shrub fraction | https://www.mrlc.gov/data → Annual NLCD Collection 1.2 (1985–2025), manual download | static; 30 m | ✅ product page; manual |

Casualty statistics for the paper come from #8 plus the published studies (Lader et al. 2016; Tong et al. 2023).

## Volume and time (130 events × ~45 forecast times)

| | Transfer | Stored |
|---|---|---|
| GOES frames (remote read, ~18 MB each; ~50 frames per event) | ~115 GB | 4 GB cache |
| GLM (30 × 0.3 MB per window) | ~55 GB | 1 GB cache |
| ADP (3 leads × 0.75 MB) | ~12 GB | — |
| HRRR fields (Herbie byte-range) | ~15 GB | 2 GB cache |
| Samples (62 ch × 200 × 250 float16, compressed) | — | ~25 GB |
| **Total** | **~200 GB** | **~35 GB** |

At 100 Mbit/s that is ~5 h of pure transfer; with request latency plan **1–2 days** with `--workers 4` to `6`. Resumable.

## Phase 2–3 (later, accounts needed)

| Data set | Use | Access |
|---|---|---|
| GOES-17 ABI/GLM 2019–2022 (`noaa-goes17`) | more training events (train only; flag loop-heat-pipe periods) | anonymous S3 |
| FHWA HPMS road geometry + AADT | impact-weighted scoring, exact I-10 geometry | `geo.dot.gov` FeatureServer (2018 release is the latest shapefile set listed) |
| OpenStreetMap I-10 centreline | road cells | Overpass API |
| MODIS / VIIRS AOD | extra weak labels | NASA Earthdata login (free) |
| Meteosat SEVIRI (Gulf / Sahara transfer) | Phase 3 transfer learning | EUMETSAT Data Store account (free) |
| ERA5 / ERA5-Land | HRRR equivalent outside the US | Copernicus CDS account (free) |

## Rules learned from the real data

1. **A dust code is required; rain does not exclude.** Haboobs arrive with thunderstorm rain (`-TSRA +DS`). The earlier
   "no rain" clause would have dropped the Phoenix 2025 event. Rain-only low visibility has no dust code and is rejected.
2. **Use the 5-minute feed** (`report_type=1`): ~10× more station truth than hourly METARs. Specials at the same minute
   are de-duplicated by keeping the richer weather string.
3. **Never keep raw GOES files.** Read the six bands remotely from the HDF5 and cache the regridded 0.6 MB frame.
4. **HRRR f00 has no precipitation.** Seven-day rain comes from the 0–6 h accumulations of the four daily synoptic runs.
5. **Storm Events file names change** with each monthly reprocessing; list the directory, never hard-code the name.
