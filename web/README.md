# DustAhead web app

A plain, light website with five tabs: Map, Road conditions (route check and a table of every stretch), Warnings (NWS),
Safety and About (how it works, accuracy, limits, data sources, and forecast status). It's a static Next.js site: no
server and no API.
It reads the files that `scripts/30_publish.py` writes.

```
PC worker ── predict() → AlertTracker → bundle writer ──► R2 bucket ──► this site (static) and the phone app
                                         latest.json + runs/<t0>/p15|p30|p60.png
```

## Run it

```bash
cd web && npm install
npm run dev                     # http://localhost:3000, plays the illustrative fixture in public/demo
npm run build                   # static export into out/
npm run serve                   # serve out/ on http://localhost:4173
```

Regenerate the fixture after changing the bundle format. It needs no model and no data:

```bash
python scripts/30_publish.py --fixture
```

## Data sources

| URL | What it shows |
|---|---|
| `/` | The fixture replay in `public/demo/`, shown with a "Demo mode" notice. |
| `/?src=https://<bucket>/` | The live forecast: polls `latest.json` every 60 s and goes stale after 25 min. |
| `/?src=replay/<event>/&replay=1` | A replay written by `30_publish.py --replay` (copy it into `public/replay/`). |
| `&tz=America/Denver` | Shows clock times in Mountain time. The default is Arizona time; the header has a picker. |
| `&lang=es` | Spanish. The browser language and the toggle in the header also set it. |

Build-time settings, all optional:

- `NEXT_PUBLIC_DATA_URL`: the public R2 URL. When it's set, `/` shows the live forecast.
- `NEXT_PUBLIC_BASEMAP_STYLE`: a MapLibre style URL. The default is OpenFreeMap "positron" (light grey), which needs no key. A self-hosted Protomaps style works here too.
- `NEXT_PUBLIC_BASE_PATH`: set this when the site lives under a sub-path.

## Rules the app keeps

- **It never decides alerts itself.** Road states come from the worker's alert list (`road_alerts` + `AlertTracker`), so the web map, the phone and the pushes agree.
- **Stale is unknown, never clear.** The age is worked out from `t0` on the viewer's clock. After `policy.stale_after_min` (25 min) the map turns grey, every segment shows "No data", and the panel says it is not an all-clear.
- **Official warnings first.** Active NWS dust products are shown whatever the model says. `nws: null` means the feed could not be read, and the app says so.
- **Genesis alerts are shown as "Developing".** These are segments with no station reporting dust within 50 km. Onset skill is still low (validation CSI about 0.1–0.2).

## Hosting on R2

1. Create the bucket and turn on its public URL (or a custom domain).
2. Add a CORS rule allowing `GET` from the site's origin. MapLibre fetches the PNGs with CORS.
3. On the PC, put `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` and `DUSTAHEAD_HC_URL` in the environment (never in the repo).
4. Schedule `python scripts/30_publish.py --live --once --upload` every 10 minutes, at :02, :12 and so on.
5. Build the site with `NEXT_PUBLIC_DATA_URL` set, then upload `out/` to the same bucket or any static host.

## Notes

- MapLibre GL JS 6 loads its worker from a URL. `tools/copy-maplibre-worker.mjs` copies the worker into `public/maplibre/` before `dev` and `build`.
- Motion turns off under `prefers-reduced-motion`.
- The Spanish copy has not been reviewed by a native speaker yet.
