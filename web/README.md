# DustAhead web app

A plain, light website with five tabs: Map, Road conditions ("Your drive" trip timing and a table of every stretch), Warnings (NWS),
Safety and About (how it works, accuracy, limits, data sources, and forecast status). It's a static Next.js site: no
server and no API.
It reads the files that `scripts/35_publish.py` writes.

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
npm test                        # drive timing and display rules (Node's built-in test runner)
```

Regenerate the fixture after changing the bundle format. It needs no model and no data:

```bash
python scripts/35_publish.py --fixture
```

## Data sources

| URL | What it shows |
|---|---|
| `/` | The fixture replay in `public/demo/`, shown with a "Demo mode" notice. |
| `/?src=https://<bucket>/` | The live forecast: polls `latest.json` every 60 s and goes stale after 25 min. |
| `/?src=replay/<event>/&replay=1` | A replay written by `35_publish.py --replay` (copy it into `public/replay/`). |
| `&tz=America/Denver` | Shows clock times in Mountain time. The default is Arizona time; the header has a picker. |
| `&lang=es` | Spanish. The browser language and the toggle in the header also set it. |

Build-time settings, all optional:

- `NEXT_PUBLIC_DATA_URL`: the public R2 URL. When it's set, `/` shows the live forecast.
- `NEXT_PUBLIC_BASEMAP_STYLE`: a MapLibre style URL. The default is OpenFreeMap "positron" (light grey), which needs no key. A self-hosted Protomaps style works here too.
- `NEXT_PUBLIC_BASE_PATH`: set this when the site lives under a sub-path.
- `NEXT_PUBLIC_SITE_URL`: the site's public address (e.g. `https://dustahead.example.org`), used for the image in link previews. Without it, previews show text only.

## Rules the app keeps

- **It never decides alerts itself.** Road states come from the worker's alert list (`road_alerts` + `AlertTracker`), so the web map, the phone and the pushes agree.
- **Stale is unknown, never clear.** The age is worked out from `t0` on the viewer's clock. After `policy.stale_after_min` (25 min) the map turns grey, every segment shows "No data", and the panel says it is not an all-clear.
- **Official warnings first.** Active NWS dust products are shown whatever the model says. `nws: null` means the feed could not be read, and the app says so.
- **Two alert tiers, with measured hit rates** (`product.tiers` on main). "Nearby" (a station within 50 km reports dust, alert at P ≥ 0.5) shows as Watch/Severe; "new storm" (none does, P ≥ 0.3) shows as "Dust possible". Alert details say how often each tier came true on 2025 validation (78 % and 14 %), but only when `policy.tiers.rated` is true, i.e. the product model is the one the rates were measured for.

## Your drive

Road conditions starts with a trip planner (`lib/drive.ts`, `components/Drive.tsx`). Pick a start (or "Start from
where I am"), a destination, when you leave and a speed. For each stretch it gives the time you'd get there and the
chance of dust at that time:
- **Distances** are the bundle's `miles` per stretch, taken from exit numbers and including state-line crossings.
- **Chance on a stretch** is read from the forecast maps at the times you're on it, interpolated between +15, +30
  and +60. Before +15 the +15 map is used.
- **After +60** a stretch shows "After the forecast hour: check again", never clear.
- **Advice:** the first stretch whose chance reaches its tier's alert threshold gives "plan to stop before <town>
  (exit N)".
- **Location** is used once, only to pick the nearest town, and stays in the browser. Choices are remembered in
  `localStorage`.

## Hosting on R2

1. Create the bucket and turn on its public URL (or a custom domain).
2. Add a CORS rule allowing `GET` from the site's origin. MapLibre fetches the PNGs with CORS.
3. On the PC, put `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` and `DUSTAHEAD_HC_URL` in `.secrets\publish.env` (template: `configs/publish.env.example`; gitignored, never in the repo).
4. On the PC, run `scripts\windows\install_publish_task.bat`. It schedules `35_publish.py --live --once --upload` every 10 minutes at :02, :12 and so on. See `RUNBOOK_WINDOWS.md`, "Live forecasts for the apps".
5. Build the site with `NEXT_PUBLIC_DATA_URL` set, then upload `out/` to the same bucket or any static host.

## Notes

- MapLibre GL JS 6 loads its worker from a URL. `tools/copy-maplibre-worker.mjs` copies the worker into `public/maplibre/` before `dev` and `build`.
- Motion turns off under `prefers-reduced-motion`.
- The free OpenFreeMap basemap can be slow: on 26 Nov a low-zoom tile was ~200 kB and ~1 s, and MapLibre decodes those
  tiles in the same worker as our roads, so in dev mode the roads took up to ~20 s to appear (about 3 s in the
  production build). For the demo, host a Protomaps extract of the domain on R2 and set `NEXT_PUBLIC_BASEMAP_STYLE`.
- Add to home screen: `app/manifest.ts` and the icons in `public/icons/` make the site installable on phones.
- The Spanish copy has not been reviewed by a native speaker yet.
