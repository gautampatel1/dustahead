// MapLibre GL JS 6 builds its worker URL at run time, so no bundler emits the file. Copy it next to the site and
// point maplibre at it (MapView: setWorkerUrl). Runs before `dev` and `build`.
import { copyFileSync, mkdirSync } from "node:fs";
mkdirSync("public/maplibre", { recursive: true });
copyFileSync("node_modules/maplibre-gl/dist/maplibre-gl-worker.mjs", "public/maplibre/maplibre-gl-worker.mjs");
