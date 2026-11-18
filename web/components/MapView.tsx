"use client";
/**
 * The live map. MapLibre GL JS on a light grey basemap (OpenFreeMap "positron"), so the brown dust layer and the
 * coloured road states stand out.
 *
 * Dust layer: one PNG per lead, each draped as an `image` source. Two buffers per lead (A/B) let a new forecast
 * cross-fade over the old one instead of blinking, and the look-ahead slider blends the two nearest leads.
 * Roads: drawn from the worker's decision. State changes the line itself (width, dash, colour) so it reads in glare
 * and without colour vision. A segment that turns severe sends out one ring, once.
 */
import { Map as MLMap, NavigationControl, setWorkerUrl, type GeoJSONSource, type ImageSource, type LayerSpecification, type MapLayerMouseEvent } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useEffect, useRef } from "react";
import { layerWeights, resolve, type Loaded, type RoadState } from "@/lib/bundle";

const STYLE = process.env.NEXT_PUBLIC_BASEMAP_STYLE || "https://tiles.openfreemap.org/styles/positron";
const VIEW: [[number, number], [number, number]] = [[-114.9, 31.55], [-106.2, 33.95]];
const BUF = ["a", "b"] as const;
const WORKER = `${process.env.NEXT_PUBLIC_BASE_PATH || ""}/maplibre/maplibre-gl-worker.mjs`;

export const ROAD_COLOR: Record<RoadState, string> = {
  clear: "#7d8b88", watch: "#e09400", severe: "#d32f2f", held: "#a08453", early: "#2f6fb5", unknown: "#9aa3a1",
};
const ROAD_WIDTH: Record<RoadState, number> = { clear: 2.2, watch: 4.5, severe: 6.5, held: 4, early: 3.6, unknown: 3 };

interface Props {
  loaded: Loaded;
  minute: number;
  states: Record<string, RoadState>;
  selected: string | null;
  rings: string[];
  stale: boolean;
  reduceMotion: boolean;
  onSelect: (id: string | null) => void;
}

function matchExpr<T>(table: Record<RoadState, T>, fallback: T): unknown[] {
  return ["match", ["get", "state"], ...Object.entries(table).flat(), fallback];
}

function diamond(fill: string, stroke: string, size: number): { width: number; height: number; data: Uint8ClampedArray } {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d")!;
  const h = size / 2;
  g.beginPath();
  g.moveTo(h, 2); g.lineTo(size - 2, h); g.lineTo(h, size - 2); g.lineTo(2, h); g.closePath();
  g.fillStyle = fill; g.fill();
  g.lineWidth = 2.5; g.strokeStyle = stroke; g.stroke();
  return { width: size, height: size, data: g.getImageData(0, 0, size, size).data };
}

/** First view: frame the road segments under alert (all of I-10 when there are none), so a phone opens on the
 *  storm instead of on four states of desert. Later forecasts never move the camera: the user may have panned. */
function frameAlerts(m: MLMap, L: Loaded) {
  const ids = new Set(L.bundle.alerts.map((a) => a.segment));
  const pts = L.bundle.segments.filter((s) => ids.has(s.id)).flatMap((s) => s.line);
  if (pts.length === 0) return;
  const lon = pts.map((p) => p[0]), lat = pts.map((p) => p[1]);
  m.fitBounds([[Math.min(...lon), Math.min(...lat)], [Math.max(...lon), Math.max(...lat)]],
    { padding: { top: 56, bottom: 72, left: 28, right: 28 }, maxZoom: 8.5, duration: 0 });
}

export default function MapView({ loaded, minute, states, selected, rings, stale, reduceMotion, onSelect }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const map = useRef<MLMap | null>(null);
  const ready = useRef(false);
  const active = useRef<0 | 1>(0);       // which buffer holds the current forecast
  const blend = useRef(1);               // 0..1 progress of the cross-fade into the active buffer
  const shownT0 = useRef<string | null>(null);
  const props = useRef({ loaded, minute, states, selected, rings, stale, reduceMotion, onSelect });
  props.current = { loaded, minute, states, selected, rings, stale, reduceMotion, onSelect };

  function corners(b: Loaded["bundle"]): [[number, number], [number, number], [number, number], [number, number]] {
    const [w, s, e, n] = b.bounds;
    return [[w, n], [e, n], [e, s], [w, s]];
  }

  function applyOpacity() {
    const m = map.current;
    if (!m || !ready.current) return;
    const { loaded: L, minute: t, stale: st } = props.current;
    const w = layerWeights(L.bundle.leads, t);
    const scale = st ? 0.45 : 0.92;
    L.bundle.leads.forEach((lead, k) => {
      BUF.forEach((tag, bi) => {
        const id = `p${lead}${tag}`;
        if (!m.getLayer(id)) return;
        const mine = bi === active.current ? blend.current : 1 - blend.current;
        m.setPaintProperty(id, "raster-opacity", w[k] * mine * scale);
      });
    });
  }

  function roadData() {
    const { loaded: L, states: S } = props.current;
    return {
      type: "FeatureCollection" as const,
      features: L.bundle.segments.map((s) => ({
        type: "Feature" as const,
        properties: { id: s.id, state: S[s.id] ?? "clear" },
        geometry: { type: "LineString" as const, coordinates: s.line },
      })),
    };
  }

  function stationData() {
    const { loaded: L } = props.current;
    return {
      type: "FeatureCollection" as const,
      features: L.bundle.stations.map((s) => ({
        type: "Feature" as const,
        properties: { id: s.id, icon: s.dust ? "st-dust" : "st-clear", dust: s.dust ? 1 : 0 },
        geometry: { type: "Point" as const, coordinates: [s.lon, s.lat] },
      })),
    };
  }

  function nwsData() {
    const { loaded: L } = props.current;
    return {
      type: "FeatureCollection" as const,
      features: (L.bundle.nws ?? [])
        .filter((w) => w.polygon)
        .map((w) => ({
          type: "Feature" as const,
          properties: { type: w.type },
          geometry: { type: "Polygon" as const, coordinates: [w.polygon!] },
        })),
    };
  }

  /** Load the new forecast's maps into the idle buffer, then cross-fade to it. */
  function showBundle() {
    const m = map.current;
    if (!m || !ready.current) return;
    const L = props.current.loaded;
    if (shownT0.current === L.bundle.t0) return;
    const first = shownT0.current === null;
    shownT0.current = L.bundle.t0;
    const next = (first ? 0 : 1 - active.current) as 0 | 1;
    const urls = L.bundle.leads.map((lead) => resolve(L.root, L.bundle.layers[String(lead)]));
    Promise.all(
      urls.map((u) => new Promise<void>((ok) => { const im = new Image(); im.onload = () => ok(); im.onerror = () => ok(); im.src = u; })),
    ).then(() => {
      if (shownT0.current !== L.bundle.t0) return;                 // a newer forecast arrived meanwhile
      L.bundle.leads.forEach((lead, k) => {
        (m.getSource(`p${lead}${BUF[next]}`) as ImageSource | undefined)?.updateImage({ url: urls[k], coordinates: corners(L.bundle) });
      });
      active.current = next;
      if (first || props.current.reduceMotion) { blend.current = 1; applyOpacity(); return; }
      const t0 = performance.now();
      blend.current = 0;
      const step = (now: number) => {
        blend.current = Math.min(1, (now - t0) / 650);
        applyOpacity();
        if (blend.current < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
    (m.getSource("stations") as GeoJSONSource | undefined)?.setData(stationData());
    (m.getSource("nws") as GeoJSONSource | undefined)?.setData(nwsData());
  }

  // ---- create the map once
  useEffect(() => {
    if (!box.current) return;
    setWorkerUrl(new URL(WORKER, window.location.origin).toString());
    const m = new MLMap({
      container: box.current,
      style: STYLE,
      bounds: VIEW,
      fitBoundsOptions: { padding: 24 },
      attributionControl: { compact: true },
      dragRotate: false,
      pitchWithRotate: false,
      maxBounds: [[-118, 29], [-100, 38.5]],
      // phone width: the map is half the page, so one finger scrolls the page and two fingers move the map
      cooperativeGestures: window.matchMedia("(max-width: 880px)").matches,
    });
    m.touchZoomRotate.disableRotation();
    m.addControl(new NavigationControl({ showCompass: false }), "top-right");
    map.current = m;

    // "style.load", not "load": draw our layers as soon as the style is parsed, without waiting for every basemap tile
    m.once("style.load", () => {
      const L = props.current.loaded;
      const firstSymbol = m.getStyle().layers.find((l: LayerSpecification) => l.type === "symbol")?.id;
      const blank = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
      L.bundle.leads.forEach((lead) => {
        BUF.forEach((tag) => {
          m.addSource(`p${lead}${tag}`, { type: "image", url: blank, coordinates: corners(L.bundle) });
          m.addLayer({ id: `p${lead}${tag}`, type: "raster", source: `p${lead}${tag}`,
            paint: { "raster-opacity": 0, "raster-fade-duration": 0, "raster-resampling": "linear",
              "raster-opacity-transition": { duration: 0 } } }, firstSymbol);
        });
      });
      m.addSource("nws", { type: "geojson", data: nwsData() });
      m.addLayer({ id: "nws-fill", type: "fill", source: "nws", paint: { "fill-color": "#f1b640", "fill-opacity": 0.08 } }, firstSymbol);
      m.addLayer({ id: "nws-line", type: "line", source: "nws",
        paint: { "line-color": "#f1b640", "line-width": 1.5, "line-opacity": 0.85, "line-dasharray": [3, 2] } }, firstSymbol);

      m.addSource("road", { type: "geojson", data: roadData() });
      const w = matchExpr(ROAD_WIDTH, 2) as never;
      m.addLayer({ id: "road-casing", type: "line", source: "road", layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": "#ffffff", "line-opacity": 0.95, "line-width": ["+", w, 3] as never } });
      m.addLayer({ id: "road-sel", type: "line", source: "road", filter: ["==", ["get", "id"], ""],
        layout: { "line-cap": "round" }, paint: { "line-color": "#1b2a3a", "line-width": ["+", w, 7] as never } });
      const solid = ["clear", "watch", "severe"];
      m.addLayer({ id: "road-solid", type: "line", source: "road", filter: ["in", ["get", "state"], ["literal", solid]],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": matchExpr(ROAD_COLOR, "#888") as never, "line-width": w,
          "line-opacity": ["match", ["get", "state"], "clear", 0.7, 1] } });
      const dashed: [RoadState, number[]][] = [["held", [2, 1.4]], ["early", [0.9, 1.3]], ["unknown", [0.6, 1.8]]];
      dashed.forEach(([st, dash]) =>
        m.addLayer({ id: `road-${st}`, type: "line", source: "road", filter: ["==", ["get", "state"], st],
          layout: { "line-join": "round" },
          paint: { "line-color": ROAD_COLOR[st], "line-width": ROAD_WIDTH[st], "line-dasharray": dash } }));
      m.addLayer({ id: "road-ring", type: "line", source: "road", filter: ["in", ["get", "id"], ["literal", []]],
        layout: { "line-cap": "round" }, paint: { "line-color": ROAD_COLOR.severe, "line-width": 6, "line-opacity": 0, "line-blur": 3 } });
      m.addLayer({ id: "road-hit", type: "line", source: "road", paint: { "line-color": "#000", "line-opacity": 0, "line-width": 22 } });

      m.addImage("st-dust", diamond("#b5502e", "#ffffff", 30), { pixelRatio: 2 });
      m.addImage("st-clear", diamond("#ffffff", "#6b7775", 22), { pixelRatio: 2 });
      m.addSource("stations", { type: "geojson", data: stationData() });
      m.addLayer({ id: "stations", type: "symbol", source: "stations",
        layout: { "icon-image": ["get", "icon"], "icon-allow-overlap": true, "symbol-sort-key": ["get", "dust"] } });

      m.on("click", "road-hit", (e: MapLayerMouseEvent) => {
        const id = e.features?.[0]?.properties?.id as string | undefined;
        if (id) props.current.onSelect(id);
      });
      m.on("mouseenter", "road-hit", () => { m.getCanvas().style.cursor = "pointer"; });
      m.on("mouseleave", "road-hit", () => { m.getCanvas().style.cursor = ""; });

      ready.current = true;
      frameAlerts(m, L);
      showBundle();
      applyOpacity();
    });
    return () => { ready.current = false; m.remove(); map.current = null; shownT0.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- new forecast
  useEffect(() => { showBundle(); }, [loaded]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- look-ahead slider and staleness
  useEffect(() => { applyOpacity(); }, [minute, stale]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- road states and selection
  useEffect(() => {
    const m = map.current;
    if (!m || !ready.current) return;
    (m.getSource("road") as GeoJSONSource | undefined)?.setData(roadData());
  }, [states, loaded]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const m = map.current;
    if (!m || !ready.current) return;
    m.setFilter("road-sel", ["==", ["get", "id"], selected ?? ""]);
  }, [selected]);

  // ---- one ring per segment that just turned severe
  useEffect(() => {
    const m = map.current;
    if (!m || !ready.current || rings.length === 0 || reduceMotion) return;
    m.setFilter("road-ring", ["in", ["get", "id"], ["literal", rings]]);
    const t0 = performance.now();
    let raf = 0;
    const step = (now: number) => {
      const u = Math.min(1, (now - t0) / 1400);
      const ease = 1 - Math.pow(1 - u, 3);
      m.setPaintProperty("road-ring", "line-width", 6 + 26 * ease);
      m.setPaintProperty("road-ring", "line-opacity", 0.75 * (1 - u));
      if (u < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [rings, reduceMotion]);

  return <div ref={box} className={`map${stale ? " is-stale" : ""}`} role="region" aria-label="Map of dust chance and I-10 road states" />;
}
