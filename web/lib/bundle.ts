/**
 * The forecast bundle written by scripts/35_publish.py (dustahead/app/bundle.py, schema 1), and the few rules the
 * app applies to it. The app never re-thresholds P: the road state comes from the worker's alert list
 * (road_alerts + AlertTracker), so the web map, the phone and the pushes always agree.
 *
 * Terms: P = probability of dust; lead = minutes after the forecast time t0; MP = milepost;
 * ASOS = Automated Surface Observing System; NWS = National Weather Service.
 */

export type Level = "watch" | "severe";
export type RoadState = "clear" | "watch" | "severe" | "held" | "early" | "unknown";
export type Source = "live" | "replay" | "fixture";

export interface Segment {
  id: string;
  name: string;
  from: string;
  to: string;
  state: string; // AZ, NM, TX or AZ/NM
  mp: [number, number];
  line: [number, number][];
  p: number[]; // max P along the segment at each lead
  genesis: boolean; // no station within 50 km reports dust now
}

export interface Alert {
  segment: string;
  name: string;
  lead_min: number;
  p_max: number;
  level: Level;
  genesis: boolean;
  held: boolean;
  /** "nearby" = a station within 50 km reports dust now; "new_storm" = none does (product.tiers). null when off. */
  tier?: Tier | null;
  /** Share of past alerts in this tier followed by dust within 60 min; null when not measured for this model. */
  hit_rate?: number | null;
}

export type Tier = "nearby" | "new_storm";

export interface TierPolicy {
  thresholds: number[];
  hit_rate: number | null;
  coverage: number | null;
}

export interface Station {
  id: string;
  lon: number;
  lat: number;
  dust: boolean;
}

export interface NwsProduct {
  type: string;
  vtec: string;
  area: string;
  onset: string | null;
  expires: string | null;
  polygon: [number, number][] | null;
}

export interface Bundle {
  schema: number;
  t0: string;
  issued: string;
  source: Source;
  model: string;
  leads: number[];
  bounds: [number, number, number, number];
  layers: Record<string, string>;
  policy: {
    thresholds: number[] | null;
    severe_margin: number;
    release: number;
    min_hold_min: number;
    stale_after_min: number;
    genesis_km: number;
    tiers?: { rated: boolean; nearby: TierPolicy; new_storm: TierPolicy } | null;
  };
  segments: Segment[];
  alerts: Alert[];
  stations: Station[];
  dust_now: string[];
  nws: NwsProduct[] | null; // null = feed could not be read; [] = none active
}

export interface Replay {
  schema: number;
  title: string;
  source: Source;
  frames: string[];
}

/** A bundle plus the URL it came from, so relative layer paths resolve against the publish root. */
export interface Loaded {
  bundle: Bundle;
  root: string; // URL of the publish root, ending in '/'
}

export function resolve(root: string, rel: string): string {
  return new URL(rel, root).toString();
}

async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return (await r.json()) as T;
}

export async function loadLatest(root: string): Promise<Loaded> {
  return { bundle: await getJSON<Bundle>(resolve(root, "latest.json")), root };
}

export async function loadReplay(root: string): Promise<{ replay: Replay; frames: Loaded[] }> {
  const replay = await getJSON<Replay>(resolve(root, "replay.json"));
  const frames = await Promise.all(
    replay.frames.map(async (f) => ({ bundle: await getJSON<Bundle>(resolve(root, f)), root })),
  );
  return { replay, frames };
}

/** Minutes between the forecast time and `now`. */
export function ageMin(b: Bundle, now: number): number {
  return (now - Date.parse(b.t0)) / 60000;
}

export function isStale(b: Bundle, now: number): boolean {
  return ageMin(b, now) > b.policy.stale_after_min;
}

/** Road state for drawing. A stale forecast makes every segment "unknown", never "clear". */
export function roadState(a: Alert | undefined, stale: boolean): RoadState {
  if (stale) return "unknown";
  if (!a) return "clear";
  if (a.held) return "held";
  if (a.tier === "new_storm" || a.genesis) return "early";
  return a.level;
}

export const STATE_RANK: Record<RoadState, number> = { severe: 0, watch: 1, early: 2, held: 3, unknown: 4, clear: 5 };

/** The alert threshold for one segment and lead: the tier's when tiers are on (new-storm tier when no station
 *  within 50 km reports dust), else the per-lead product threshold. Used only to highlight numbers. */
export function alertThreshold(b: Bundle, seg: Segment, k: number): number {
  const T = b.policy.tiers;
  if (T) return (seg.genesis ? T.new_storm : T.nearby).thresholds[k] ?? 0.5;
  return b.policy.thresholds?.[k] ?? 0.8;
}

/** "About 1 in 7": the hit rate as a whole-number ratio, or null when not measured. */
export function oneIn(rate: number | null | undefined): number | null {
  return rate ? Math.max(1, Math.round(1 / rate)) : null;
}

/** P at an arbitrary look-ahead, linear between the published leads (the map blends the same way). */
export function pAt(seg: Segment, leads: number[], minute: number): number {
  if (minute <= leads[0]) return seg.p[0];
  for (let k = 1; k < leads.length; k++) {
    if (minute <= leads[k]) {
      const u = (minute - leads[k - 1]) / (leads[k] - leads[k - 1]);
      return seg.p[k - 1] + u * (seg.p[k] - seg.p[k - 1]);
    }
  }
  return seg.p[seg.p.length - 1];
}

/** Opacity of each lead's map at a look-ahead minute: the two nearest maps cross-fade. */
export function layerWeights(leads: number[], minute: number): number[] {
  const w = leads.map(() => 0);
  if (minute <= leads[0]) w[0] = 1;
  else if (minute >= leads[leads.length - 1]) w[leads.length - 1] = 1;
  else
    for (let k = 1; k < leads.length; k++)
      if (minute <= leads[k]) {
        const u = (minute - leads[k - 1]) / (leads[k] - leads[k - 1]);
        w[k - 1] = 1 - u;
        w[k] = u;
        break;
      }
  return w;
}

/** Ray-casting point-in-polygon on lon/lat (fine at county scale). */
function inside(pt: [number, number], poly: [number, number][]): boolean {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > pt[1] !== yj > pt[1] && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}

/** NWS products whose polygon touches this segment (endpoints or midpoint inside). */
export function warningsFor(seg: Segment, nws: NwsProduct[] | null): NwsProduct[] {
  if (!nws) return [];
  const [a, b] = seg.line;
  const pts: [number, number][] = [a, b, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]];
  return nws.filter((w) => w.polygon && pts.some((p) => inside(p, w.polygon!)));
}

/** Names for the ASOS stations along I-10 (falls back to the id). */
export const STATION_NAMES: Record<string, string> = {
  TUS: "Tucson", MZJ: "Marana", AVQ: "Marana", CGZ: "Casa Grande", PHX: "Phoenix", IWA: "Mesa", CHD: "Chandler",
  GYR: "Goodyear", BXK: "Buckeye", DMA: "Davis-Monthan", SAD: "Safford", P33: "Willcox", LSB: "Lordsburg",
  DMN: "Deming", LRU: "Las Cruces", ELP: "El Paso", BIF: "Fort Bliss"
};
