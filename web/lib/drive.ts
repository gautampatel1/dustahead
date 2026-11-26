/**
 * "Your drive": when would a driver reach each stretch of I-10, and what is the chance of dust there at that time?
 *
 * Time on the road comes from the stretch lengths in the bundle (miles from exit numbers) and a steady speed.
 * The chance on a stretch is read at the times the driver is on it (entering, middle, leaving), from the forecast
 * maps at +15, +30 and +60 minutes after the forecast time t0:
 *   - before +15 the +15 map is the nearest forecast, so it is used;
 *   - between leads the chance is interpolated, as the map does;
 *   - after +60 there is no forecast, and the stretch is marked "beyond the forecast" (never "clear").
 * A stretch is a danger when that chance reaches the alert threshold for its tier. The first danger on the way gives
 * the advice: stop before the town where that stretch starts.
 */
import { alertThreshold, pAt, type Bundle, type RoadState, type Segment } from "./bundle.ts";

export interface Leg {
  seg: Segment;
  reverse: boolean;           // driving east -> west (the segment is stored west -> east)
  entry: string;              // town where the driver enters this stretch
  exit: string;
  entryMp: number;            // exit number at the entry town
  entryState: string;
  enterMin: number;           // minutes after t0
  exitMin: number;
  chance: number | null;      // highest chance while on it, null when entirely after the forecast hour
  danger: boolean;
  beyond: boolean;            // entered after the forecast hour ends
  stateNow: RoadState;
}

export interface Plan {
  legs: Leg[];
  miles: number;
  arriveMin: number;          // minutes after t0
  horizonMin: number;         // end of the forecast hour, minutes after t0
  firstDanger: Leg | null;
}

/** Towns along I-10, west -> east (the waypoints), with their positions. */
export function towns(b: Bundle): { name: string; lon: number; lat: number }[] {
  const first = b.segments[0];
  return [
    { name: first.from, lon: first.line[0][0], lat: first.line[0][1] },
    ...b.segments.map((s) => ({ name: s.to, lon: s.line[1][0], lat: s.line[1][1] })),
  ];
}

function chanceAt(seg: Segment, leads: number[], minute: number): number | null {
  if (minute > leads[leads.length - 1]) return null;
  return pAt(seg, leads, Math.max(minute, leads[0]));
}

function leadIndex(leads: number[], minute: number): number {
  const k = leads.findIndex((l) => minute <= l);
  return k < 0 ? leads.length - 1 : k;
}

/**
 * Plan a drive from town index `from` to town index `to` (indices into towns(b)), leaving `departMin` minutes after
 * the forecast time t0, at a steady `mph`. `states` are the current road states (for display only).
 */
export function planDrive(b: Bundle, from: number, to: number, departMin: number, mph: number,
                          states: Record<string, RoadState>): Plan {
  const horizonMin = b.leads[b.leads.length - 1];
  const reverse = to < from;
  const idx = reverse ? Array.from({ length: from - to }, (_, k) => from - 1 - k) : Array.from({ length: to - from }, (_, k) => from + k);
  const legs: Leg[] = [];
  let t = departMin;
  let miles = 0;
  for (const k of idx) {
    const seg = b.segments[k];
    const miles_k = seg.miles ?? Math.abs(seg.mp[1] - seg.mp[0]);
    const dur = (miles_k / mph) * 60;
    const enter = t, exit = t + dur;
    let chance: number | null = null;
    let danger = false;
    for (const m of [enter, (enter + exit) / 2, exit]) {
      const c = chanceAt(seg, b.leads, m);
      if (c === null) continue;
      chance = chance === null ? c : Math.max(chance, c);
      if (c >= alertThreshold(b, seg, leadIndex(b.leads, m))) danger = true;
    }
    const [st0, st1] = seg.state.split("/");
    legs.push({
      seg, reverse,
      entry: reverse ? seg.to : seg.from,
      exit: reverse ? seg.from : seg.to,
      entryMp: reverse ? seg.mp[1] : seg.mp[0],
      entryState: reverse ? (st1 ?? st0) : st0,
      enterMin: enter, exitMin: exit, chance, danger, beyond: enter > horizonMin,
      stateNow: states[seg.id] ?? "clear",
    });
    t = exit;
    miles += miles_k;
  }
  return { legs, miles, arriveMin: t, horizonMin, firstDanger: legs.find((l) => l.danger) ?? null };
}

/** Index of the town nearest to a position (for "start from where I am"). */
export function nearestTown(b: Bundle, lon: number, lat: number): { index: number; km: number } {
  const R = 6371, rad = Math.PI / 180;
  let best = { index: 0, km: Infinity };
  towns(b).forEach((tw, i) => {
    const dLat = (tw.lat - lat) * rad, dLon = (tw.lon - lon) * rad;
    const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat * rad) * Math.cos(tw.lat * rad) * Math.sin(dLon / 2) ** 2;
    const km = 2 * R * Math.asin(Math.sqrt(a));
    if (km < best.km) best = { index: i, km };
  });
  return best;
}
