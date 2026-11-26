"use client";
/** Map tab: the map, a 15/30/60-minute switch, the road alert list, a segment's details, and station reports. */
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { STATE_RANK, STATION_NAMES, alertThreshold, oneIn, unplacedWarnings, warningsFor, type Alert, type Loaded, type RoadState } from "@/lib/bundle";
import { clockShort, type Copy, type Lang } from "@/lib/i18n";
import { ROAD_COLOR } from "./MapView";

const MapView = dynamic(() => import("./MapView"), { ssr: false, loading: () => <div className="map map-loading" /> });

interface ReplayCtl {
  frames: Loaded[]; idx: number; setIdx: (i: number) => void; playing: boolean; toggle: () => void;
  simStale: boolean; toggleStale: () => void;
}

interface Props {
  cur: Loaded; states: Record<string, RoadState>; rings: string[]; stale: boolean; age: number;
  lead: number; setLead: (m: number) => void; selected: string | null; setSelected: (id: string | null) => void;
  t: Copy; lang: Lang; tz?: string; reduceMotion: boolean; replay: ReplayCtl | null; visible: boolean;
}

export function StateLine({ state }: { state: RoadState }) {
  const dash = state === "held" ? "6 4" : state === "early" ? "3 3" : state === "unknown" ? "1.5 4" : undefined;
  const w = state === "severe" ? 6 : state === "watch" ? 5 : state === "clear" ? 3 : 4;
  return (
    <svg className="state-line" viewBox="0 0 30 10" aria-hidden="true">
      <line x1="2" y1="5" x2="28" y2="5" stroke={ROAD_COLOR[state]} strokeWidth={w} strokeDasharray={dash} strokeLinecap={dash ? "butt" : "round"} />
    </svg>
  );
}

export function Badge({ state, t }: { state: RoadState; t: Copy }) {
  return <span className={`badge s-${state}`}>{t.state[state]}</span>;
}

/** Map key: every road state, the two station markers, the NWS outline and the dust scale. Open by default on wide
 *  screens; on a phone it folds to one "Map key" line so it doesn't cover the storm. */
function Legend({ t }: { t: Copy }) {
  const [open, setOpen] = useState(true);
  useEffect(() => {
    try { if (window.matchMedia("(max-width: 900px)").matches) setOpen(false); } catch { /* old browser */ }
  }, []);
  return (
    <details className="legend" open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>{t.mapKey}</summary>
      <div className="legend-row">
        {(["severe", "watch", "early", "held", "clear", "unknown"] as const).map((k) => (
          <span key={k}><StateLine state={k} />{t.state[k]}</span>
        ))}
      </div>
      <div className="legend-row">
        <span><i className="dia dust" aria-hidden="true" />{t.keyStationDust}</span>
        <span><i className="dia clear" aria-hidden="true" />{t.keyStationClear}</span>
        <span><i className="key-nws" aria-hidden="true" />{t.keyNws}</span>
      </div>
      <div className="legend-row ramp-row">
        <span>{t.legendChance}</span><span className="ramp-txt">{t.low}</span><i className="ramp" /><span className="ramp-txt">{t.high}</span>
      </div>
    </details>
  );
}

export default function MapTab(p: Props) {
  const { cur, states, stale, t, lang, tz } = p;
  const b = cur.bundle;
  const byId = Object.fromEntries(b.alerts.map((a) => [a.segment, a]));
  const rows = b.segments
    .filter((s) => byId[s.id])
    .map((s) => ({ seg: s, alert: byId[s.id], state: states[s.id] }))
    .sort((x, y) => STATE_RANK[x.state] - STATE_RANK[y.state] || x.alert.lead_min - y.alert.lead_min);
  const seg = p.selected ? b.segments.find((s) => s.id === p.selected) : undefined;
  const leadTime = clockShort(Date.parse(b.t0) + p.lead * 60_000, lang, tz);

  return (
    <div className="map-page">
      <section className="map-area">
        <div className="map-wrap">
          <MapView loaded={cur} minute={p.lead} states={states} selected={p.selected} rings={p.rings} stale={stale}
            reduceMotion={p.reduceMotion} visible={p.visible} onSelect={p.setSelected}
            stationLabels={{ dust: t.stationDustNow, clear: t.stationClearNow }} />
          <Legend t={t} />
        </div>

        <div className="lead-bar">
          <span className="lead-label">{t.showChanceIn}</span>
          <div className="seg-buttons" role="group" aria-label={t.showChanceIn}>
            {b.leads.map((m) => (
              <button key={m} type="button" className={p.lead === m ? "on" : ""} aria-pressed={p.lead === m} onClick={() => p.setLead(m)}>
                {t.minutes(m)}
              </button>
            ))}
          </div>
          <span className="lead-time">{t.atTime(leadTime)}</span>
        </div>

        {p.replay && (
          <div className="replay-bar">
            <button type="button" className="btn" onClick={() => p.replay!.setIdx(Math.max(0, p.replay!.idx - 1))} disabled={p.replay.idx === 0}>‹ {t.replayPrev}</button>
            <button type="button" className="btn btn-primary" onClick={p.replay.toggle}>{p.replay.playing ? t.replayPause : t.replayPlay}</button>
            <button type="button" className="btn" onClick={() => p.replay!.setIdx(Math.min(p.replay!.frames.length - 1, p.replay!.idx + 1))}
              disabled={p.replay.idx >= p.replay.frames.length - 1}>{t.replayNext} ›</button>
            <input type="range" min={0} max={p.replay.frames.length - 1} value={p.replay.idx} aria-label={t.forecastAt(clockShort(b.t0, lang, tz))}
              onChange={(e) => p.replay!.setIdx(Number(e.target.value))} />
            <span className="replay-time">{t.forecastAt(clockShort(b.t0, lang, tz))}</span>
            <label className="check">
              <input type="checkbox" checked={p.replay.simStale} onChange={p.replay.toggleStale} /> {t.simulateStale}
            </label>
          </div>
        )}
      </section>

      <aside className="panel">
        {stale && (
          <div className="alert-box grey" role="alert">
            <strong>{t.staleTitle(p.age)}</strong>
            <p>{t.staleBody}</p>
          </div>
        )}

        {seg ? (
          <SegmentDetails seg={seg} state={states[seg.id]} alert={byId[seg.id]} cur={cur} t={t} onBack={() => p.setSelected(null)} />
        ) : (
          <>
            {!stale && (
              <div className={`summary${rows.length ? " has" : ""}`}>
                <p>{rows.length ? t.summarySome(rows.length) : t.summaryNone}</p>
                {rows.length > 0 && b.policy.tiers && (
                  <p className="summary-sub">
                    {t.summaryBreakdown(rows.filter((r) => r.state !== "early").length, rows.filter((r) => r.state === "early").length)}
                  </p>
                )}
              </div>
            )}
            {rows.length > 0 && (
              <>
                <h2 className="panel-h">{t.roadAlerts}</h2>
                <ul className="alert-list">
                  {rows.map((r) => (
                    <li key={`${r.seg.id}-${r.state}`}>
                      <button type="button" onClick={() => p.setSelected(r.seg.id)}>
                        <span className="al-top">
                          <span className="al-name">{r.seg.name}</span>
                          <Badge state={r.state} t={t} />
                        </span>
                        <span className="al-sub">
                          {r.state === "unknown" ? t.state.unknown : `${t.dustWhen(r.alert.lead_min)} · ${listDetail(r.alert, r.state, t)}`}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </>
        )}

        <Stations cur={cur} t={t} lang={lang} tz={tz} />
      </aside>
    </div>
  );
}

function SegmentDetails({ seg, state, alert, cur, t, onBack }: {
  seg: Loaded["bundle"]["segments"][number]; state: RoadState; alert?: Loaded["bundle"]["alerts"][number]; cur: Loaded; t: Copy; onBack: () => void;
}) {
  const b = cur.bundle;
  const warn = warningsFor(seg, b.nws);
  return (
    <section className="details">
      <button type="button" className="link-btn" onClick={onBack}>‹ {t.backToList}</button>
      <div className="details-head">
        <h2>{seg.name}</h2>
        <Badge state={state} t={t} />
      </div>
      <p className="details-help">{t.stateHelp[state]}</p>
      {alert && state !== "unknown" && <p className="details-when">{t.dustWhen(alert.lead_min)}</p>}
      {alert && state !== "unknown" && <Reliability alert={alert} cur={cur} t={t} />}
      <table className="chance-table">
        <caption>{t.chanceOfDust}</caption>
        <thead><tr>{b.leads.map((m) => <th key={m} scope="col">{t.inMin(m)}</th>)}</tr></thead>
        <tbody><tr>{seg.p.map((v, k) => <td key={k} className={v >= alertThreshold(b, seg, k) ? "hi" : ""}>{Math.round(v * 100)}%</td>)}</tr></tbody>
      </table>
      <dl className="facts">
        <div><dt>{t.mileposts}</dt><dd>{seg.state} {seg.mp[0]}–{seg.mp[1]}</dd></div>
        <div><dt>{t.reportedNearby}</dt><dd>{seg.genesis ? t.no : t.yes}</dd></div>
        <div><dt>{t.nwsWarning}</dt><dd>{b.nws === null ? t.unavailable : warn.length ? warn.map((w) => w.type).join(", ") : t.none}</dd></div>
      </dl>
      {unplacedWarnings(b.nws).map((w, i) => (
        <p key={i} className="details-rel rel-low">{t.mayApply(w.type, w.area)}</p>
      ))}
    </section>
  );
}

/** List line after the timing: a new-storm alert states how often such alerts came true, not a probability. */
function listDetail(a: Alert, state: RoadState, t: Copy): string {
  const n = oneIn(a.hit_rate);
  return state === "early" && n ? t.verifyShort(n) : `${Math.round(a.p_max * 100)}%`;
}

/** How often alerts of this tier were followed by dust (2025 validation), shown only when measured for this model. */
function Reliability({ alert, cur, t }: { alert: Alert; cur: Loaded; t: Copy }) {
  const T = cur.bundle.policy.tiers;
  if (!T || !alert.tier) return null;
  if (!T.rated || alert.hit_rate == null) return <p className="details-rel muted">{t.reliabilityNone}</p>;
  const cov = Math.round(100 * (T[alert.tier].coverage ?? 0));
  return (
    <p className={`details-rel ${alert.tier === "new_storm" ? "rel-low" : "rel-high"}`}>
      {alert.tier === "nearby" ? t.reliabilityNearby(Math.round(100 * alert.hit_rate), cov) : t.reliabilityNew(oneIn(alert.hit_rate)!, cov)}
    </p>
  );
}

function Stations({ cur, t, lang, tz }: { cur: Loaded; t: Copy; lang: Lang; tz?: string }) {
  const b = cur.bundle;
  const name = (id: string) => STATION_NAMES[id] ?? id;
  const dust = b.stations.filter((s) => s.dust).map((s) => name(s.id));
  const clear = b.stations.filter((s) => !s.dust).map((s) => name(s.id));
  return (
    <section className="stations">
      <h3>{t.stationsTitle(clockShort(b.t0, lang, tz))}</h3>
      {!b.stations.length && <p>{t.stationsNone}</p>}
      {dust.length > 0 && <p><i className="dia dust" aria-hidden="true" /><strong>{t.stationsDust}</strong> {dust.join(", ")}</p>}
      {clear.length > 0 && <p><i className="dia clear" aria-hidden="true" /><strong>{t.stationsClear}</strong> {clear.slice(0, 10).join(", ")}{clear.length > 10 ? ` +${clear.length - 10}` : ""}</p>}
    </section>
  );
}
