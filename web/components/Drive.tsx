"use client";
/**
 * "Your drive" card (Road conditions tab): pick start, destination, when you leave and how fast you drive, and see
 * when you'd reach each stretch and the chance of dust then. "Start from where I am" uses the browser's location
 * once to pick the nearest town; the position never leaves the device. Choices are remembered in this browser only.
 */
import { useEffect, useMemo, useState } from "react";
import type { Bundle, RoadState } from "@/lib/bundle";
import { nearestTown, planDrive, towns } from "@/lib/drive";
import { clockShort, type Copy, type Lang } from "@/lib/i18n";
import { Badge, StateLine } from "./MapTab";

const LEAVE = [0, 15, 30, 45];
const SPEEDS = [55, 65, 75];
const KEY = "dustahead.drive";

interface Saved { from: string; to: string; leave: number; mph: number }

function load(): Partial<Saved> {
  try { return JSON.parse(window.localStorage.getItem(KEY) || "{}"); } catch { return {}; }
}
function save(s: Saved) {
  try { window.localStorage.setItem(KEY, JSON.stringify(s)); } catch { /* private mode */ }
}

export default function DriveCard({ b, states, stale, now, t, lang, tz, onOpen }: {
  b: Bundle; states: Record<string, RoadState>; stale: boolean; now: number; t: Copy; lang: Lang; tz?: string;
  onOpen: (id: string) => void;
}) {
  const list = useMemo(() => towns(b), [b]);
  const names = list.map((x) => x.name);
  const [from, setFrom] = useState(Math.max(0, names.indexOf("Phoenix")));
  const [to, setTo] = useState(names.indexOf("Tucson") >= 0 ? names.indexOf("Tucson") : names.length - 1);
  const [leave, setLeave] = useState(0);
  const [mph, setMph] = useState(65);
  const [loc, setLoc] = useState<"idle" | "busy" | "fail" | "far">("idle");

  useEffect(() => {        // restore the last choices (names, so they survive a change in the town list)
    const s = load();
    if (s.from && names.includes(s.from)) setFrom(names.indexOf(s.from));
    if (s.to && names.includes(s.to)) setTo(names.indexOf(s.to));
    if (s.leave !== undefined && LEAVE.includes(s.leave)) setLeave(s.leave);
    if (s.mph && SPEEDS.includes(s.mph)) setMph(s.mph);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => { save({ from: names[from], to: names[to], leave, mph }); }, [from, to, leave, mph, names]);

  const useLocation = () => {
    if (!navigator.geolocation) { setLoc("fail"); return; }
    setLoc("busy");
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const n = nearestTown(b, pos.coords.longitude, pos.coords.latitude);
        if (n.km > 80) { setLoc("far"); return; }          // not near I-10 in the domain: don't guess
        setFrom(n.index);
        if (n.index === to) setTo(n.index > 0 ? n.index - 1 : n.index + 1);
        setLoc("idle");
      },
      () => setLoc("fail"),
      { enableHighAccuracy: false, timeout: 10_000, maximumAge: 300_000 },
    );
  };

  const t0 = Date.parse(b.t0);
  const departMin = (now - t0) / 60_000 + leave;
  const plan = from !== to ? planDrive(b, from, to, departMin, mph, states) : null;
  const at = (min: number) => clockShort(t0 + min * 60_000, lang, tz);
  const dur = plan ? Math.round(plan.arriveMin - departMin) : 0;

  let summary: { cls: string; text: string } | null = null;
  if (plan) {
    if (stale) summary = { cls: "grey", text: t.driveStale };
    else if (plan.firstDanger) {
      const d = plan.firstDanger;
      const first = d === plan.legs[0];
      summary = {
        cls: "warn",
        text: `${t.driveDanger(d.reverse ? `${d.entry} → ${d.exit}` : d.seg.name, at(d.enterMin), Math.round(100 * (d.chance ?? 0)))} `
          + (first ? t.driveStartBad : t.driveStop(d.entry, d.entryState, d.entryMp)),
      };
    } else {
      summary = {
        cls: plan.arriveMin > plan.horizonMin ? "grey" : "ok",
        text: t.driveClear(names[to], at(plan.arriveMin)) + (plan.arriveMin > plan.horizonMin ? ` ${t.driveBeyond(at(plan.horizonMin))}` : ""),
      };
    }
  }

  return (
    <section className="card">
      <h2>{t.driveTitle}</h2>
      <div className="route-form">
        <label>{t.from}
          <select value={from} onChange={(e) => setFrom(Number(e.target.value))}>
            {names.map((n, i) => <option key={n} value={i}>{n}</option>)}
          </select>
        </label>
        <label>{t.to}
          <select value={to} onChange={(e) => setTo(Number(e.target.value))}>
            {names.map((n, i) => <option key={n} value={i}>{n}</option>)}
          </select>
        </label>
        <label>{t.leave}
          <select value={leave} onChange={(e) => setLeave(Number(e.target.value))}>
            {LEAVE.map((m) => <option key={m} value={m}>{m === 0 ? t.leaveNow : t.leaveIn(m)}</option>)}
          </select>
        </label>
        <label>{t.speed}
          <select value={mph} onChange={(e) => setMph(Number(e.target.value))}>
            {SPEEDS.map((v) => <option key={v} value={v}>{t.mph(v)}</option>)}
          </select>
        </label>
      </div>
      <p className="loc-row">
        <button type="button" className="btn" onClick={useLocation} disabled={loc === "busy"}>
          {loc === "busy" ? t.locating : t.useLocation}
        </button>
        <span className="muted">
          {loc === "fail" ? t.locFail : loc === "far" ? t.locFar : t.locNote}
        </span>
      </p>

      {plan && summary && (
        <>
          <p className={`route-result ${summary.cls}`}>{summary.text}</p>
          <p className="muted drive-meta">{t.driveMeta(plan.miles, dur, at(departMin))}</p>
          <ol className="drive-legs">
            {plan.legs.map((l) => (
              <li key={l.seg.id} className={l.danger && !stale ? "danger" : ""}>
                <button type="button" onClick={() => onOpen(l.seg.id)}>
                  <span className="dl-time">{at(l.enterMin)}</span>
                  <StateLine state={l.stateNow} />
                  <span className="dl-name">
                    {l.reverse ? `${l.entry} → ${l.exit}` : l.seg.name}
                    <small>
                      {stale ? t.state.unknown
                        : l.beyond ? t.afterForecast
                        : l.danger ? t.dustThen(Math.round(100 * (l.chance ?? 0)))
                        : t.chanceThen(Math.round(100 * (l.chance ?? 0)))}
                    </small>
                  </span>
                  <Badge state={l.stateNow} t={t} />
                </button>
              </li>
            ))}
          </ol>
          <p className="note">{t.driveNote}</p>
        </>
      )}
    </section>
  );
}
