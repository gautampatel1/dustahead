"use client";
/**
 * App shell: data source, clock, staleness, settings and tabs.
 *
 * Data source (first match wins): ?src=<publish root>, NEXT_PUBLIC_DATA_URL (the R2 bucket), else the demo fixture in
 * public/demo/. Live mode polls latest.json every 60 s; ?replay=1 (or the demo) steps through replay.json instead.
 * Staleness is worked out from the forecast time itself, so "no new forecast" shows even if the whole pipeline is down.
 * In a replay the clock is the replayed forecast's publish time (plus 31 min when the stale demo switch is on).
 * Tabs are hash links (#map, #roads, #warnings, #safety, #about) so the static export needs no router.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ageMin, isStale, loadLatest, loadReplay, roadState, type Loaded, type Replay, type RoadState } from "@/lib/bundle";
import { COPY, clockShort, type Lang } from "@/lib/i18n";
import MapTab from "./MapTab";
import { AboutTab, RoadsTab, SafetyTab, WarningsTab } from "./Tabs";

export type Tab = "map" | "roads" | "warnings" | "safety" | "about";
const TABS: Tab[] = ["map", "roads", "warnings", "safety", "about"];

function storageGet(k: string): string | null {
  try { return window.localStorage.getItem(k); } catch { return null; }
}
function storageSet(k: string, v: string) {
  try { window.localStorage.setItem(k, v); } catch { /* private mode */ }
}

interface Setup { root: string; replay: boolean }

function readSetup(): Setup {
  const q = new URLSearchParams(window.location.search);
  const env = process.env.NEXT_PUBLIC_DATA_URL;
  const src = q.get("src") || env || "demo/";
  const root = new URL(src.endsWith("/") ? src : src + "/", window.location.href).toString();
  const isDemo = !q.get("src") && !env;
  return { root, replay: q.get("replay") === "1" || (isDemo && q.get("replay") !== "0") };
}

const TZ_OPTIONS = ["America/Phoenix", "America/Denver", "device"] as const;

export default function LiveApp() {
  const [setup, setSetup] = useState<Setup | null>(null);
  const [frames, setFrames] = useState<Loaded[]>([]);
  const [replayInfo, setReplayInfo] = useState<Replay | null>(null);
  const [idx, setIdx] = useState(0);
  const [failed, setFailed] = useState(false);
  const [lead, setLead] = useState(15);
  const [replaying, setReplaying] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [simStale, setSimStale] = useState(false);
  const [wall, setWall] = useState(() => Date.now());
  const [lang, setLang] = useState<Lang>("en");
  const [tzPick, setTzPick] = useState<(typeof TZ_OPTIONS)[number]>("America/Phoenix");
  const [tab, setTab] = useState<Tab>("map");
  const [reduceMotion, setReduceMotion] = useState(false);
  const prevStates = useRef<{ t0: string; states: Record<string, RoadState> } | null>(null);
  const [rings, setRings] = useState<string[]>([]);

  // ---- setup, settings, tab from the URL hash
  useEffect(() => {
    setSetup(readSetup());
    const q = new URLSearchParams(window.location.search);
    const l = q.get("lang") || storageGet("dustahead.lang") || (navigator.language?.startsWith("es") ? "es" : "en");
    setLang(l === "es" ? "es" : "en");
    const tz = q.get("tz") || storageGet("dustahead.tz");
    if (tz && (TZ_OPTIONS as readonly string[]).includes(tz)) setTzPick(tz as (typeof TZ_OPTIONS)[number]);
    try { setReduceMotion(window.matchMedia("(prefers-reduced-motion: reduce)").matches); } catch { /* old browser */ }
    const fromHash = () => {
      const h = window.location.hash.slice(1) as Tab;
      setTab(TABS.includes(h) ? h : "map");
      window.scrollTo(0, 0);
    };
    fromHash();
    window.addEventListener("hashchange", fromHash);
    return () => window.removeEventListener("hashchange", fromHash);
  }, []);

  useEffect(() => {
    if (!setup) return;
    let alive = true;
    if (setup.replay) {
      loadReplay(setup.root)
        .then(({ replay, frames }) => { if (alive) { setReplayInfo(replay); setFrames(frames); setIdx(0); } })
        .catch(() => alive && setFailed(true));
      return () => { alive = false; };
    }
    const pull = () =>
      loadLatest(setup.root)
        .then((l) => { if (!alive) return; setFailed(false); setFrames((f) => (f[0]?.bundle.t0 === l.bundle.t0 ? f : [l])); })
        .catch(() => alive && setFailed(true));
    pull();
    const iv = setInterval(pull, 60_000);
    const vis = () => document.visibilityState === "visible" && pull();
    document.addEventListener("visibilitychange", vis);
    return () => { alive = false; clearInterval(iv); document.removeEventListener("visibilitychange", vis); };
  }, [setup]);

  useEffect(() => {
    const iv = setInterval(() => setWall(Date.now()), 15_000);
    return () => clearInterval(iv);
  }, []);

  // ---- replay autoplay: one forecast every 2.4 s
  useEffect(() => {
    if (!replaying || frames.length < 2) return;
    const iv = setInterval(() => setIdx((i) => Math.min(frames.length - 1, i + 1)), 2400);
    return () => clearInterval(iv);
  }, [replaying, frames.length]);
  useEffect(() => { if (replaying && idx >= frames.length - 1) setReplaying(false); }, [replaying, idx, frames.length]);

  useEffect(() => { document.documentElement.lang = lang; }, [lang]);

  const cur = frames[Math.min(idx, frames.length - 1)];
  const b = cur?.bundle;
  const isLive = !!b && b.source === "live" && !setup?.replay;
  const now = !b || isLive ? wall : Date.parse(b.issued) + (simStale ? 31 * 60_000 : 0);
  const stale = !!b && isStale(b, now);
  const age = b ? ageMin(b, now) : 0;
  const t = COPY[lang];
  const tz = tzPick === "device" ? undefined : tzPick;

  const states = useMemo(() => {
    const out: Record<string, RoadState> = {};
    if (!b) return out;
    const byId = Object.fromEntries(b.alerts.map((a) => [a.segment, a]));
    b.segments.forEach((s) => { out[s.id] = roadState(byId[s.id], stale); });
    return out;
  }, [b, stale]);

  // a ring for each segment that became severe since the previous forecast
  useEffect(() => {
    if (!b) return;
    const prev = prevStates.current;
    if (prev && prev.t0 !== b.t0) setRings(Object.keys(states).filter((id) => states[id] === "severe" && prev.states[id] !== "severe"));
    prevStates.current = { t0: b.t0, states };
  }, [b, states]);

  const activeNws = b?.nws?.filter((w) => !w.expires || Date.parse(w.expires) > now) ?? [];

  const changeLang = useCallback((l: Lang) => { setLang(l); storageSet("dustahead.lang", l); }, []);
  const changeTz = useCallback((v: (typeof TZ_OPTIONS)[number]) => { setTzPick(v); storageSet("dustahead.tz", v); }, []);
  const openSegment = useCallback((id: string) => { setSelected(id); window.location.hash = "map"; }, []);

  return (
    <div className="app">
      <header className="top">
        <div className="top-row">
          <a className="brand" href="#map">
            <svg viewBox="0 0 28 28" fill="none" aria-hidden="true"><rect width="28" height="28" rx="7" fill="#b5502e" /><path d="M5 17c3.4-1.2 5.6-4.4 9.6-4.4S20 15.8 23.4 15" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" /><path d="M7 21.6c2.2-.7 4.4-2.2 7.4-2.2s4.4 1.7 7.4 1" stroke="#f3c9a0" strokeWidth="2.4" strokeLinecap="round" /><path d="M7.6 10.4c2.2-2.2 4.4-3.3 7.6-3.3 2.2 0 3.8.9 5.4 2.2" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" /></svg>
            <span>DustAhead</span>
            <small>I-10 Arizona · New Mexico</small>
          </a>
          {b && (
            <span className={`updated${stale ? " is-stale" : ""}`}>
              <i aria-hidden="true" />
              {stale ? t.noNewForecast(clockShort(b.t0, lang, tz)) : t.updated(clockShort(b.issued, lang, tz))}
            </span>
          )}
          <div className="settings">
            <label>
              <span className="sr">{t.timeZone}</span>
              <select value={tzPick} onChange={(e) => changeTz(e.target.value as (typeof TZ_OPTIONS)[number])} aria-label={t.timeZone}>
                <option value="America/Phoenix">{t.tzAZ}</option>
                <option value="America/Denver">{t.tzMT}</option>
                <option value="device">{t.tzDevice}</option>
              </select>
            </label>
            <label>
              <span className="sr">{t.language}</span>
              <select value={lang} onChange={(e) => changeLang(e.target.value as Lang)} aria-label={t.language}>
                <option value="en">English</option>
                <option value="es">Español</option>
              </select>
            </label>
          </div>
        </div>
        <nav className="tabs" aria-label="Sections">
          {TABS.map((k) => (
            <a key={k} href={`#${k}`} className={tab === k ? "on" : ""} aria-current={tab === k ? "page" : undefined}>
              {t.tabs[k]}
              {k === "warnings" && activeNws.length > 0 && <span className="dot-count">{activeNws.length}</span>}
            </a>
          ))}
        </nav>
      </header>

      {b && b.source === "fixture" && <div className="notice demo">{t.demoFixture}</div>}
      {b && b.source === "replay" && <div className="notice demo">{t.demoReplay(replayInfo?.title ?? "")}</div>}
      {b && activeNws.length > 0 && tab !== "warnings" && (
        <a className="notice nws" href="#warnings">
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.5 15 14H1z" fill="currentColor" /></svg>
          {t.nwsActive(activeNws[0].type, activeNws[0].area, activeNws[0].expires ? clockShort(activeNws[0].expires, lang, tz) : "?")}
          {activeNws.length > 1 && ` (+${activeNws.length - 1})`}
        </a>
      )}

      <main className={`page page-${tab}`}>
        {!b && <div className="empty-page">{failed ? t.loadFail : t.loading}</div>}
        {b && tab === "map" && (
          <MapTab cur={cur} states={states} rings={rings} stale={stale} age={age} lead={lead} setLead={setLead}
            selected={selected} setSelected={setSelected} t={t} lang={lang} tz={tz} reduceMotion={reduceMotion}
            replay={!isLive && frames.length > 1 ? {
              frames, idx, setIdx: (i: number) => { setReplaying(false); setIdx(i); }, playing: replaying,
              toggle: () => { if (idx >= frames.length - 1) setIdx(0); setReplaying((v) => !v); },
              simStale, toggleStale: () => setSimStale((v) => !v),
            } : null} />
        )}
        {b && tab === "roads" && <RoadsTab b={b} states={states} stale={stale} t={t} onOpen={openSegment} />}
        {b && tab === "warnings" && <WarningsTab b={b} active={activeNws} t={t} lang={lang} tz={tz} />}
        {tab === "safety" && <SafetyTab t={t} />}
        {tab === "about" && <AboutTab b={b} stale={stale} age={age} t={t} lang={lang} tz={tz} />}
      </main>

      <footer className="foot">{t.disclaimer}</footer>
    </div>
  );
}
