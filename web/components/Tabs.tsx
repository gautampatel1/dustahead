"use client";
/** The other tabs: road conditions (route check + full table), NWS warnings, safety advice, about + forecast status. */
import { useState } from "react";
import { alertThreshold, type Bundle, type NwsProduct, type RoadState } from "@/lib/bundle";
import { clockShort, dateLong, type Copy, type Lang } from "@/lib/i18n";
import DriveCard from "./Drive";
import { Badge } from "./MapTab";

// ------------------------------------------------------------------------------------------- road conditions
export function RoadsTab({ b, states, stale, now, t, lang, tz, onOpen }: {
  b: Bundle; states: Record<string, RoadState>; stale: boolean; now: number; t: Copy; lang: Lang; tz?: string;
  onOpen: (id: string) => void;
}) {
  const [onlyAlerts, setOnlyAlerts] = useState(false);
  const rows = b.segments.filter((s) => !onlyAlerts || states[s.id] !== "clear");

  return (
    <div className="doc">
      <DriveCard b={b} states={states} stale={stale} now={now} t={t} lang={lang} tz={tz} onOpen={onOpen} />

      <section className="card">
        <div className="card-head">
          <h2>{t.allStretches}</h2>
          <label className="check"><input type="checkbox" checked={onlyAlerts} onChange={(e) => setOnlyAlerts(e.target.checked)} /> {t.onlyAlerts}</label>
        </div>
        <div className="table-scroll">
          <table className="roads-table">
            <thead>
              <tr>
                <th scope="col">{t.colStretch}</th><th scope="col">{t.colMileposts}</th><th scope="col">{t.colStatus}</th>
                {b.leads.map((m) => <th key={m} scope="col" className="num">{t.colIn(m)}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.map((s) => (
                <tr key={s.id} onClick={() => onOpen(s.id)}>
                  <td><button type="button" className="link-btn" onClick={() => onOpen(s.id)}>{s.name}</button></td>
                  <td className="muted">{s.state} {s.mp[0]}–{s.mp[1]}</td>
                  <td><Badge state={states[s.id]} t={t} /></td>
                  {s.p.map((v, k) => <td key={k} className={`num${v >= alertThreshold(b, s, k) ? " hi" : ""}`}>{stale ? "–" : `${Math.round(v * 100)}%`}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="note">{t.roadsNote}</p>
      </section>
    </div>
  );
}

// ------------------------------------------------------------------------------------------- warnings
export function WarningsTab({ b, active, t, lang, tz }: { b: Bundle; active: NwsProduct[]; t: Copy; lang: Lang; tz?: string }) {
  return (
    <div className="doc">
      <section className="card">
        <h2>{t.warnTitle}</h2>
        <p>{t.warnIntro}</p>
        {b.nws === null ? (
          <p className="alert-box grey">{b.source === "replay" ? t.warnReplay : t.warnUnknown}</p>
        ) : active.length === 0 ? (
          <p className="alert-box ok">{t.warnNone}</p>
        ) : (
          <ul className="warn-list">
            {active.map((w, i) => (
              <li key={i} className="warn-item">
                <strong>{w.type}</strong>
                <span>{w.area}</span>
                <span className="muted">
                  {w.onset && <>{t.warnFrom} {clockShort(w.onset, lang, tz)} · </>}
                  {w.expires && <>{t.warnUntil} {dateLong(w.expires, lang, tz)}</>}
                </span>
              </li>
            ))}
          </ul>
        )}
        <p><a href="https://www.weather.gov/alerts" target="_blank" rel="noreferrer">{t.warnLink} ↗</a></p>
      </section>
    </div>
  );
}

// ------------------------------------------------------------------------------------------- safety
export function SafetyTab({ t }: { t: Copy }) {
  const block = (title: string, items: string[], ordered = false) => {
    const List = ordered ? "ol" : "ul";
    return (
      <section className="card">
        <h2>{title}</h2>
        <List className="steps">{items.map((s) => <li key={s}>{s}</li>)}</List>
      </section>
    );
  };
  return (
    <div className="doc">
      <section className="card intro-card">
        <h1>{t.safetyTitle}</h1>
        <p>{t.safetyIntro}</p>
      </section>
      <div className="grid-2">
        {block(t.aheadTitle, t.ahead)}
        {block(t.caughtTitle, t.caught, true)}
        {block(t.cantTitle, t.cant)}
        {block(t.beforeTitle, t.before)}
      </div>
      <section className="card">
        <h2>{t.linksTitle}</h2>
        <ul className="links">{t.links.map((l) => <li key={l.href}><a href={l.href} target="_blank" rel="noreferrer">{l.label} ↗</a></li>)}</ul>
        <p className="emergency">{t.emergency}</p>
      </section>
    </div>
  );
}

// ------------------------------------------------------------------------------------------- about + status
export function AboutTab({ b, stale, age, t, lang, tz }: { b?: Bundle; stale: boolean; age: number; t: Copy; lang: Lang; tz?: string }) {
  return (
    <div className="doc">
      <section className="card intro-card">
        <h1>{t.aboutTitle}</h1>
        <p>{t.aboutIntro}</p>
      </section>
      <div className="grid-2">
        <section className="card">
          <h2>{t.howTitle}</h2>
          <ol className="steps">{t.how.map((s) => <li key={s}>{s}</li>)}</ol>
        </section>
        <section className="card">
          <h2>{t.accTitle}</h2>
          {t.acc.map((para) => <p key={para}>{para}</p>)}
          <h3>{t.limitsTitle}</h3>
          <ul className="steps">{t.limits.map((s) => <li key={s}>{s}</li>)}</ul>
        </section>
        <section className="card">
          <h2>{t.statusTitle}</h2>
          {b ? (
            <dl className="facts">
              <div><dt>{t.stForecast}</dt><dd>{dateLong(b.t0, lang, tz)}</dd></div>
              <div><dt>{t.stIssued}</dt><dd>{clockShort(b.issued, lang, tz)}</dd></div>
              <div><dt>{t.stAge}</dt><dd>{t.stAgeVal(age)} <span className={`badge ${stale ? "s-unknown" : "s-ok"}`}>{stale ? t.stStale : t.stOk}</span></dd></div>
              <div><dt>{t.stSource}</dt><dd>{t.stSourceVal[b.source]}</dd></div>
              <div><dt>{t.stModel}</dt><dd>{b.model}</dd></div>
              <div><dt>{t.stNws}</dt><dd>{b.nws === null ? t.stNwsDown : t.stNwsOk(b.nws.length)}</dd></div>
              <div><dt>{t.stStations}</dt><dd>{b.stations.length}</dd></div>
            </dl>
          ) : <p>{t.loading}</p>}
        </section>
        <section className="card">
          <h2>{t.sourcesTitle}</h2>
          <ul className="steps">{t.sources.map((s) => <li key={s}>{s}</li>)}</ul>
          <h3>{t.teamTitle}</h3>
          <p>{t.team}</p>
        </section>
      </div>
    </div>
  );
}
