/** Tests for the drive timing and the core display rules. Run: npm test (Node's built-in runner, no extra packages). */
import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { layerWeights, roadState, type Alert, type Bundle } from "./bundle.ts";
import { nearestTown, planDrive, towns } from "./drive.ts";

const demo = (): Bundle => JSON.parse(readFileSync(new URL("../public/demo/latest.json", import.meta.url), "utf8"));
const idx = (b: Bundle, name: string) => towns(b).findIndex((t) => t.name === name);

test("Phoenix to Tucson at 65 mph: seven stretches, 113 miles, timed from departure", () => {
  const b = demo();
  const p = planDrive(b, idx(b, "Phoenix"), idx(b, "Tucson"), 2, 65, {});
  assert.equal(p.legs.length, 7);
  assert.equal(p.miles, 258 - 145);
  assert.ok(Math.abs(p.arriveMin - (2 + (113 / 65) * 60)) < 1e-6);
  assert.equal(p.legs[0].entry, "Phoenix");
  assert.equal(p.legs.at(-1)!.exit, "Tucson");
  for (let k = 1; k < p.legs.length; k++) assert.equal(p.legs[k].enterMin, p.legs[k - 1].exitMin);
});

test("driving west reverses the stretches and enters each at its eastern exit", () => {
  const b = demo();
  const p = planDrive(b, idx(b, "Tucson"), idx(b, "Phoenix"), 0, 65, {});
  assert.equal(p.legs[0].entry, "Tucson");
  assert.equal(p.legs[0].entryMp, 258);
  assert.equal(p.legs.at(-1)!.exit, "Phoenix");
});

test("crossing into New Mexico counts the rest of Arizona plus the New Mexico milepost", () => {
  const b = demo();
  const p = planDrive(b, idx(b, "San Simon"), idx(b, "Lordsburg"), 0, 60, {});
  assert.equal(p.miles, 391 - 378 + 22);
  assert.equal(p.legs[0].entryState, "AZ");
});

test("a stretch reached after the forecast hour has no chance and is never a danger", () => {
  const b = demo();
  const p = planDrive(b, idx(b, "Ehrenberg (CA line)"), idx(b, "El Paso"), 0, 65, {});
  const late = p.legs.filter((l) => l.beyond);
  assert.ok(late.length > 0);
  for (const l of late) { assert.equal(l.chance, null); assert.equal(l.danger, false); }
});

test("dust above the tier threshold at the time you get there is a danger, and the first one gives the advice", () => {
  const b = demo();
  b.segments.forEach((s) => { s.p = [0, 0, 0]; s.genesis = false; });
  const eloy = b.segments.find((s) => s.from === "Eloy")!;
  eloy.p = [0, 0.9, 0.9];                               // dust from +30 on
  // 63 miles Phoenix -> Eloy at 75 mph: you reach Eloy about 50 min after t0, inside the forecast hour
  const p = planDrive(b, idx(b, "Phoenix"), idx(b, "Tucson"), 0, 75, {});
  assert.equal(p.firstDanger?.seg.id, eloy.id);
  assert.equal(p.firstDanger?.entry, "Eloy");
  // leave much later: you reach Eloy after the forecast hour, so it is "beyond", not safe and not a danger
  const later = planDrive(b, idx(b, "Phoenix"), idx(b, "Tucson"), 40, 65, {});
  assert.equal(later.legs.find((l) => l.seg.id === eloy.id)?.beyond, true);
});

test("nearest town to a position on I-10 at Tucson is Tucson", () => {
  const b = demo();
  assert.equal(towns(b)[nearestTown(b, -110.97, 32.23).index].name, "Tucson");
});

test("map blending and road states", () => {
  assert.deepEqual(layerWeights([15, 30, 60], 22.5), [0.5, 0.5, 0]);
  assert.deepEqual(layerWeights([15, 30, 60], 60), [0, 0, 1]);
  const a: Alert = { segment: "x", name: "x", lead_min: 15, p_max: 0.9, level: "severe", genesis: true, held: false, tier: "new_storm" };
  assert.equal(roadState(a, false), "early");
  assert.equal(roadState({ ...a, genesis: false, tier: "nearby" }, false), "severe");
  assert.equal(roadState(a, true), "unknown");             // stale is never clear
  assert.equal(roadState(undefined, false), "clear");
});
