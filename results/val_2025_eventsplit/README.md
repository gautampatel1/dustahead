# Validation results — 2025 event split (half of the 2025 events by hash; the other half trains)

Files: `<model>_<version>.json` = `05_evaluate.py --split val` output; `*_trainlog.json` = per-epoch training log.
Versions: v1/v2 = 62 input channels; **v3 = + station-observation channel** (current ASOS dust/clear reports painted
as ±1 discs, 12 km). Calibration: Platt (a, b) fitted **in-sample** on val (2 parameters) — Brier skill scores are
therefore optimistic; report them as such. Station persistence = "every station reporting dust now still reports dust
at +τ"; it is the baseline to beat.

## Station CSI at the best threshold (all stations; n_pos = 158 / 208 / 208)

| model                | +15   | +30   | +60   | BSS +30 (Platt) |
|----------------------|-------|-------|-------|-----------------|
| station persistence  | 0.443 | 0.354 | 0.191 | −0.011 |
| unet_all v3          | **0.476** | 0.369 | **0.225** | **+0.054** |
| convlstm v3          | 0.463 | **0.377** | 0.212 | −0.251 |
| gated_unet v3        | 0.417 | 0.360 | 0.225 | −0.093 |
| unet_dust_only v3    | 0.210 | 0.178 | 0.159 | −0.707 |
| gated_unet v2 (no station ch.) | 0.226 | 0.210 | 0.190 | −1.50 |
| convlstm v1          | 0.243 | 0.207 | 0.191 | −0.50 |
| unet_all v1          | 0.180 | 0.176 | 0.180 | −0.67 |

## Genesis stations (no dust-reporting station within 50 km at t0; n_pos = 47 / 83 / 130), CSI at 0.5, event-bootstrap 95 % CI

| model             | +15 | +30 | +60 | p vs persistence (+30, Holm) |
|-------------------|-----|-----|-----|------------------------------|
| station persistence | 0 | 0 | 0 | — (0 by construction) |
| unet_all v3       | 0.052 [0, 0.08] | 0.047 [0, 0.08] | **0.099 [0, 0.15]** (raw 0.108 [0.01, 0.16]) | 0.23 (0.90) |
| gated_unet v3     | 0.022 [0, 0.04] | 0.060 [0, 0.08] | 0.075 [0, 0.14] | 0.08 (0.31) |
| convlstm v3       | 0.010 [0, 0.03] | 0.046 [0, 0.08] | 0.059 [0.01, 0.08] | 0.25 (1.0) |
| unet_dust_only v3 | 0.066 [0, 0.10] | 0.055 [0, 0.08] | 0.073 [0, 0.12] | 0.08 (0.31) |

Reading: the station channel lifted every learned model from ~0.2 to the persistence level overall and above it at
+60 min (+0.03 CSI, ~15 % relative). On genesis cases the models have POD 0.1–0.2 at FAR 0.8–0.9 — real but small
skill where persistence has none, and with 83 positives at +30 the CIs include zero. The satellite-only ablation
(`unet_dust_only`) is far below everything else: ground state (HRRR, static, physics prior) carries most of the skill.
The gate (`gated_unet`) does not beat the plain U-Net on this data; keep it as the interpretable variant, not the
headline model.

## Offline analysis from `eval_val_preds.npz` (unet_all v3 unless noted)

Thresholds swept 0.05–0.95; "persist" = station persistence at 0.5.

| subset | +15 model / persist | +30 model / persist | +60 model / persist |
|---|---|---|---|
| all stations          | 0.476 / 0.443 | 0.369 / 0.354 | 0.225 / 0.191 |
| non-genesis (dust within 50 km at t0) | **0.660 / 0.553** | **0.550 / 0.475** | **0.421 / 0.295** |
| genesis, threshold tuned on genesis   | 0.071 / 0 | 0.081 / 0 | **0.124 / 0** (42 hits, 210 FA, 88 misses @ 0.25) |

- **Hybrid max(model, persistence) adds nothing** (0.441 / 0.352 / 0.209 — slightly below the model alone): with the
  station channel the model already contains persistence, so the product can run one model, no ensemble rule.
- **Where persistence has information the model beats it clearly** (+0.10–0.13 CSI on non-genesis stations). The
  "all stations" numbers hide this because genesis rows contribute false alarms at every threshold.
- **Genesis needs its own threshold**: best genesis threshold is 0.15–0.30, best overall is 0.65–0.75. The product should
  use a lower alert threshold where no station reports dust yet (two-threshold policy) — this is a calibration choice,
  not a new model.
- **Per-event genesis hits at +60 are concentrated in the spring synoptic wind events** (2025-03-07: 18/26,
  2025-04-17: 8/15, 2025-04-18: 15/23). The monsoon haboobs (2025-08-21, 08-25 Phoenix, 09-02) get **0 genesis hits**
  (gated_unet: same pattern). Interpretation: HRRR wind + ground state predicts frontal/synoptic onset; convective
  outflow onset (satellite + lightning cues, minutes of lead) is not yet learned. That is the open research question
  for the paper and the argument for more convective training cases (2018–22 GOES-16) and a genesis-weighted loss.

## v4 — training years 2018–2024 (+ half of 2025 events); 13,730 train samples, 427 + ~470 station positives

Same validation set. Station CSI at the best threshold (all stations) and genesis CSI at 0.5 with 95 % CI:

| model | +15 | +30 | +60 | genesis +30 | genesis +60 | BSS +30 |
|---|---|---|---|---|---|---|
| station persistence | 0.443 | 0.354 | 0.191 | 0 | 0 | −0.01 |
| unet_all v3 (2023–25) | 0.476 | 0.369 | 0.225 | 0.047 [0, 0.08] | 0.099 [0, 0.15] | +0.05 |
| unet_all v4 | **0.481** | 0.368 | 0.244 | 0.051 [0, 0.09] | 0.083 [0, 0.15] | +0.01 |
| gated_unet v3 | 0.417 | 0.360 | 0.225 | 0.060 [0, 0.08] | 0.075 [0, 0.14] | −0.09 |
| gated_unet v4 | 0.407 | 0.333 | **0.271** | **0.102 [0, 0.15]** (POD 0.27) | **0.152 [0, 0.28]** (POD 0.37) | −0.08 |

Offline (threshold swept on val, so optimistic by the same amount for every row):

| | +15 all / genesis | +30 all / genesis | +60 all / genesis |
|---|---|---|---|
| unet_all v4 | 0.483 / 0.065 | 0.375 / 0.062 | 0.250 / 0.093 |
| gated_unet v4 | 0.425 / 0.100 | 0.327 / 0.111 | 0.261 / **0.176** (34 hits, 58 FA, 96 misses) |
| **max(unet_all v4, gated v4)** | **0.495 / 0.095** | **0.395 / 0.109** | **0.276 / 0.182** |
| station persistence | 0.443 / 0 | 0.354 / 0 | 0.191 / 0 |

Reading: 2018–22 added only ~250 station positives (many old DS warnings have no qualifying 5-minute report), so the
genesis CIs still include zero. But the gated model changed character: at +60 it now has genesis CSI 0.15–0.18 with a
quarter of the false alarms of v3 (58 vs 210) — the ground-state gate buys precision on onset cases at the cost of hit
rate on persistence cases, where the plain U-Net is better. The two are complementary: max of the two is the best
single product everywhere (+0.05 / +0.04 / +0.085 CSI over persistence at +15 / +30 / +60). Genesis hits are still only
in the spring synoptic events; the monsoon haboobs remain unhit at +60. Next: v5 = genesis-weighted station loss
(`configs/genesis_w3.yaml`, positives at genesis stations ×4) for both models.

## v5 — genesis-weighted station loss (×4 on genesis positives): negative result

| model | +15 | +30 | +60 | genesis +30 (POD / FAR) | genesis +60 (POD / FAR) |
|---|---|---|---|---|---|
| gated_unet v4 | 0.425 | 0.327 | 0.261 | 0.086 (0.28 / 0.89) | 0.134 (0.37 / 0.83) |
| gated_unet v5 | 0.290 | 0.237 | 0.157 | 0.020 (0.59 / 0.98) | 0.024 (0.58 / 0.98) |
| unet_all v4 | 0.483 | 0.375 | 0.250 | 0.052 (0.13 / 0.92) | 0.080 (0.18 / 0.87) |
| unet_all v5 | 0.442 | 0.314 | 0.188 | 0.037 (0.46 / 0.96) | 0.061 (0.61 / 0.94) |

The extra weight (effective positive weight 40 at genesis stations) tripled genesis POD but drove FAR to 0.98 and
collapsed CSI everywhere; both runs early-stopped with the best epoch at 0. Genesis positives are too few (~130 in
training) to be up-weighted this hard — the model learns "forecast dust wherever the ground is erodible". Keep v4.
A mild weight (×2, `configs/genesis_w1.yaml`) is queued as v6 for completeness; the product baseline stays
max(unet_all v4, gated_unet v4).

## v6 — mild genesis weight (×2): also negative, monotonic in the weight

| model | +15 | +30 | +60 | genesis +60 best-thr (hits / FA / misses) |
|---|---|---|---|---|
| gated_unet v4 | 0.425 | 0.327 | 0.261 | 0.176 (34 / 58 / 96) |
| gated_unet v6 | 0.408 | 0.293 | 0.233 | 0.130 (35 / 140 / 95) |
| unet_all v4 | 0.483 | 0.375 | 0.250 | 0.093 (17 / 51 / 113) |
| unet_all v6 | 0.462 | 0.387 | 0.205 | 0.103 (43 / 288 / 87) |

Same mechanism at half strength: a few more genesis hits bought with 2–5× the false alarms. Loss re-weighting of
genesis positives is closed as a direction (v5, v6); the lever is onset examples, not weights. **Model freeze for the
30 Nov prototype: max(unet_all v4, gated_unet v4)** with a two-threshold policy (lower alert threshold where no
station within 50 km reports dust).

## v7 — station-dropout augmentation (30 % of current reports hidden in training)

| model | +15 | +30 | +60 | genesis +30 (POD / FAR) | genesis +60 (POD / FAR) |
|---|---|---|---|---|---|
| unet_all v4 | 0.483 | 0.375 | 0.250 | 0.052 (0.13 / 0.92) | 0.080 (0.18 / 0.87) |
| unet_all v7 | 0.456 | 0.328 | 0.202 | 0.070 (0.41 / 0.92) | 0.102 (0.41 / 0.88) |
| gated_unet v4 | 0.425 | 0.327 | 0.261 | 0.086 (0.28 / 0.89) | 0.134 (0.37 / 0.83) |
| gated_unet v7 | 0.407 | 0.301 | 0.218 | 0.045 (0.25 / 0.95) | 0.064 (0.39 / 0.93) |
| max(v7, v7) | 0.456 | 0.333 | 0.225 | genesis best-thr 0.067 / 0.064 / 0.116 | |
| max(v4, v4) | **0.495** | **0.395** | **0.276** | genesis best-thr 0.095 / 0.109 / 0.182 | |

Dropout did what it was meant to do on the Phoenix 2025-08-25 replay — the domain-max probability no longer flickers
0.9 → 0.0 → 0.9 between 10-min steps (v7: a smooth 0.7–0.9 through the whole sequence) — but it is also ~0.8 three
hours before and two hours after the storm: the model became smoother by becoming less discriminating, and CSI drops
at every lead. **Product keeps v4 + alert hysteresis** (`AlertTracker`, hold ≥ 30 min, release < 0.5); v7 stays as a
robustness ablation.

## Why the gated model misses every monsoon haboob (`scripts/08_inspect_gate.py`, gated_unet v4)

| event | gate σ(h): domain mean / p90 / max | sky branch σ(z) max | HRRR rain7d (norm.) |
|---|---|---|---|
| 2025-08-25 Phoenix haboob (all 9 times) | **0.000 / 0.000 / 0.000** | 1.00 | 0.077–0.089 |
| 2025-08-21 haboob (all 7 times) | **0.000 / 0.000 / 0.000** | 0.96–1.00 | 0.097 |
| 2025-03-07 synoptic | 0.003–0.142 / up to 0.92 / 1.000 | 1.00 | 0.028–0.078 |
| 2025-04-18 synoptic | 0.003–0.237 / up to 0.99 / 1.000 | 1.00 | 0.007–0.019 |

The sky branch says "dust" (σ(z) = 1 somewhere in the domain in every sample, so it carries little discrimination on
its own); the gate does all the work, and in monsoon season it is shut over the entire domain. The ground inputs that
differ between the two seasons are the HRRR 7-day rain and humidity, not the soil/terrain layers: the gate has learned
"rainy week ⇒ no dust", a shortcut supplied by the storm-only null days (severe-thunderstorm days are monsoon days).
This is the mechanism behind "zero genesis hits on monsoon cases" in every version so far.
Fix under test as **v8 = gate on static + physics prior only** (`configs/gate_static.yaml`; HRRR moves to the U-Net
branch), so the gate can only express erodibility, which is what P(dust) = P(outflow) × P(erodible) was meant to mean.

## v8 — gate on static + physics prior only (`configs/gate_static.yaml`): the gate opens in monsoon season

Gate inspection (same samples as above): 2025-08-25 Phoenix σ(h) mean 0.22–0.30, p90 0.74–0.82, max 1.0 (was 0.000
everywhere); 2025-08-21 mean 0.17–0.30, max ≥ 0.99. The seasonal shortcut is gone.

| model | +15 | +30 | +60 | genesis +15 / +30 / +60 at 0.5, 95 % CI | p vs persistence +30 (Holm) |
|---|---|---|---|---|---|
| gated_unet v4 | 0.425 | 0.327 | **0.261** | 0.02 [0, 0.04] / 0.086 [0, 0.15] / 0.134 [0, 0.27] | 0.078 (0.31) |
| **gated_unet v8** | **0.508** | **0.383** | 0.210 | **0.056 [0.01, 0.08] / 0.076 [0.03, 0.10] / 0.096 [0.03, 0.13]** | **0.014 (0.056)** |
| station persistence | 0.443 | 0.354 | 0.191 | 0 | — |

First model whose genesis confidence intervals exclude zero at every lead, and the first genesis hits on monsoon
haboobs (+60, threshold 0.55: Phoenix 2025-08-25 4 of 13, 2025-08-21 1 of 6; the spring events keep 12/26, 15/15,
17/23). The price is +60 min skill on persistence cases (0.210 vs 0.261): with HRRR in the U-Net branch the model is
sharper at short leads and less so at +60.

Ensembles (offline, best threshold):

| | +15 all / genesis | +30 all / genesis | +60 all / genesis |
|---|---|---|---|
| max(unet_all v4, gated v4) | 0.495 / 0.095 | 0.395 / 0.109 | 0.276 / 0.182 |
| max(unet_all v4, gated v8) | 0.484 / 0.061 | 0.385 / 0.068 | 0.241 / 0.113 |
| **max(unet_all v4, gated v4, gated v8)** | 0.489 / 0.082 | 0.388 / 0.105 | 0.272 / 0.180 |

**Product: the three-member max** — same skill as the v4 pair within noise, plus monsoon coverage from v8.
**Paper headline model: gated_unet v8** (interpretable gate, genesis significance, Holm p = 0.056 vs persistence).

## 30 Oct — fine-tuning (v9), the other 2025 fold, 2-fold pooling, and the +15 persistence floor

Four cheap experiments after the 2026 test run (none touches the test split; chosen on validation only).

### v9 = v8 fine-tuned (`configs/finetune_v9.yaml`: init from v8, lr 1e-4, EMA 0.999, 12 epochs max): neutral on CSI, better calibrated

| model | +15 | +30 | +60 | genesis +15 / +30 / +60 at 0.5 (raw) | Brier +30 (raw) |
|---|---|---|---|---|---|
| gated_unet v8 | **0.508** | **0.383** | 0.210 | 0.050 [0.01, 0.07] / 0.060 [0.02, 0.08] / 0.093 [0.03, 0.13] | 0.0027 |
| gated_unet v9 | 0.488 | 0.362 | 0.211 | 0.046 [0.02, 0.06] / 0.065 [0.03, 0.09] / 0.093 [0.03, 0.13] | **0.0024** |

v8's best epoch was its second (`gated_unet_v8_trainlog.json`): the recipe peaks early and every later epoch overfits,
and a 10× lower learning rate with weight averaging does not change that — training loss fell 1.04 → 0.69 over nine
epochs while the validation score stayed flat (best at epoch 4). The averaged weights are better calibrated (Brier
−10 % before any Platt fit, same Holm p = 0.056) but not more skilful. **v8 stays the paper model; v9 is not used.**
Fine-tuning is closed as a direction unless the data change.

### Fold 1 = the v8 recipe trained on the *other* half of the 2025 events (`configs/fold1.yaml`)

The md5 split is unbalanced: fold 0's validation half (v8's) holds 158 / 208 / 208 station positives at +15 / +30 / +60,
fold 1's holds 553 / 739 / 750 — fold 1 contains the big spring events, and station persistence is stronger there
(0.546 / 0.473 / 0.382).

| fold 1 validation | +15 | +30 | +60 | genesis +15 / +30 / +60 at 0.5 (raw), 95 % CI | p vs persistence +30 (Holm) |
|---|---|---|---|---|---|
| station persistence | **0.546** | **0.473** | **0.382** | 0 | — |
| gated_unet v8 recipe, fold 1 | 0.528 | 0.453 | 0.361 | 0.052 [0.02, 0.08] / 0.054 [0.01, 0.08] / 0.071 [0.02, 0.11] | **0.004 (0.016)** |

On this half the model is *below* station persistence on all-station CSI at every lead (−0.02) while keeping the same
onset skill. The all-station advantage seen on fold 0 therefore does not generalise; the onset skill does.

### 2-fold pooled: every 2025 event scored out of sample once (`scripts/12_pool_genesis.py`, `val_pool_v8_2fold*.json`)

129 events, 445k station-times; genesis positives 171 / 296 / 414 (was 47 / 83 / 130 on fold 0 alone).

| pooled 2025, threshold 0.5 | +15 | +30 | +60 |
|---|---|---|---|
| genesis CSI, raw model (95 % CI) | **0.051 [0.03, 0.07]** | **0.057 [0.03, 0.08]** | **0.081 [0.05, 0.11]** |
| genesis POD / FAR | 0.16 / 0.93 | 0.18 / 0.92 | 0.25 / 0.89 |
| p vs station persistence (paired event bootstrap) | 0.0000 | 0.0000 | 0.0000 |
| all-station CSI, best threshold: model / persistence | 0.501 / 0.521 | 0.414 / 0.445 | 0.314 / 0.336 |

**This is the paper's onset claim**: with three times the positives the intervals sit well above zero at every lead
and the paired p-values are < 0.001 before correction (< 0.004 after Holm over four baselines) — the fold-0 result
(p = 0.056) was limited by sample size, not effect size. The honest companion statement: pooled over all of 2025 the
model is at or slightly below station persistence on all-station CSI; its contribution is the 15–25 % of onsets it
catches where persistence catches none, at FAR ≈ 0.9. The same pattern appeared on the 2026 test (`results/test_2026`).

### Alert policy: max(model, station persistence) at +15 — negative on validation

| | +15 CSI (thr) fold 0 | fold 1 | pooled |
|---|---|---|---|
| gated v8 / v9 alone | 0.510 (0.85) / 0.493 (0.70) | 0.528 (0.75) | 0.501 (0.80) |
| + persistence floor at +15 | 0.442 / 0.450 | 0.548 | 0.523 |
| station persistence | 0.443 | 0.546 | 0.521 |

The floor drags the forecast to persistence's score wherever it is applied: it imports persistence's false alarms
(stations whose dust clears within 15 min), which the model was correctly discounting (P < 0.7 at those stations).
It helps on fold 1 only because persistence is better than the model there. **Not adopted**: `product.persistence_leads`
stays empty, and the +15 lead keeps the model's own threshold. The 2026 +15 loss to persistence is a distribution
shift between years, not a policy gap.

### Ensembles with calibrated members (`scripts/11_ensemble_eval.py`, best overall threshold; genesis at that threshold)

| max ensemble | +15 | +30 | +60 | genesis +60 at the overall threshold |
|---|---|---|---|---|
| unet_all v4 + gated v4 + gated v8 (product) | 0.513 | 0.399 | 0.246 | 0.158 [0.05, 0.20] |
| unet_all v4 + gated v4 + gated v9 | 0.500 | 0.394 | 0.252 | 0.156 [0.03, 0.21] |
| unet_all v4 + gated v9 | 0.495 | 0.385 | 0.229 | 0.090 [0, 0.17] |

No ensemble with v9 beats the product ensemble; `product.checkpoints` is unchanged. Note that at the best *overall*
threshold (0.85) gated v8 alone has zero onset skill — onset needs the lower genesis threshold the product policy
already provides (`product.genesis_thresholds`); the table above is the "single threshold" view.

**Closed directions after today:** fine-tuning/EMA (v9), persistence floor. **Open:** HRRR short-range forecast
fields as inputs, longer satellite history, self-supervised pre-training of the sky branch — all aimed at onset POD.

## 1 Nov — v10: HRRR short-range FORECAST channels (`configs/hrrr_fcst_v10*.yaml`, `scripts/13_add_hrrr_forecast.py`)

Six channels appended to every existing sample in place (no other data refetched): from the HRRR run initialised
2 h before the hour of t0 (complete on AWS by t0), 10 m u/v, gust, CAPE and composite reflectivity valid at the first
full hour after t0, and gust one hour later. The forecast outflow is the physical driver of onset; the analysis only
says what the wind does now. v10 = warm start from v8 (new input weights zero, so epoch 0 reproduces v8; lr 3e-4);
v10s = same inputs from scratch with the v8 recipe.

| model (val, best threshold; genesis at 0.5, raw) | +15 | +30 | +60 | genesis CSI +15 / +30 / +60 | genesis POD +15 / +30 / +60 | FAR | p vs persist. +30 (Holm) |
|---|---|---|---|---|---|---|---|
| gated_unet v8 | 0.508 | 0.383 | 0.210 | 0.050 / 0.060 / 0.093 | 0.30 / 0.35 / 0.42 | 0.94 / 0.93 / 0.89 | 0.056 |
| gated_unet v10 (warm, best = epoch 0) | 0.505 | 0.359 | 0.217 | 0.040 / 0.051 / 0.081 | **0.36 / 0.42 / 0.49** | 0.96 / 0.95 / 0.91 | 0.056 |
| gated_unet v10s (scratch, best = epoch 1) | 0.500 | **0.402** | 0.217 | 0.036 / 0.057 / 0.077 | 0.23 / 0.36 / 0.41 | 0.96 / 0.94 / 0.91 | **< 0.004** |
| max(unet_all v4, gated v4, gated v10s) | 0.515 | **0.409** | 0.255 | (at the overall threshold) 0.019 / 0.057 / 0.137 | | | |
| max(unet_all v4, gated v4, gated v8) — product | 0.513 | 0.399 | 0.246 | 0 / 0.032 / 0.158 | | | |

Reading. The forecast block does what it was expected to do physically — the warm start raises onset POD by
0.06–0.07 at every lead — but buys it with the same proportion of extra false alarms, so onset CSI is unchanged
within the intervals (every v10 interval overlaps v8's). Trained from scratch the extra channels go into
all-station skill instead: v10s has the best +30 CSI of any single model (0.402 vs 0.383) and the strongest
single-fold onset significance (paired p < 0.001 at +30, Holm < 0.004), with onset CSI level with v8. Both runs
early-stop by epoch 1–2 like every version before them: with ~130 onset positives in the training half the models
saturate within two passes over the data whatever the inputs. **The bottleneck is onset examples, not information.**

Decisions: paper model stays v8 (same onset skill, simpler input set, interpretable gate); v10s is reported as the
"forecast inputs" ablation (+0.02 CSI at +30, no onset gain). The product ensemble with v10s instead of v8 is
+0.01 at +30/+60 on validation — within noise; `product.checkpoints` unchanged unless the team prefers the
forecast-aware member for the live demo (it needs the HRRR forecast fetch at run time, ~10 s extra per cycle).

## 2–3 Nov — v11: onset-window densification (5-minute sampling; `03_build_samples.py --onset-step 5`)

Every TRAINING dust event (2018–24 and the 2025 training half; never the validation half, never 2026) got extra
samples every 5 min from 3 h before its first warning to 1 h after it — the window where genesis positives live.
GOES scans every 5 min, so the in-between frames are new observations; labels come from the 5-min station
reports. +5,008 samples (+23 %), +229 / 226 / 297 positive station labels at +15 / +30 / +60 (+20 %), each with the
HRRR forecast block. Validation is the same 60 events / 202k station-times as every table above.

| model | best epoch | +15 | +30 | +60 | genesis CSI at 0.5, calibrated (raw) | genesis POD raw | p vs persist. +30 (Holm) |
|---|---|---|---|---|---|---|---|
| gated_unet v8 (10-min data) | 1 | 0.508 | 0.383 | 0.210 | 0.056 / 0.076 / 0.096 (0.050 / 0.060 / 0.093) | 0.30 / 0.35 / 0.42 | 0.014 (0.056) |
| gated_unet v11 = v8 recipe, densified | **13** | 0.493 | 0.388 | 0.209 | 0.049 / 0.067 / **0.107 [0.04, 0.17]** (0.054 / 0.067 / 0.094) | 0.21 / 0.28 / 0.42 | 0.014 (0.056) |
| gated_unet v11f = + HRRR forecast channels | 3 | 0.483 | 0.337 | 0.222 | **0.075 / 0.077 / 0.099** (0.059 / 0.063 / 0.081) | **0.30 / 0.51 / 0.62** | **0.006 (0.024)** |

Ensembles (calibrated members, best overall threshold): max(unet_all v4, gated v4, v11) 0.490 / 0.390 / 0.235;
with v11f 0.495 / 0.398 / 0.247 — level with the product's 0.513 / 0.399 / 0.246.

Reading.
- **The training curve changed.** Every version before v11 peaked at epoch 1–2 and overfit from there; v11 kept
  improving until epoch 13 and stopped at 21. The models were starved of onset examples — a 20 % increase in
  positives bought twelve useful epochs. That is the clearest evidence so far about what limits this problem.
- **Onset skill moved, all-station skill did not.** v11's calibrated onset CSI at +60 (0.107) is the best single-model
  value posted; v11f, which adds the forecast channels, is the first single-fold model with a Holm-corrected
  p below 0.05 (0.024) and detects 62 % of +60 onsets (v8: 42 %) at the same false-alarm ratio. All-station CSI
  stays within ±0.02 of v8 at every lead, because the extra samples are all pre-onset and add nothing to the
  persistence-type cases that dominate that score.
- **Compounding is partial.** v11f's gains over v11 are at the onset stratum only; its +30 all-station CSI drops to
  0.337, and it again peaks early (epoch 3). More onset examples and more input information help the same cases.

Decisions: product unchanged (no ensemble beats it). Paper: v8 remains the clean reference; v11 / v11f go in as the
"data-limited" evidence (training-curve figure + the POD gain). Next data levers, in order: road-weather (RWIS) and
PM10 ground truth through Synoptic Data (needs an API token), then self-supervised pre-training of the sky branch.

## 3 Nov — v12: self-supervised pre-training of the sky branch (`scripts/15_pretrain.py`, `configs/pretrain_v12.yaml`)

The U-Net was first trained with no labels to reconstruct randomly hidden 16 × 16-cell patches (50 % of each
crop) of its own 56-channel input on the 21,742 training samples: masked MSE 0.136 → 0.098 over 15 epochs
(`pretrain_mae_v1_log.json`). v12 then trains the v8 recipe from those weights (head and gate new, lr 5e-4).

| model | best epoch | +15 | +30 | +60 | genesis CSI at 0.5 (raw) | genesis POD | Brier +30 | p vs persist. (Holm) |
|---|---|---|---|---|---|---|---|---|
| gated_unet v8 | 1 | 0.508 | 0.383 | 0.210 | 0.050 / 0.060 / 0.093 | 0.30 / 0.35 / 0.42 | 0.0027 | 0.014 (0.056) |
| gated_unet v11 (densified) | 13 | 0.493 | 0.388 | 0.209 | 0.054 / 0.067 / 0.094 | 0.21 / 0.28 / 0.42 | 0.0020 | 0.014 (0.056) |
| gated_unet v12 (pre-trained, densified) | 9 | 0.497 | 0.388 | 0.201 | **0.024 / 0.007 / 0.049** | **0.04 / 0.01 / 0.09** | **0.0012** | 0.70 (1.0) |

**Negative for onset.** All-station skill and calibration are as good as any model (the lowest Brier score so far),
but onset detection collapsed: 1 % of +30 onsets caught. The training loss fell to 0.20 (v8 stopped at 1.0), i.e.
the pre-trained features let the network memorise the training set within ten epochs, and what it memorised is
the persistence-type signal, which dominates the loss. Reconstruction pre-training rewards low-level texture, not
the rare precursor patterns. Follow-up queued as v12b (lr 1e-4, EMA 0.999) to test whether a gentler fine-tune keeps
the onset sensitivity; if that is also negative, pre-training is closed in this form (a contrastive or
future-frame objective would be the next thing to try, not more of the same).

v12b (same pre-trained U-Net, lr 1e-4 + EMA, best epoch 9): 0.385 / 0.304 / 0.184, onset CSI 0.000 / 0.014 / 0.022 —
worse everywhere. **Masked-reconstruction pre-training is closed**: it hands the supervised stage features tuned for
texture, which it then either overwrites (v12) or cannot escape (v12b). A pre-text task tied to motion (future-frame
prediction) or to contrast between pre-onset and quiet skies would be the next thing to try, not more of this one.

## 4 Nov — v13: road-weather (RWIS) visibility as additional ground truth

Setup. Synoptic Data gave historical access (academic tier). Of the three new sources, only ADOT / NMDOT road-weather
visibility survived calibration against the ASOS rule (`results/labels/README.md`): 37 stations, 887 positive rows
in 61 events (+40 % positives), the same quantity the product forecasts, measured on the highway itself. The samples
now carry those stations (`st_src` = 1) next to the 85 ASOS stations (`st_src` = 0); the validation benchmark is
273,684 station-times, of which 202,340 are the ASOS rows of every earlier table. The ASOS-only view of v8 is
unchanged to three decimals (0.508 / 0.383 / 0.210; persistence 0.443 / 0.354 / 0.191), so the numbers below are
comparable with everything above. v13 = the forecast-channel recipe, v13a = the v8 recipe, both trained on
ASOS + RWIS labels.

| validation (raw) | best epoch | ASOS-only CSI +15 / +30 / +60 | ASOS onset CSI at 0.5 (POD) | RWIS stratum CSI at 0.5 (POD / FAR), n = 139 | Brier +30 |
|---|---|---|---|---|---|
| station persistence | — | 0.443 / 0.354 / 0.191 | 0 | 0.236 / 0.178 / 0.112 (0.65 / 0.52 / 0.35; 0.73 / 0.79 / 0.86) | — |
| gated v8 (ASOS labels) | 1 | **0.508 / 0.383 / 0.210** | **0.051 / 0.063 / 0.093** (0.30 / 0.36 / 0.42) | **0.267 / 0.225 / 0.155** (0.53 / 0.42 / 0.28; 0.65 / 0.67 / 0.74) | **0.0026** |
| gated v11f (ASOS, densified + forecast) | 3 | 0.483 / 0.335 / 0.221 | 0.059 / 0.062 / 0.081 (0.30 / 0.51 / 0.62) | 0.238 / 0.181 / 0.088 | 0.0034 |
| gated v13a (ASOS + RWIS labels, v8 recipe) | 1 | 0.467 / 0.358 / 0.217 | 0.037 / 0.045 / 0.058 (0.35 / 0.42 / 0.46) | 0.188 / 0.143 / 0.071 (0.65 / 0.58 / 0.38; 0.79 / 0.84 / 0.92) | 0.0046 |
| gated v13 (ASOS + RWIS labels, forecast recipe) | 1 | 0.453 / 0.365 / 0.199 | 0.033 / 0.042 / 0.059 (0.28 / 0.39 / 0.48) | 0.197 / 0.174 / 0.102 | 0.0041 |

**Verdict: the RWIS labels do not help — they hurt, on the road stations included.** v13a against v8, same recipe:
−0.04 / −0.025 / +0.007 ASOS-only CSI, onset CSI down a third at every lead, Brier almost doubled, and on the RWIS
stations it was trained on 0.188 / 0.143 / 0.071 against 0.267 / 0.225 / 0.155 for v8, which never saw an RWIS label.
What the models trained on RWIS learned is to alert more often (RWIS-stratum POD 0.65 at FAR 0.79–0.92): the
RWIS "dust" rows pull the decision boundary toward low-visibility episodes the satellite cannot see, which is what
the agreement check hinted at (only 19 of 61 RWIS dust events coincide with any ASOS dust report). The paired
p-values improve (Holm 0.024 / 0.008) only because the all-station onset stratum now contains the RWIS positives —
a benchmark artefact, not skill.

Two findings worth keeping. (1) **The ASOS-trained model generalises to the road sensors it never saw**: v8 beats
station persistence on the RWIS stratum at every lead (0.267 vs 0.236, 0.225 vs 0.178, 0.155 vs 0.112) — the first
direct evidence of skill on the highway itself, which is where the product issues alerts. (2) The right role for the
new sources is evaluation and live input, not training labels: `label.train_sources: [asos]` (default from today)
masks the extended stations' discs and rows in the loss while they stay in the sample for the station channel and
for per-source scoring; `ext_sources: [rwis]` keeps PM10 and the range sensors out altogether.

Decisions: product unchanged; v8 remains the paper model, v11f the onset-detection variant; RWIS becomes a reported
stratum ("skill on road-weather stations, zero-shot"). Closed: PM10 labels, range-sensor labels, RWIS labels.

## 5–7 Nov — v14: radar inputs and a dense satellite head (`configs/radar_v14.yaml`, `aux_v14a.yaml`, `radar_only_v14b.yaml`)

Two ingredients borrowed from the Saharan dust systems (DustNet, AI-GAMFS), tested together and apart, on top of the
v11f recipe (densified onset windows + HRRR forecast block):
- **radar**: NEXRAD composite reflectivity (IEM N0Q mosaic) at t0−20, t0−10, t0, maximum dBZ per 4-km cell so the
  outflow "fine line" survives (`data/radar.py`, `scripts/19_add_radar.py`; 28,088 frames, 425 of 27,251 samples
  with a missing frame filled with zeros);
- **dense head**: a second output predicting the painted satellite + station target at every labelled pixel at full
  weight (`loss.w_aux: 1`), the "learn from a dense satellite product" idea.

Same validation half and ASOS-only view as every earlier table (raw probabilities; onset at threshold 0.5):

| model | best epoch | CSI +15 / +30 / +60 | onset CSI +15 / +30 / +60 | onset POD / FAR at +60 | Brier +30 | p vs persist. +30 (Holm) |
|---|---|---|---|---|---|---|
| v8 (paper model) | 1 | **0.508** / 0.383 / 0.210 | 0.051 / 0.063 / 0.093 | 0.42 / 0.89 | 0.0026 | 0.014 (0.056) |
| v11f (densified + forecast) | 3 | 0.483 / 0.335 / 0.221 | 0.059 / 0.062 / 0.081 | 0.62 / 0.91 | 0.0034 | 0.012 (0.048) |
| v14a = v11f recipe + dense head only | 1 | 0.510 / 0.381 / 0.213 | 0.024 / 0.051 / 0.051 | 0.17 / 0.93 | 0.0024 | 0.030 (0.12) |
| v14 = + radar + dense head | 8 | 0.497 / **0.394** / **0.235** | 0.061 / 0.070 / 0.100 | 0.43 / 0.89 | 0.0024 | 0.014 (0.056) |
| **v14b = + radar only** | 12 | 0.493 / 0.384 / 0.229 | **0.088 / 0.103 / 0.142** | 0.42 / **0.82** | **0.0020** | **0.004 (0.016)** |

**Attribution: the gain is the radar; the dense head is neutral alone and harmful in combination.** v14a (head only)
behaves like v8 — peaks at epoch 1, onset worse — so a dense satellite target adds nothing the station-centred crops
did not already give. Radar alone (v14b) lifts onset CSI by **+73 % / +63 % / +53 %** over v8 at the three leads,
the largest single-step onset gain of the project, *and* lowers the onset false-alarm ratio at +60 from 0.89 to
0.82 — the first time a detection gain was not paid for in false alarms. It is the first single-fold model with a
Holm-corrected p below 0.02, keeps all-station CSI within ±0.02 of v8 (−0.015 at +15, +0.001 at +30, +0.019 at +60),
trains to epoch 12, and has the best Brier score of any run. Adding the dense head to radar (v14) gives back about
a third of the onset gain. Physically this is what the haboob literature predicts: the outflow boundary that
precedes the dust is a radar feature (a thin line of weak echo) that no satellite channel resolves.

Caveats, in order of importance:
1. **Radar latency.** The mosaic frame stamped t0 is published a few minutes after t0; v14b used it. A forecast issued
   at t0 has only t0−10 and t0−20. `configs/radar_lag_v14c.yaml` (`channels.radar_latency_frames: 1`) retrains with
   the t0 frame blanked in training *and* evaluation; v14b's gain counts only if v14c keeps it. Running 7 Nov.
2. **Road-weather stratum at +60.** On the 37 RWIS stations (never trained on), best-threshold CSI is
   0.263 / 0.230 / 0.075 for v14b vs 0.296 / 0.245 / 0.155 for v8 (v14: 0.270 / 0.143 / 0.043). The radar models
   lose skill on the highway sensors at the longest lead while gaining it at the airports. 139 positives per lead,
   so the interval is wide, but the sign is consistent across v14 and v14b. Not yet understood.
3. Single fold, 83–130 onset positives per lead; the pooled two-fold test that settled v8 has not been run for v14b.
4. The ensemble check failed on 5 Nov because the v4 members had been scored on the pre-RWIS benchmark (rows do not
   line up); they are being re-scored on the current benchmark (job 90) and the ensembles re-run.

**v14c — the latency test (7 Nov): the radar gain does not survive a 10-minute delay.** Blanking the t0 frame in
training and evaluation (`configs/radar_lag_v14c.yaml`), so the model sees only t0−10 and t0−20:

| model | best epoch | CSI +15 / +30 / +60 | onset CSI +15 / +30 / +60 | onset FAR +60 | p vs persist. (Holm) | RWIS best-thr CSI |
|---|---|---|---|---|---|---|
| v8 (no radar) | 1 | 0.508 / 0.383 / 0.210 | 0.051 / 0.063 / 0.093 | 0.89 | 0.014 (0.056) | 0.296 / 0.245 / 0.155 |
| v14b (radar t0−20, −10, **0**) | 12 | 0.493 / 0.384 / 0.229 | **0.088 / 0.103 / 0.142** | **0.82** | 0.004 (0.016) | 0.263 / 0.230 / 0.075 |
| v14c (radar t0−20, −10 only) | 1 | 0.498 / 0.373 / 0.187 | 0.031 / 0.046 / 0.051 | 0.94 | 0.008 (0.032) | 0.299 / 0.256 / 0.141 |

Every part of v14b's gain sits in the newest frame: without it, onset skill falls *below* v8, the training curve
reverts to the one-epoch peak, and (consistently) the road-weather regression at +60 disappears too. Physically the
signal is fast: an outflow boundary moves 10–20 m/s, 6–12 km (two to three grid cells) in ten minutes, so a frame
ten minutes old puts the gust front in the wrong cells, and the satellite and station channels already carry what
older radar knows. Radar here is a *now-observation* of the gust front, not a precursor with lead time to spare.

That makes freshness the whole question. The IEM mosaic publishes roughly 5 min after valid time; MRMS about 2 min;
single-site Level-II data from the Phoenix and Tucson radars within 1–2 min on AWS. **v14d** stores the frames a live
system would actually have from IEM (t0−25, −15, **−5**; `scripts/19_add_radar.py --offsets 25 15 5 --key xr_lag5`,
`configs/radar_lag5_v14d.yaml`) and retrains; it decides whether radar enters the paper model.

Until then: v8 stays the paper model and the product is unchanged. v14b is reported as the upper bound of what
radar can add with zero latency; v14c as the 10-minute-latency negative. The dense head (`loss.w_aux`) is closed.

Ensembles on the current benchmark (all 273,684 station-times including the road-weather stratum, best overall
threshold; not comparable to the ASOS-only rows above): max(unet_all v4, gated v4, v8) 0.325 / 0.261 / 0.184;
with v14b instead of v8 0.334 / 0.295 / 0.190 (`val_ensemble_v4v4v8_rwisbench.json`, `val_ensemble_v4v4v14b.json`).
