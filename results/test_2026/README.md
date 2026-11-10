# 2026 test set — evaluated once, 28 Oct 2026, no tuning after

Test = every 2026 event (Jan–Oct: 37 dust events, 62 storm-only days; 3,029 samples; 67 / 86 / 86 station positives
at +15 / +30 / +60, of which 33 / 62 / 76 are genesis). Temperature fitted on validation, thresholds swept as on
validation; the product threshold (0.80) was fixed on validation before this run. Persistence collapses on 2026
because 72 % of its positives are onset cases: this is the year the project was built for.

## Station CSI at the best threshold (and at the deployed threshold 0.80)

| model | +15 | +30 | +60 | at 0.80: +15 / +30 / +60 |
|---|---|---|---|---|
| station persistence | **0.193** | 0.097 | 0.051 | 0.193 / 0.097 / 0.051 (hits 21 / 15 / 8) |
| unet_all v4 | 0.190 | 0.113 | 0.079 | — |
| gated_unet v4 | 0.040 | 0.021 | 0.017 | — |
| **gated_unet v8** | 0.165 | **0.152** | **0.099** | 0.116 / 0.102 / 0.095 |
| max ensemble (product) | 0.165 | 0.152 | 0.096 | 0.116 / 0.101 / 0.091 (hits 25 / 29 / 18) |

## Genesis CSI at 0.5 with event-bootstrap 95 % CI (33 / 62 / 76 positives)

| model | +15 | +30 | +60 | p vs persistence (+30, Holm) |
|---|---|---|---|---|
| station persistence | 0 | 0 | 0 | — |
| unet_all v4 | 0.054 [0.00, 0.10] | 0.051 [0.00, 0.10] | 0.038 [0.01, 0.07] | 0.096 (0.38) |
| gated_unet v4 | 0.029 [0.00, 0.06] | 0.009 [0.00, 0.02] | 0.006 [0.00, 0.01] | 0.70 (1.0) |
| **gated_unet v8** | **0.030 [0.01, 0.04]** | **0.034 [0.01, 0.05]** | **0.030 [0.01, 0.05]** | **< 0.001 (< 0.001)** |

Best-threshold genesis CSI of the ensemble: 0.081 / 0.102 / 0.064.

## Reading
- **The headline survives out of sample at +30 and +60**: gated v8 beats persistence by +57 % relative at +30
  (0.152 vs 0.097) and +94 % at +60 (0.099 vs 0.051), with onset intervals above zero at every lead and p < 0.001.
  At +15 persistence is better (0.193 vs 0.165): in the first 15 minutes, "what is dusty stays dusty" is hard to beat.
- **Absolute skill is far lower than on validation** (0.15 vs 0.39 at +30) because 2026 is a different kind of year:
  monsoon-dominated (26 of 37 events in July–August), positives mostly onset, few persistent events. Persistence
  itself drops from 0.354 to 0.097. The validation numbers were the easy year; these are the honest ones.
- **gated v4 collapses on 2026** (0.04 / 0.02 / 0.02): the seasonal shortcut found in section 6 of the report
  ("rainy week ⇒ no dust") hits a monsoon-dominated year head on. The static-gate fix (v8) is what makes the model
  work out of sample; the diagnosis was not cosmetic.
- **The ensemble adds nothing on 2026** (it equals v8); on validation it added +0.03–0.08. For the product, v8 alone
  is the safer choice and the ensemble is a validation-year artefact.
- **At the deployed threshold** the model's +60 advantage holds (18 hits vs 8), +30 is a tie, +15 favours persistence.
  A lead-dependent policy (persistence-like at +15, model at +30/+60) is the natural next product step.
- Brier skill scores with the validation temperature are negative on 2026 (over-confident); calibration needs its own
  held-out year, which we do not have.

## Second look, 10 Nov 2026 — final models trained on all of 2025 (pre-registered 9 Nov)

Disclosed second look at the sealed year. What it is for, and what it cannot change, was fixed before training
(`results/val_2025_eventsplit/README.md`, "9 Nov"). v8full is the paper model and v11ffull the onset variant
whatever they score here. The 2026 data, labels and baselines are those of the first look; the old models were
re-scored with today's code and reproduce it (v8 at +60: 0.097, was 0.099).

Scores are **raw** (T = 1). The models trained on the 2025 validation half, so no calibration was fitted on it. A
first scoring of the two (job 97) still fitted one there; it was re-run raw (job 98, commit 09e130e).

Raw scoring files: `second_look/` (eval_test_*.json per model, ensemble JSONs).

### Station CSI at the best threshold, ASOS stations only

| model | +15 | +30 | +60 |
|---|---|---|---|
| station persistence | **0.193** | 0.097 | 0.051 |
| gated_unet v8 (first look) | 0.165 | 0.152 | 0.097 |
| **gated_unet v8full** (paper model) | 0.165 | **0.159** | 0.086 |
| **gated_unet v11ffull** (onset variant) | 0.181 | 0.133 | **0.103** |

At the fixed 0.5 cutoff (ASOS): persistence 0.193 / 0.097 / 0.051, v8full 0.076 / 0.055 / 0.035,
v11ffull 0.121 / 0.100 / 0.070 (FAR 0.84 / 0.87 / 0.91 against v8full's 0.92 / 0.94 / 0.96).

### Onset CSI at 0.5, all stations, event-bootstrap 95 % CI (59 / 114 / 142 onset positives)

| model | +15 | +30 | +60 | Holm p at +30, every baseline |
|---|---|---|---|---|
| station persistence | 0 | 0 | 0 | — |
| gated_unet v8 | 0.025 [0.01, 0.04] | 0.029 [0.01, 0.05] | 0.023 [0.01, 0.04] | ≤ 0.002 |
| gated_unet v8full | 0.031 [0.01, 0.05] | 0.036 [0.02, 0.06] | 0.024 [0.01, 0.04] | < 0.001 |
| **gated_unet v11ffull** | **0.045 [0.01, 0.08]** | **0.057 [0.02, 0.09]** | **0.038 [0.02, 0.06]** | 0.008 |

ASOS-only onset at 0.5: v8 0.030 / 0.034 / 0.029, v8full 0.035 / 0.042 / 0.029, v11ffull 0.059 / 0.073 / 0.049.

### Ensembles, all stations (ASOS + road-weather), best threshold (`second_look/test_ensemble_*.json`)

| | +15 | +30 | +60 | onset +30 [95 % CI] |
|---|---|---|---|---|
| station persistence | 0.204 | 0.080 | 0.023 | 0 |
| v8full | 0.139 | 0.116 | 0.060 | 0.070 [0.02, 0.13] |
| v11ffull | 0.131 | 0.079 | 0.069 | 0.057 [0.02, 0.09] |
| max(v8full, v11ffull) | 0.141 | 0.112 | 0.064 | 0.068 [0.02, 0.12] |
| current product max(unet_all v4, gated v4, v8) | 0.126 | 0.113 | 0.067 | 0.069 [0.02, 0.11] |
| product with v8full: max(unet_all v4, gated v4, v8full) | 0.144 | 0.115 | 0.060 | 0.069 [0.02, 0.13] |

### Reading
- **The result holds on the second look.** Both final models beat persistence at +30 and +60 on the year it
  collapses, with onset intervals above zero and Holm-corrected p ≤ 0.008. Persistence still wins at +15.
- **The extra 2025 data barely moved v8** (+30 0.152 → 0.159, +60 0.097 → 0.086, onset slightly up). These
  differences are inside the event-bootstrap noise; v8full replaces v8 because it was pre-registered to, not because
  it scored better.
- **The onset variant is the better 2026 model on the thing the project is about.** v11ffull has the best +60 score
  and the best onset scores, roughly 1.5× v8full's onset CSI with fewer false alarms at 0.5. On validation it was
  behind v8 at +30. The two years disagree, so this is reported, not acted on: v8full stays the paper model as
  pre-registered, v11ffull is reported beside it as the onset variant.
- **Ensembles add nothing on 2026.** max(v8full, v11ffull) sits between its members, and swapping v8full into the
  product ensemble gains at most 0.005 over v8full alone. This repeats the first look.
- This is the last look at 2026 before the deliverables. Any further change to the models is scored on
  validation folds only.
