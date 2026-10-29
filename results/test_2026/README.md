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
