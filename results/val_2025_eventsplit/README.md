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
