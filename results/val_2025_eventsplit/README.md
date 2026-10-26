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
