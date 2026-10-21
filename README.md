# DustAhead

**Predicting deadly dust storms before they form.** A deep-learning nowcast that warns Interstate 10
drivers in Arizona and New Mexico 15–60 minutes before a wall of dust (haboob) reaches the road.

CS 7150 Deep Learning, Northeastern University (Silicon Valley), Fall 2026.
Gautam Patel · Sanidhya Chimurkar · Aayush Katoch · Professor Ilmi Yoon.

## Idea in one line
A haboob needs two things at once: a thunderstorm outflow (seen from space) and bare, dry ground
(known from weather models and maps). The **gated U-Net** multiplies the two:

```
P_tau(r) = sigmoid( z_tau(r) ) * sigmoid( h(r) )
            ^ U-Net on satellite + lightning      ^ gate CNN on wind, soil moisture, terrain, physics prior
```

## Quick start
```bash
git clone <repo> && cd dustahead
conda env create -f environment.yml && conda activate dustahead   # or pip install -r requirements.txt
pytest -q                                   # ~1 min on CPU: unit tests + synthetic end-to-end
python scripts/00_make_synthetic.py --small # fake samples (no internet)
python scripts/04_train.py --epochs 2       # smoke-train
streamlit run dustahead/app/dashboard.py -- --ckpt runs/latest/best.pt --demo
```

## Real data (run on a machine with internet, ~100 GB free)
| Step | Script | Source | Output |
|---|---|---|---|
| 1 | `01_build_event_list.py` | IEM NWS warnings (DS.W, SV.W) | `data/labels/events.csv` |
| 2 | `02_build_static.py` | USGS 3DEP, SoilGrids, USDA CDL, (NLCD) | `data/static/*.npy` |
| 3 | `03_build_samples.py` | GOES-18 ABI + GLM + ADP (AWS), HRRR (Herbie), ASOS (IEM) | `data/samples/<split>/*.npz` |
| 4 | `04_train.py` | — | `runs/<run>/best.pt` |
| 5 | `05_evaluate.py` | — | `runs/<run>/eval_<split>.json` |

Splits: train 2023–2024 · validate 2025 · test 2026 (run once).

## Layout
```
configs/default.yaml   all settings           dustahead/models/   gated U-Net, ablations, baselines
dustahead/grid.py      the 4 km grid          dustahead/train/    dataset, losses, loop
dustahead/events/      warnings + labels      dustahead/eval/     metrics, bootstrap, evaluate
dustahead/data/        GOES, GLM, HRRR, static, physics, sampler, synthetic
dustahead/app/         predict() + Streamlit dashboard            scripts/  tests/
```

See `CLAUDE.md` for the rules every contributor (human or AI) must follow, and the proposal document
for the full research plan, maths and citations.
