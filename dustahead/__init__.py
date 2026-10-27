"""DustAhead: short-range dust-storm nowcasting for Interstate 10 (Arizona / New Mexico)."""
from pathlib import Path
import yaml

__version__ = "0.1.0"

ROOT = Path(__file__).resolve().parent.parent


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Load configs/default.yaml; another YAML is an EXPERIMENT OVERLAY merged on top of it (only the keys it sets)."""
    default = ROOT / "configs" / "default.yaml"
    with open(default) as f:
        cfg = yaml.safe_load(f)
    if path and Path(path).resolve() != default.resolve():
        with open(path) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg
