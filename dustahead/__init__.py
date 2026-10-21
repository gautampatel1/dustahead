"""DustAhead: short-range dust-storm nowcasting for Interstate 10 (Arizona / New Mexico)."""
from pathlib import Path
import yaml

__version__ = "0.1.0"

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    """Load configs/default.yaml (or another YAML) as a plain dict."""
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
