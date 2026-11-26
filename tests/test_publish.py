"""The live worker's plumbing (scripts/35_publish.py): secrets file, upload guard. No network, no model."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publish", ROOT / "scripts" / "35_publish.py")
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)


def test_env_file_sets_missing_keys_only_and_skips_comments(tmp_path, monkeypatch):
    f = tmp_path / "publish.env"
    f.write_text('# comment\n\nR2_BUCKET="dustahead-live"\nR2_ENDPOINT=https://x.r2.cloudflarestorage.com\n'
                 "DUSTAHEAD_HC_URL=https://hc-ping.com/abc=def\nnot a pair\n", encoding="utf-8")
    monkeypatch.delenv("R2_BUCKET", raising=False)
    monkeypatch.delenv("DUSTAHEAD_HC_URL", raising=False)
    monkeypatch.setenv("R2_ENDPOINT", "https://already-set")
    names = pub.load_env_file(f)
    assert sorted(names) == ["DUSTAHEAD_HC_URL", "R2_BUCKET"]            # R2_ENDPOINT was already set
    assert os.environ["R2_BUCKET"] == "dustahead-live"                   # quotes removed
    assert os.environ["DUSTAHEAD_HC_URL"] == "https://hc-ping.com/abc=def"   # only the first '=' splits
    assert os.environ["R2_ENDPOINT"] == "https://already-set"
    assert pub.load_env_file(tmp_path / "missing.env") == []


def test_upload_without_credentials_stops_before_any_work(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("R2_")}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "35_publish.py"), "--live", "--once", "--upload",
                        "--env-file", str(tmp_path / "none.env"), "--out", str(tmp_path)],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 2 and "--upload needs R2_ENDPOINT" in r.stderr
    assert not (tmp_path / "latest.json").exists()


def test_watchdog_stops_a_hung_run():
    code = ("import importlib.util, time, sys; sys.argv=['x'];"
            f"s=importlib.util.spec_from_file_location('p', r'{ROOT / 'scripts' / '35_publish.py'}');"
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.start_watchdog(0.5); time.sleep(30)")
    env = {k: v for k, v in os.environ.items() if k != "DUSTAHEAD_HC_URL"}       # no heartbeat request in tests
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 3 and "exceeded" in r.stdout
