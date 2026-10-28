"""Start the Streamlit dashboard as a detached process (so a job runner can launch it and move on).

    python scripts/launch_dashboard.py [--port 8501] [--demo] [--samples data/samples/val]

Logs to logs/dashboard.log. Stop it by deleting the PID file written to logs/dashboard.pid and killing the PID.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="8501")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--samples", default="data/samples/val")
    a = ap.parse_args()
    (ROOT / "logs").mkdir(exist_ok=True)
    log = open(ROOT / "logs" / "dashboard.log", "ab")
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "dustahead" / "app" / "dashboard.py"),
           "--server.headless", "true", "--server.port", a.port, "--browser.gatherUsageStats", "false",
           "--", "--samples", a.samples] + (["--demo"] if a.demo else [])
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
    p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, creationflags=flags,
                         stdin=subprocess.DEVNULL, close_fds=True)
    (ROOT / "logs" / "dashboard.pid").write_text(str(p.pid))
    print(f"dashboard started, pid {p.pid}, http://localhost:{a.port}  (log: logs/dashboard.log)")


if __name__ == "__main__":
    main()
