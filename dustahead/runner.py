"""Job runner for the Windows PC: runs any .bat dropped into jobs/q2/ (oldest first), one at a time,
with the project venv active, logging each job to logs/jobs/<name>.log and moving it to jobs/q2_done/.
Keeps the PC awake while running. Start it once with runner2.bat and leave the window open."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dustahead.keep_awake import keep_awake  # noqa: E402

Q, DONE, LOGS = ROOT / "jobs" / "q2", ROOT / "jobs" / "q2_done", ROOT / "logs" / "jobs"
for d in (Q, DONE, LOGS):
    d.mkdir(parents=True, exist_ok=True)
RUNLOG = ROOT / "logs" / "runner2.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RUNLOG, "a") as f:
        f.write(line + "\n")


log("runner2 started; keep-awake " + ("on" if keep_awake(True) else "n/a"))
while True:
    jobs = sorted(Q.glob("*.bat"), key=lambda p: p.stat().st_mtime)
    for job in jobs:
        target = DONE / job.name
        if target.exists():
            target = DONE / f"{job.stem}_{int(time.time())}.bat"
        job.replace(target)
        out = LOGS / f"{target.stem}.log"
        log(f"START {target.name} -> {out.name}")
        t = time.time()
        with open(out, "w") as f:
            rc = subprocess.run(["cmd", "/c", str(target)], cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT).returncode
        log(f"END   {target.name} rc={rc} after {time.time() - t:.0f}s")
        keep_awake(True)
    time.sleep(10)
