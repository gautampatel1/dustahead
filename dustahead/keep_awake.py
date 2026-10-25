"""Keep Windows from sleeping while a long job runs, without touching power settings.

Uses the same Win32 call media players use (SetThreadExecutionState with ES_SYSTEM_REQUIRED): the
request lasts only while the calling process is alive. The display may still turn off; the machine won't sleep.
No-op on Linux/macOS.
"""
from __future__ import annotations

import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


def keep_awake(enable: bool = True) -> bool:
    if sys.platform != "win32":
        return False
    import ctypes
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED if enable else 0)
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))


if __name__ == "__main__":
    import time
    ok = keep_awake(True)
    print("keep-awake", "ON (close this window to allow sleep again)" if ok else "not available on this OS", flush=True)
    from pathlib import Path
    log = Path(__file__).resolve().parents[1] / "logs" / "keep_awake.log"
    log.parent.mkdir(exist_ok=True)
    while True:
        log.write_text(f"keep-awake alive {time.strftime('%Y-%m-%d %H:%M:%S')} ok={ok}\n")
        time.sleep(60)
        ok = keep_awake(True)
