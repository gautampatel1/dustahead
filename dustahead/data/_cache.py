"""Per-key locks so parallel sample builders compute each cached frame/window/field only once."""
from __future__ import annotations

import threading
from collections import defaultdict

_locks: dict = defaultdict(threading.Lock)
_guard = threading.Lock()


def keyed_lock(key) -> threading.Lock:
    with _guard:
        return _locks[str(key)]
