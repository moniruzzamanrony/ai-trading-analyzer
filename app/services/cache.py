"""
Thread-safe in-memory signal cache.

The background worker writes here every 5 minutes.
API endpoints read from here – no DB, no latency spike on request.
"""

import threading
from typing import Optional
from app.models.schemas import SignalResponse


class SignalCache:
    """Holds the latest computed signal for every watched symbol."""

    def __init__(self) -> None:
        self._store: dict[str, SignalResponse] = {}
        self._lock = threading.Lock()

    def set(self, symbol: str, signal: SignalResponse) -> None:
        """Upsert a signal entry (called by the worker)."""
        with self._lock:
            self._store[symbol.upper()] = signal

    def get(self, symbol: str) -> Optional[SignalResponse]:
        """Return the cached signal for *symbol*, or None if not yet computed."""
        with self._lock:
            return self._store.get(symbol.upper())

    def get_all(self) -> list[SignalResponse]:
        """Return all cached signals as a list (preserves insertion order)."""
        with self._lock:
            return list(self._store.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# Module-level singleton shared across the whole application
cache = SignalCache()
