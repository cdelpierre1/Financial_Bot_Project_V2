"""API Usage & Rate Limiting Utilities (Plan Lite)

But:
  - Suivi minute / jour / total par endpoint
  - Limiteur global 500 calls/min (soft cap configurable)
  - Interface simple: acquire(endpoint) -> bool, register(status)

Notes:
  - TokenBucket ré-initialise le crédit chaque minute glissante basée sur fenêtre 60s (pas aligné strictement sur horloge calendrier -> plus fluide).
  - Les requêtes sont enregistrées SEULEMENT si réellement envoyées (les hits cache ne doivent pas comptabiliser).
  - Conçu pour être threadsafe (Scheduler multi-threads).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import threading
import time

DEFAULT_MINUTE_LIMIT = 500      # Limite dure plan
DEFAULT_SOFT_CAP = 450          # Alerte interne (couleur amber)


@dataclass
class EndpointStats:
    calls: int = 0
    successes: int = 0
    client_errors: int = 0
    server_errors: int = 0
    last_status: Optional[int] = None

    def register(self, status: int) -> None:
        self.calls += 1
        self.last_status = status
        if 200 <= status < 400:
            self.successes += 1
        elif 400 <= status < 500:
            self.client_errors += 1
        elif status >= 500:
            self.server_errors += 1


class TokenBucket:
    """Fenêtre glissante sur 60s simple: on conserve timestamps des appels.
    Acquire = accepter si len(window) < hard_limit.
    """
    def __init__(self, hard_limit: int = DEFAULT_MINUTE_LIMIT) -> None:
        self.hard_limit = hard_limit
        self._lock = threading.Lock()
        self._events: list[float] = []  # timestamps (seconds)

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        while self._events and self._events[0] < cutoff:
            self._events.pop(0)

    def acquire(self) -> bool:
        now = time.time()
        with self._lock:
            self._prune(now)
            if len(self._events) >= self.hard_limit:
                return False
            self._events.append(now)
            return True

    def usage(self) -> Tuple[int, int]:
        now = time.time()
        with self._lock:
            self._prune(now)
            return len(self._events), self.hard_limit


class ApiUsageTracker:
    _instance: "ApiUsageTracker" | None = None
    _inst_lock = threading.Lock()

    def __init__(self, hard_limit: int = DEFAULT_MINUTE_LIMIT, soft_cap: int = DEFAULT_SOFT_CAP) -> None:
        self.bucket = TokenBucket(hard_limit=hard_limit)
        self.soft_cap = soft_cap
        self._lock = threading.Lock()
        self._endpoints: Dict[str, EndpointStats] = {}
        self._day_start = int(time.time() // 86400)
        self._daily_total = 0

    @classmethod
    def instance(cls) -> "ApiUsageTracker":
        with cls._inst_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def acquire(self, endpoint: str) -> bool:
        return self.bucket.acquire()

    def register_call(self, endpoint: str, status: int) -> None:
        day_key = int(time.time() // 86400)
        with self._lock:
            if day_key != self._day_start:
                self._day_start = day_key
                self._daily_total = 0
                for ep in self._endpoints.values():
                    ep.calls = ep.successes = ep.client_errors = ep.server_errors = 0
            st = self._endpoints.setdefault(endpoint, EndpointStats())
            st.register(status)
            self._daily_total += 1

    def snapshot(self) -> Dict[str, any]:  # type: ignore[override]
        calls_minute, limit = self.bucket.usage()
        with self._lock:
            per_endpoint = {
                ep: {
                    "calls": s.calls,
                    "success": s.successes,
                    "4xx": s.client_errors,
                    "5xx": s.server_errors,
                    "last_status": s.last_status,
                }
                for ep, s in sorted(self._endpoints.items())
            }
            usage_pct_soft = (calls_minute / self.soft_cap * 100) if self.soft_cap else 0
            state = "GREEN"
            if calls_minute >= self.soft_cap:
                state = "AMBER"
            if calls_minute >= (limit * 0.9):
                state = "RED"
            return {
                "minute_calls": calls_minute,
                "minute_limit": limit,
                "soft_cap": self.soft_cap,
                "state": state,
                "soft_cap_usage_pct": round(usage_pct_soft, 1),
                "daily_total": self._daily_total,
                "endpoints": per_endpoint,
            }


def api_acquire(endpoint: str) -> bool:
    return ApiUsageTracker.instance().acquire(endpoint)


def api_register(endpoint: str, status: int) -> None:
    ApiUsageTracker.instance().register_call(endpoint, status)


def api_snapshot() -> Dict[str, any]:
    return ApiUsageTracker.instance().snapshot()
