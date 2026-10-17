"""Small in-memory sliding-window rate limiter (single process, no dependencies)."""
import time
from collections import defaultdict, deque
from typing import Optional, Tuple


class RateLimiter:
    """Allow at most `limit` hits per `window` seconds for each key (e.g. a client IP).

    limit <= 0 disables it. State lives in this process only, which is correct for a
    single Railway instance; run several replicas and each keeps its own count.
    """

    def __init__(self, limit: int, window: float = 60.0):
        self.limit = limit
        self.window = window
        self.hits = defaultdict(deque)
        self._last_sweep: Optional[float] = None

    def check(self, key: str, now: Optional[float] = None) -> Tuple[bool, int, int]:
        """Return (allowed, remaining, retry_after_seconds)."""
        if self.limit <= 0:
            return True, 0, 0
        now = time.monotonic() if now is None else now
        q = self.hits[key]
        cutoff = now - self.window
        while q and q[0] <= cutoff:
            q.popleft()
        if len(q) >= self.limit:
            return False, 0, max(1, int(q[0] + self.window - now) + 1)
        q.append(now)
        self._sweep(now)
        return True, self.limit - len(q), 0

    def _sweep(self, now: float) -> None:
        if self._last_sweep is None:
            self._last_sweep = now
            return
        if now - self._last_sweep < self.window:
            return
        self._last_sweep = now
        cutoff = now - self.window
        for k in [k for k, q in self.hits.items() if not q or q[-1] <= cutoff]:
            del self.hits[k]