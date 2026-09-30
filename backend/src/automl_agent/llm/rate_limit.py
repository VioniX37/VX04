"""Client-side throttling for the Gemini free tier.

Two mechanisms are combined:

* a sliding-window requests-per-minute limiter per model, and
* a global cap on concurrent in-flight requests.

Both are process-wide (the quota belongs to the API key, not to a run) and
are recreated transparently when a different event loop is used (tests,
CLI and server each run their own loop).
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class RateLimiter:
    """Sliding-window limiter allowing at most `rpm` acquisitions per 60 seconds."""

    def __init__(self, rpm: int, *, window_s: float = 60.0) -> None:
        if rpm < 1:
            raise ValueError("rpm must be >= 1")
        self.rpm = rpm
        self.window_s = window_s
        self._times: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available in the current window, then take it."""
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= self.window_s:
                    self._times.popleft()
                if len(self._times) < self.rpm:
                    self._times.append(now)
                    return
                wait = self.window_s - (now - self._times[0])
            await asyncio.sleep(max(wait, 0.01))


class _LoopState:
    def __init__(self) -> None:
        self.limiters: dict[tuple[str, int], RateLimiter] = {}
        self.semaphores: dict[int, asyncio.Semaphore] = {}


_states: dict[int, _LoopState] = {}


def _state() -> _LoopState:
    loop_id = id(asyncio.get_running_loop())
    if loop_id not in _states:
        _states.clear()  # a new loop means the old one is gone; drop stale primitives
        _states[loop_id] = _LoopState()
    return _states[loop_id]


def get_limiter(model: str, rpm: int) -> RateLimiter:
    """Return the shared limiter for `model` in the running event loop."""
    state = _state()
    key = (model, rpm)
    if key not in state.limiters:
        state.limiters[key] = RateLimiter(rpm)
    return state.limiters[key]


@asynccontextmanager
async def concurrency_slot(max_concurrency: int) -> AsyncIterator[None]:
    """Hold one of `max_concurrency` global in-flight request slots."""
    state = _state()
    if max_concurrency not in state.semaphores:
        state.semaphores[max_concurrency] = asyncio.Semaphore(max_concurrency)
    async with state.semaphores[max_concurrency]:
        yield
