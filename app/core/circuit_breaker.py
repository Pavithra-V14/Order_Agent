"""
Circuit breaker - architecture doc Layer 5's reliability pattern, applied
per external dependency (payment, carrier, WMS). Real state machine
(CLOSED -> OPEN -> HALF_OPEN -> CLOSED), not just a retry loop - the
whole point is to FAIL FAST once a dependency is clearly down, rather
than keep hammering it with retries that will predictably also fail.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit is OPEN - the
    caller (Execution Agent) catches this and routes the case to
    'pending_retry' rather than treating it as a normal failure."""


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 3
    reset_timeout_seconds: float = 30.0
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    call_attempts: int = field(default=0, init=False)
    call_successes: int = field(default=0, init=False)
    call_failures: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at >= self.reset_timeout_seconds:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def call(self, fn: Callable[[], T]) -> T:
        """Executes fn() through the breaker. Raises CircuitOpenError
        without calling fn() at all if the circuit is OPEN - this is what
        "fail fast" actually means, not just "retry a few times and give up." """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit '{self.name}' is OPEN ({self._consecutive_failures} consecutive "
                f"failures) - failing fast without attempting the call. Will retry after "
                f"{self.reset_timeout_seconds}s."
            )

        self.call_attempts += 1
        try:
            result = fn()
        except Exception:
            self.call_failures += 1
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
            raise
        else:
            self.call_successes += 1
            self._consecutive_failures = 0
            self._state = CircuitState.CLOSED
            return result

    def reset(self) -> None:
        """Test helper - force back to a clean CLOSED state."""
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self.call_attempts = 0
        self.call_successes = 0
        self.call_failures = 0


_breakers = {}


def get_circuit_breaker(name: str, failure_threshold: int = 3, reset_timeout_seconds: float = 30.0) -> CircuitBreaker:
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name=name, failure_threshold=failure_threshold,
                                          reset_timeout_seconds=reset_timeout_seconds)
    return _breakers[name]


def reset_all_breakers() -> None:
    _breakers.clear()
