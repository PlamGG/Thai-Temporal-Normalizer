"""ReDoS guardrails, input length validation, and execution timeout protection."""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Callable, TypeVar

MAX_INPUT_LENGTH: int = 120
DEFAULT_TIMEOUT_SEC: float = 0.010  # 10.0 ms

T = TypeVar("T")


class ReDoSProtectionError(TimeoutError, RuntimeError):
    """Raised when an operation exceeds execution timeout guardrail."""


def check_input_length(text: str, strict_mode: bool = False) -> bool:
    """Validate that input text does not exceed the 120 character limit.

    Args:
        text: Input string to validate.
        strict_mode: If True, raises ValueError when limit exceeded.
                     If False, returns False.

    Returns:
        True if text length is within limit, False otherwise.

    Raises:
        ValueError: If text length exceeds limit and strict_mode is True.
    """
    if len(text) > MAX_INPUT_LENGTH:
        if strict_mode:
            raise ValueError(
                f"Input text length ({len(text)} chars) exceeds maximum allowed limit of {MAX_INPUT_LENGTH} characters."
            )
        return False
    return True


class TimeoutGuard:
    """Cooperative context manager measuring elapsed time against SLA threshold."""

    def __init__(self, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> None:
        self.timeout_sec = timeout_sec
        self.start_time: float = 0.0
        self._tripped: bool = False

    def __enter__(self) -> TimeoutGuard:
        self.start_time = time.perf_counter()
        self._tripped = False
        return self

    def check(self) -> None:
        """Check if elapsed time has exceeded threshold."""
        elapsed = time.perf_counter() - self.start_time
        if elapsed > self.timeout_sec:
            self._tripped = True
            raise ReDoSProtectionError(
                f"Execution exceeded timeout guardrail ({elapsed * 1000:.2f} ms > {self.timeout_sec * 1000:.1f} ms)"
            )

    def elapsed_ms(self) -> float:
        """Return elapsed milliseconds since enter."""
        return (time.perf_counter() - self.start_time) * 1000.0

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is None and not self._tripped:
            self.check()


def run_with_timeout(
    func: Callable[..., T],
    *args: Any,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    **kwargs: Any,
) -> T:
    """Execute a function with hard thread-based timeout guardrail.

    Args:
        func: Callable to execute.
        *args: Positional arguments for func.
        timeout_sec: Timeout in seconds (default 0.010 = 10ms).
        **kwargs: Keyword arguments for func.

    Returns:
        Result of func execution.

    Raises:
        ReDoSProtectionError: If execution exceeds timeout_sec.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError as exc:
            raise ReDoSProtectionError(
                f"Execution timed out after {timeout_sec * 1000:.1f} ms"
            ) from exc
