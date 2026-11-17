"""Unit tests for ReDoS guardrails, input length validation, and execution timeout."""

import time
import pytest
from thai_temporal_normalizer.normalizer import parse_thai_temporal
from thai_temporal_normalizer.redos_guard import (
    ReDoSProtectionError,
    TimeoutGuard,
    check_input_length,
    run_with_timeout,
)


def test_input_length_guard_boundaries():
    # Exactly 120 characters: valid
    text_120 = "ก" * 120
    assert len(text_120) == 120
    assert check_input_length(text_120, strict_mode=False) is True
    assert check_input_length(text_120, strict_mode=True) is True

    # 121 characters: invalid
    text_121 = "ก" * 121
    assert len(text_121) == 121
    assert check_input_length(text_121, strict_mode=False) is False
    with pytest.raises(ValueError, match="exceeds maximum allowed limit of 120 characters"):
        check_input_length(text_121, strict_mode=True)


def test_parse_thai_temporal_length_rejection():
    text_121 = "พรุ่งนี้ " + ("ข้อความทดสอบยาวเกินไป " * 10)
    assert len(text_121) > 120

    # Non-strict returns empty list
    assert parse_thai_temporal(text_121, strict_mode=False) == []

    # Strict raises ValueError
    with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
        parse_thai_temporal(text_121, strict_mode=True)


def test_repetitive_input_no_catastrophic_backtracking():
    # Repetitive pattern designed to induce backtracking in naive nested regexes
    malicious_pattern = "ช่วงบ่าย" * 15  # 120 chars exactly
    assert len(malicious_pattern) == 120

    t0 = time.perf_counter()
    res = parse_thai_temporal(malicious_pattern)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Must complete far below the 5.0ms SLA threshold
    assert elapsed_ms < 5.0
    assert isinstance(res, list)


def test_timeout_guard_cooperative():
    with TimeoutGuard(timeout_sec=0.010) as guard:
        # Should not raise when within limit
        guard.check()

    # When simulated sleep exceeds timeout
    with TimeoutGuard(timeout_sec=0.005) as guard:
        time.sleep(0.010)
        with pytest.raises(ReDoSProtectionError, match="Execution exceeded timeout guardrail"):
            guard.check()


def test_run_with_timeout_safeguard():
    def quick_task():
        return 42

    def slow_task():
        time.sleep(0.050)
        return "slow"

    assert run_with_timeout(quick_task, timeout_sec=0.010) == 42

    with pytest.raises(ReDoSProtectionError, match="timed out"):
        run_with_timeout(slow_task, timeout_sec=0.005)


def test_timeout_guard_exit_enforcement():
    # Context manager without cooperative check() must still enforce timeout upon exit
    with pytest.raises(TimeoutError):
        with TimeoutGuard(timeout_sec=0.001):
            time.sleep(0.008)
