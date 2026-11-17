"""Adversarial stress tests, ReDoS probe, and linguistic boundary tests."""

from __future__ import annotations

import math
import time
from datetime import datetime

import pytest

from thai_temporal_normalizer import parse_thai_temporal
from thai_temporal_normalizer.metrics.range_iou import compute_range_iou
from thai_temporal_normalizer.redos_guard import (
    ReDoSProtectionError,
    TimeoutGuard,
    check_input_length,
    run_with_timeout,
)
from thai_temporal_normalizer.rules.buddhist_era import resolve_be_year

REF_TIME = datetime(2026, 11, 8, 12, 0, 0)  # Sunday, Nov 8, 2026 (BE 2569)


class TestReDoSAndSecurityGuardrails:
    """Adversarial security probes against ReDoS, catastrophic backtracking, and unbounded inputs."""

    def test_input_length_exact_boundaries(self) -> None:
        """Verify strict and non-strict boundary enforcement at 120 and 121 characters."""
        assert check_input_length("x" * 120, strict_mode=False) is True
        assert check_input_length("x" * 121, strict_mode=False) is False
        assert parse_thai_temporal("x" * 121, strict_mode=False) == []

        with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
            check_input_length("x" * 121, strict_mode=True)

        with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
            parse_thai_temporal("x" * 121, strict_mode=True)

    def test_massive_string_instant_rejection(self) -> None:
        """Verify that massive strings up to 50,000 characters are rejected in sub-millisecond time."""
        for length in [121, 500, 1000, 10000, 50000]:
            payload = "พรุ่งนี้ " * (length // 8 + 1)
            assert len(payload) > 120
            t_start = time.perf_counter()
            result = parse_thai_temporal(payload, reference_time=REF_TIME)
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            assert result == []
            assert elapsed_ms < 2.0, f"Rejection too slow ({elapsed_ms:.2f} ms) on len={len(payload)}"

    def test_backtracking_stress_payloads(self) -> None:
        """Probe for catastrophic backtracking across worst-case 120-character strings."""
        stress_payloads = [
            (" " * 120)[:120],
            ("บ่าย" + " " * 110 + "โมง")[:120],
            (" " * 59 + "ตีห้า" + " " * 56)[:120],
            ("๙" * 120)[:120],
            ("9" * 120)[:120],
            ("โมง" * 60)[:120],
            ("ทุ่ม" * 30)[:120],
            ("วัน" * 40)[:120],
            ("ปี" * 60)[:120],
            ("ตี" * 60)[:120],
            ("นาที" * 24)[:120],
            ("อีกสิบห้านาที" * 8)[:120],
            ("ไตรมาสที่ 1 ปี " * 7)[:120],
            ("เมื่อวานซืน" * 10)[:120],
            ("บ่ายโมงบ่ายโมง" * 8)[:120],
            ("คืนวันศุกร์" * 10)[:120],
            ("12/12/12/12/12/12/12/12/12/12/12/12/12/12/12/12/12/12/12")[:120],
            ("23:59:59:59:59:59:59:59:59:59:59:59:59:59:59:59:59:59:59")[:120],
            ("ตี 1 ตี 2 ตี 3 ตี 4 ตี 5 ตี 1 ตี 2 ตี 3 ตี 4 ตี 5 ตี 1 ตี 2 ตี 3")[:120],
            ("31 ก.พ. 30 ก.พ. 31 เม.ย. 31 มิ.ย. 31 ก.ย. 31 พ.ย. 31 ก.พ. 30 ก.พ.")[:120],
            ("วัน\u200B" * 30)[:120],
            ("ก\u0E31\u0E49" * 40)[:120],
            ("\u0E48" * 120)[:120],
        ]

        # Warm-up call to ensure regular expression compilation does not distort payload latency
        _ = parse_thai_temporal("ทดสอบ", reference_time=REF_TIME)

        latencies = []
        for payload in stress_payloads:
            assert len(payload) <= 120
            t_s = time.perf_counter()
            _ = parse_thai_temporal(payload, reference_time=REF_TIME)
            dur_ms = (time.perf_counter() - t_s) * 1000.0
            latencies.append(dur_ms)
            assert dur_ms < 10.0, f"SLA breach: {dur_ms:.2f} ms on {payload[:20]}"

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        assert p95 < 10.0

    def test_timeout_guard_protection(self) -> None:
        """Verify that TimeoutGuard and run_with_timeout trigger ReDoSProtectionError on exceeded budget."""
        with pytest.raises(ReDoSProtectionError):
            with TimeoutGuard(timeout_sec=0.005) as guard:
                time.sleep(0.010)
                guard.check()

        def slow_worker() -> str:
            time.sleep(0.040)
            return "done"

        with pytest.raises(ReDoSProtectionError):
            run_with_timeout(slow_worker, timeout_sec=0.010)


class TestLinguisticBoundaryProbes:
    """Linguistic boundary probes for BE century pivot, night rollover, and clock modifiers."""

    def test_short_be_century_pivot_boundaries(self) -> None:
        """Verify dynamic century pivot formula for BE 2569: Pivot=79 -> 79:2036 CE vs 80:1937 CE."""
        # 1. Direct pivot formula check
        assert resolve_be_year(79, 2569) == 2036
        assert resolve_be_year(80, 2569) == 1937

        # 2. Standalone year tokens
        ent_79 = parse_thai_temporal("ปี 79", reference_time=REF_TIME)
        ent_80 = parse_thai_temporal("ปี 80", reference_time=REF_TIME)
        assert ent_79[0].iso_value == "2036-01-01T00:00:00+07:00"
        assert ent_80[0].iso_value == "1937-01-01T00:00:00+07:00"

        # 3. Full dates with Thai month names
        d_79 = parse_thai_temporal("15 ม.ค. 79", reference_time=REF_TIME)
        d_80 = parse_thai_temporal("15 ม.ค. 80", reference_time=REF_TIME)
        assert d_79[0].iso_value == "2036-01-15T00:00:00+07:00"
        assert d_80[0].iso_value == "1937-01-15T00:00:00+07:00"

        # 4. Slash/Dash dates
        s_79 = parse_thai_temporal("15/01/79", reference_time=REF_TIME)
        s_80 = parse_thai_temporal("15/01/80", reference_time=REF_TIME)
        assert s_79[0].iso_value == "2036-01-15T00:00:00+07:00"
        assert s_80[0].iso_value == "1937-01-15T00:00:00+07:00"

        # 5. Seasonal and Quarter expressions
        q_79 = parse_thai_temporal("ไตรมาส 1 ปี 79", reference_time=REF_TIME)
        q_80 = parse_thai_temporal("ไตรมาส 1 ปี 80", reference_time=REF_TIME)
        assert q_79[0].range_start == "2036-01-01T00:00:00+07:00"
        assert q_80[0].range_start == "1937-01-01T00:00:00+07:00"

        sea_79 = parse_thai_temporal("ต้นปี 79", reference_time=REF_TIME)
        sea_80 = parse_thai_temporal("ต้นปี 80", reference_time=REF_TIME)
        assert sea_79[0].range_start == "2036-01-01T00:00:00+07:00"
        assert sea_80[0].range_start == "1937-01-01T00:00:00+07:00"

    def test_night_rollover_boundaries(self) -> None:
        """Verify night rollover boundary: 05:00 (+1 day) vs 06:00 (no shift)."""
        # Reference is Sunday Nov 8, 2026. Target Friday is Nov 13, 2026.
        # Rollover cases: [00:00, 05:59] shifts Friday 13 -> Saturday 14.
        rollover_cases = [
            ("คืนวันศุกร์ หกทุ่ม", "2026-11-14T00:00:00+07:00"),
            ("คืนวันศุกร์ เที่ยงคืน", "2026-11-14T00:00:00+07:00"),
            ("คืนวันศุกร์ ตีหนึ่ง", "2026-11-14T01:00:00+07:00"),
            ("คืนวันศุกร์ ตีสอง", "2026-11-14T02:00:00+07:00"),
            ("คืนวันศุกร์ ตีสาม", "2026-11-14T03:00:00+07:00"),
            ("คืนวันศุกร์ ตีสี่", "2026-11-14T04:00:00+07:00"),
            ("คืนวันศุกร์ ตีห้า", "2026-11-14T05:00:00+07:00"),
        ]
        for text, expected in rollover_cases:
            res = parse_thai_temporal(text, reference_time=REF_TIME)
            assert len(res) == 1
            assert res[0].iso_value == expected

        # No rollover cases: >= 06:00 or morning/neutral anchors remain Friday 13.
        no_rollover_cases = [
            ("คืนวันศุกร์ หกโมงเช้า", "2026-11-13T06:00:00+07:00"),
            ("คืนวันศุกร์ เจ็ดโมงเช้า", "2026-11-13T07:00:00+07:00"),
            ("คืนวันศุกร์ สี่โมงเย็น", "2026-11-13T16:00:00+07:00"),
            ("คืนวันศุกร์ สองทุ่ม", "2026-11-13T20:00:00+07:00"),
            ("คืนวันศุกร์ สี่ทุ่ม", "2026-11-13T22:00:00+07:00"),
            ("คืนวันศุกร์ ห้าทุ่ม", "2026-11-13T23:00:00+07:00"),
            ("เช้าวันศุกร์ ตีห้า", "2026-11-13T05:00:00+07:00"),
            ("วันศุกร์ ตีห้า", "2026-11-13T05:00:00+07:00"),
        ]
        for text, expected in no_rollover_cases:
            res = parse_thai_temporal(text, reference_time=REF_TIME)
            assert len(res) == 1
            assert res[0].iso_value == expected

    def test_clock_modifier_combinations(self) -> None:
        """Verify clock modifier parsing: บ่ายโมงสี่สิบห้า, ทุ่มครึ่ง, หกทุ่ม, เที่ยงคืน."""
        cases = [
            ("บ่ายโมงสี่สิบห้า", "13:45:00"),
            ("ทุ่มครึ่ง", "19:30:00"),
            ("หกทุ่ม", "00:00:00"),
            ("เที่ยงคืน", "00:00:00"),
            ("เที่ยงตรง", "12:00:00"),
            ("เที่ยงวัน", "12:00:00"),
            ("บ่ายสองโมงสี่สิบห้า", "14:45:00"),
            ("บ่ายสามโมงตรง", "15:00:00"),
            ("หนึ่งทุ่มตรง", "19:00:00"),
            ("สองทุ่มครึ่ง", "20:30:00"),
            ("สามทุ่มสี่สิบห้า", "21:45:00"),
            ("สี่ทุ่มยี่สิบห้า", "22:25:00"),
            ("ห้าทุ่มห้าสิบห้า", "23:55:00"),
            ("ตีสามครึ่ง", "03:30:00"),
            ("อีกสิบห้านาทีบ่ายสอง", "13:45:00"),
            ("อีกยี่สิบนาทีเที่ยง", "11:40:00"),
        ]
        for text, exp_time in cases:
            res = parse_thai_temporal(text, reference_time=REF_TIME)
            assert len(res) == 1
            assert exp_time in res[0].iso_value


class TestContinuousRangeIoUMidnightWrap:
    """Continuous Range IoU tests with midnight wrapping."""

    def test_range_iou_midnight_wrap_exact_values(self) -> None:
        """Verify continuous interval IoU under midnight wrapping."""
        # 1. Identical midnight interval [22:00, 02:00] vs [22:00, 02:00] == 1.0
        iou1 = compute_range_iou("22:00", "02:00", "22:00", "02:00")
        assert math.isclose(iou1, 1.0)

        # 2. Overlap [22:00, 02:00] vs [00:00, 04:00] == 2h / 6h = 0.333333
        iou2 = compute_range_iou("22:00", "02:00", "00:00", "04:00")
        assert math.isclose(iou2, 1.0 / 3.0, rel_tol=1e-5)

        # 3. [23:00, 01:00] vs [00:00, 02:00] == 1h / 3h = 0.333333
        iou3 = compute_range_iou("23:00", "01:00", "00:00", "02:00")
        assert math.isclose(iou3, 1.0 / 3.0, rel_tol=1e-5)

        # 4. Disjoint across midnight: [22:00, 23:30] vs [00:30, 02:00] == 0.0
        iou4 = compute_range_iou("22:00", "23:30", "00:30", "02:00")
        assert math.isclose(iou4, 0.0)

        # 5. Continuous datetimes with ISO strings across midnight
        dt_iou1 = compute_range_iou(
            "2026-11-08T22:00:00+07:00",
            "2026-11-09T02:00:00+07:00",
            "2026-11-08T22:00:00+07:00",
            "2026-11-09T02:00:00+07:00",
        )
        assert math.isclose(dt_iou1, 1.0)

        dt_iou2 = compute_range_iou(
            "2026-11-08T22:00:00+07:00",
            "2026-11-09T02:00:00+07:00",
            "2026-11-09T00:00:00+07:00",
            "2026-11-09T04:00:00+07:00",
        )
        assert math.isclose(dt_iou2, 1.0 / 3.0, rel_tol=1e-5)
