"""Unit tests for Buddhist Era (BE) century pivot and calendar date normalization."""

from datetime import datetime
import pytest
from thai_temporal_normalizer.rules.buddhist_era import (
    BuddhistEraRuleEngine,
    resolve_be_year,
)
from thai_temporal_normalizer.rules.constants import BANGKOK_TZ


@pytest.fixture
def ref_time_2026() -> datetime:
    # 2026 CE = 2569 BE. Pivot = (69 + 10) = 79
    return datetime(2026, 11, 8, 12, 0, 0, tzinfo=BANGKOK_TZ)


def test_dynamic_century_pivot_formula():
    current_be = 2569  # CE 2026

    # Within current century window (YY <= 79 -> 2500 + YY)
    assert resolve_be_year(67, current_be) == 2024  # 2567 - 543 = 2024
    assert resolve_be_year(69, current_be) == 2026  # 2569 - 543 = 2026
    assert resolve_be_year(70, current_be) == 2027  # 2570 - 543 = 2027
    assert resolve_be_year(72, current_be) == 2029  # 2572 - 543 = 2029
    assert resolve_be_year(79, current_be) == 2036  # 2579 - 543 = 2036 (boundary)

    # Outside window (YY > 79 -> 2400 + YY)
    assert resolve_be_year(80, current_be) == 1937  # 2480 - 543 = 1937
    assert resolve_be_year(85, current_be) == 1942  # 2485 - 543 = 1942

    # 4-digit full BE
    assert resolve_be_year(2569, current_be) == 2026
    assert resolve_be_year(2567, current_be) == 2024


def test_short_be_standalone_years(ref_time_2026: datetime):
    engine = BuddhistEraRuleEngine()

    res69 = engine.parse_be_span("ปี 69", ref_time_2026)
    assert len(res69) == 1
    assert res69[0].iso_value == "2026-01-01T00:00:00+07:00"
    assert res69[0].grain == "year"

    res_apostrophe = engine.parse_be_span("ปี'69", ref_time_2026)
    assert len(res_apostrophe) == 1
    assert res_apostrophe[0].iso_value == "2026-01-01T00:00:00+07:00"

    res67 = engine.parse_be_span("ปี 67", ref_time_2026)
    assert len(res67) == 1
    assert res67[0].iso_value == "2024-01-01T00:00:00+07:00"


def test_short_be_full_dates(ref_time_2026: datetime):
    engine = BuddhistEraRuleEngine()

    cases = [
        ("15 มี.ค. 67", "2024-03-15T00:00:00+07:00"),
        ("1 ม.ค. 69", "2026-01-01T00:00:00+07:00"),
        ("31 ธ.ค. 68", "2025-12-31T00:00:00+07:00"),
        ("28 ก.พ. 67", "2024-02-28T00:00:00+07:00"),
        ("23 ต.ค. 67", "2024-10-23T00:00:00+07:00"),
        ("15/03/67", "2024-03-15T00:00:00+07:00"),
        ("01/01/69", "2026-01-01T00:00:00+07:00"),
        ("15-03-67", "2024-03-15T00:00:00+07:00"),
    ]
    for text, exp_iso in cases:
        res = engine.parse_be_span(text, ref_time_2026)
        assert len(res) == 1, f"Failed on '{text}'"
        assert res[0].iso_value == exp_iso
        assert res[0].grain == "day"


def test_seasonal_and_quarter_short_be(ref_time_2026: datetime):
    engine = BuddhistEraRuleEngine()

    # ต้นปี 68: 2025-01-01 to 2025-04-30
    res_ton = engine.parse_be_span("ต้นปี 68", ref_time_2026)
    assert len(res_ton) == 1
    assert res_ton[0].type == "RANGE"
    assert res_ton[0].range_start == "2025-01-01T00:00:00+07:00"
    assert res_ton[0].range_end == "2025-04-30T23:59:59+07:00"

    # กลางปี 67: 2024-05-01 to 2024-08-31
    res_klang = engine.parse_be_span("กลางปี 67", ref_time_2026)
    assert len(res_klang) == 1
    assert res_klang[0].range_start == "2024-05-01T00:00:00+07:00"
    assert res_klang[0].range_end == "2024-08-31T23:59:59+07:00"

    # ปลายปี 69: 2026-09-01 to 2026-12-31
    res_plai = engine.parse_be_span("ปลายปี 69", ref_time_2026)
    assert len(res_plai) == 1
    assert res_plai[0].range_start == "2026-09-01T00:00:00+07:00"
    assert res_plai[0].range_end == "2026-12-31T23:59:59+07:00"

    # ไตรมาส 1 ปี 68: Q1 2025
    res_q1 = engine.parse_be_span("ไตรมาส 1 ปี 68", ref_time_2026)
    assert len(res_q1) == 1
    assert res_q1[0].range_start == "2025-01-01T00:00:00+07:00"
    assert res_q1[0].range_end == "2025-03-31T23:59:59+07:00"

    # สิ้นปี 67: 2024-12-31
    res_end = engine.parse_be_span("สิ้นปี 67", ref_time_2026)
    assert len(res_end) == 1
    assert res_end[0].type == "POINT"
    assert res_end[0].iso_value == "2024-12-31T00:00:00+07:00"


def test_calendar_validation_rejects_impossible_dates(ref_time_2026: datetime):
    engine = BuddhistEraRuleEngine()
    # 30 Feb is invalid
    assert len(engine.parse_be_span("30 ก.พ. 67", ref_time_2026)) == 0
    # 31 April is invalid
    assert len(engine.parse_be_span("31 เม.ย. 68", ref_time_2026)) == 0
