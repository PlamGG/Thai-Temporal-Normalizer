"""Unit tests for configurable coarse time ranges and custom configuration injection."""

from datetime import datetime
import pytest
from thai_temporal_normalizer.rules.coarse_ranges import CoarseRangeRuleEngine
from thai_temporal_normalizer.rules.constants import BANGKOK_TZ


@pytest.fixture
def ref_sunday() -> datetime:
    # 2026-11-08 (Sunday)
    return datetime(2026, 11, 8, 12, 0, 0, tzinfo=BANGKOK_TZ)


def test_standard_coarse_ranges(ref_sunday: datetime):
    engine = CoarseRangeRuleEngine()

    # เช้ามืด: 04:00 - 06:00
    res_chao = engine.parse_coarse_span("เช้ามืด", ref_sunday)
    assert len(res_chao) == 1
    assert res_chao[0].type == "RANGE"
    assert res_chao[0].precision == "coarse"
    assert res_chao[0].range_start == "2026-11-08T04:00:00+07:00"
    assert res_chao[0].range_end == "2026-11-08T06:00:00+07:00"
    assert res_chao[0].confidence == 0.85

    # สายๆ: 09:00 - 11:30
    res_sai = engine.parse_coarse_span("ตอนสายๆ", ref_sunday)
    assert len(res_sai) == 1
    assert res_sai[0].range_start == "2026-11-08T09:00:00+07:00"
    assert res_sai[0].range_end == "2026-11-08T11:30:00+07:00"

    # บ่ายๆ: 13:00 - 15:30
    res_bai = engine.parse_coarse_span("ช่วงบ่าย", ref_sunday)
    assert len(res_bai) == 1
    assert res_bai[0].range_start == "2026-11-08T13:00:00+07:00"
    assert res_bai[0].range_end == "2026-11-08T15:30:00+07:00"

    # บ่ายแก่ๆ: 15:00 - 17:00
    res_kae = engine.parse_coarse_span("บ่ายแก่ๆ", ref_sunday)
    assert len(res_kae) == 1
    assert res_kae[0].range_start == "2026-11-08T15:00:00+07:00"
    assert res_kae[0].range_end == "2026-11-08T17:00:00+07:00"

    # เย็นๆ: 16:30 - 18:30
    res_yen = engine.parse_coarse_span("ตอนเย็นๆ", ref_sunday)
    assert len(res_yen) == 1
    assert res_yen[0].range_start == "2026-11-08T16:30:00+07:00"
    assert res_yen[0].range_end == "2026-11-08T18:30:00+07:00"

    # หัวค่ำ: 18:30 - 20:30
    res_hua = engine.parse_coarse_span("หัวค่ำ", ref_sunday)
    assert len(res_hua) == 1
    assert res_hua[0].range_start == "2026-11-08T18:30:00+07:00"
    assert res_hua[0].range_end == "2026-11-08T20:30:00+07:00"

    # ค่ำๆ: 19:00 - 21:30
    res_kam = engine.parse_coarse_span("ช่วงค่ำ", ref_sunday)
    assert len(res_kam) == 1
    assert res_kam[0].range_start == "2026-11-08T19:00:00+07:00"
    assert res_kam[0].range_end == "2026-11-08T21:30:00+07:00"


def test_coarse_ranges_midnight_crossing(ref_sunday: datetime):
    engine = CoarseRangeRuleEngine()

    # ดึกๆ: 22:00 to 02:00 next day (Midnight Crossing)
    res_duek = engine.parse_coarse_span("ดึกๆ", ref_sunday)
    assert len(res_duek) == 1
    assert res_duek[0].range_start == "2026-11-08T22:00:00+07:00"
    assert res_duek[0].range_end == "2026-11-09T02:00:00+07:00"


def test_compound_relative_coarse_ranges(ref_sunday: datetime):
    engine = CoarseRangeRuleEngine()

    # พรุ่งนี้สายๆ (Nov 9: 09:00-11:30)
    res1 = engine.parse_coarse_span("พรุ่งนี้สายๆ", ref_sunday)
    assert len(res1) == 1
    assert res1[0].text_span == "พรุ่งนี้สายๆ"
    assert res1[0].range_start == "2026-11-09T09:00:00+07:00"
    assert res1[0].range_end == "2026-11-09T11:30:00+07:00"

    # เมื่อวานหัวค่ำ (Nov 7: 18:30-20:30)
    res2 = engine.parse_coarse_span("เมื่อวานหัวค่ำ", ref_sunday)
    assert len(res2) == 1
    assert res2[0].range_start == "2026-11-07T18:30:00+07:00"
    assert res2[0].range_end == "2026-11-07T20:30:00+07:00"

    # คืนนี้ดึกๆ (Nov 8 22:00 to Nov 9 02:00)
    res3 = engine.parse_coarse_span("คืนนี้ดึกๆ", ref_sunday)
    assert len(res3) == 1
    assert res3[0].range_start == "2026-11-08T22:00:00+07:00"
    assert res3[0].range_end == "2026-11-09T02:00:00+07:00"


def test_custom_config_injection(ref_sunday: datetime):
    custom = {
        "สายๆ": ("08:00:00", "10:30:00"),
    }
    engine = CoarseRangeRuleEngine(custom_config=custom)

    res = engine.parse_coarse_span("สายๆ", ref_sunday)
    assert len(res) == 1
    assert res[0].range_start == "2026-11-08T08:00:00+07:00"
    assert res[0].range_end == "2026-11-08T10:30:00+07:00"


def test_custom_config_dict_support(ref_sunday: datetime):
    # Defense-in-depth: test dict format with 'start' and 'end' keys
    custom = {
        "สายๆ": {"start": "10:00:00", "end": "12:00:00"},
    }
    engine = CoarseRangeRuleEngine(custom_config=custom)

    res = engine.parse_coarse_span("สายๆ", ref_sunday)
    assert len(res) == 1
    assert res[0].range_start == "2026-11-08T10:00:00+07:00"
    assert res[0].range_end == "2026-11-08T12:00:00+07:00"

