"""Unit tests for relative days, weeks, and weekday normalization."""

from datetime import date, datetime
import pytest
from thai_temporal_normalizer.models import WeekdayAnchorPolicy
from thai_temporal_normalizer.rules.constants import BANGKOK_TZ
from thai_temporal_normalizer.rules.relative import RelativeRuleEngine


@pytest.fixture
def ref_sunday() -> datetime:
    # 2026-11-08 is a Sunday (weekday=6)
    return datetime(2026, 11, 8, 12, 0, 0, tzinfo=BANGKOK_TZ)


def test_relative_single_days(ref_sunday: datetime):
    engine = RelativeRuleEngine()

    cases = [
        ("วันนี้", date(2026, 11, 8)),
        ("พรุ่งนี้", date(2026, 11, 9)),
        ("มะรืนนี้", date(2026, 11, 10)),
        ("เมื่อวาน", date(2026, 11, 7)),
        ("เมื่อวานนี้", date(2026, 11, 7)),
        ("วานซืน", date(2026, 11, 6)),
        ("เมื่อวานซืน", date(2026, 11, 6)),
        ("เจอกัน พน", date(2026, 11, 9)),
        ("ส่งของ มรน", date(2026, 11, 10)),
    ]
    for text, exp_date in cases:
        res = engine.parse_relative_span(text, ref_time=ref_sunday)
        assert len(res) == 1, f"Failed on '{text}'"
        assert res[0][3] == exp_date


def test_relative_numeric_offsets(ref_sunday: datetime):
    engine = RelativeRuleEngine()

    assert engine.parse_relative_span("อีก 3 วัน", ref_sunday)[0][3] == date(2026, 11, 11)
    assert engine.parse_relative_span("อีก 5 วัน", ref_sunday)[0][3] == date(2026, 11, 13)
    assert engine.parse_relative_span("2 วันก่อน", ref_sunday)[0][3] == date(2026, 11, 6)
    assert engine.parse_relative_span("3 วันที่แล้ว", ref_sunday)[0][3] == date(2026, 11, 5)
    assert engine.parse_relative_span("อีก 1 สัปดาห์", ref_sunday)[0][3] == date(2026, 11, 15)
    assert engine.parse_relative_span("อีกสองวัน", ref_sunday)[0][3] == date(2026, 11, 10)


def test_relative_weekday_resolution(ref_sunday: datetime):
    engine = RelativeRuleEngine()

    # Sunday Nov 8 -> Wednesday Nov 11 (+3d)
    assert engine.parse_relative_span("พุธหน้า", ref_sunday)[0][3] == date(2026, 11, 11)
    # Sunday Nov 8 -> Past Tuesday Nov 3 (-5d)
    assert engine.parse_relative_span("อังคารที่แล้ว", ref_sunday)[0][3] == date(2026, 11, 3)
    # Sunday Nov 8 -> Friday Nov 13 (+5d)
    assert engine.parse_relative_span("ศุกร์นี้", ref_sunday)[0][3] == date(2026, 11, 13)
    # Sunday Nov 8 -> Upcoming Monday Nov 9 (+1d)
    assert engine.parse_relative_span("จันทร์ที่จะถึงนี้", ref_sunday)[0][3] == date(2026, 11, 9)
    # Sunday Nov 8 -> Thursday Nov 12 (+4d)
    assert engine.parse_relative_span("พฤหัสหน้า", ref_sunday)[0][3] == date(2026, 11, 12)
    # Sunday Nov 8 -> Saturday Nov 14 (+6d)
    assert engine.parse_relative_span("เสาร์หน้า", ref_sunday)[0][3] == date(2026, 11, 14)
    # Sunday Nov 8 -> Sunday Nov 15 (+7d)
    assert engine.parse_relative_span("อาทิตย์หน้า", ref_sunday)[0][3] == date(2026, 11, 15)
    # Days ahead <= 2 with 'วัน...หน้า': Monday Nov 16 (+8d), Tuesday Nov 17 (+9d)
    assert engine.parse_relative_span("วันจันทร์หน้า", ref_sunday)[0][3] == date(2026, 11, 16)
    assert engine.parse_relative_span("วันอังคารหน้า", ref_sunday)[0][3] == date(2026, 11, 17)


def test_relative_weekday_anchor_policy():
    ref_mon = datetime(2026, 11, 9, 12, 0, 0, tzinfo=BANGKOK_TZ)  # Monday Nov 9

    # Under NEXT_OCCURRENCE: "พุธหน้า" -> Wednesday Nov 11 (+2d)
    engine_occ = RelativeRuleEngine(policy=WeekdayAnchorPolicy.NEXT_OCCURRENCE)
    res_occ = engine_occ.parse_relative_span("พุธหน้า", ref_time=ref_mon)
    assert res_occ[0][3] == date(2026, 11, 11)

    # Under NEXT_WEEK: "พุธหน้า" -> Wednesday Nov 18 (+9d)
    engine_wk = RelativeRuleEngine(policy=WeekdayAnchorPolicy.NEXT_WEEK)
    res_wk = engine_wk.parse_relative_span("พุธหน้า", ref_time=ref_mon)
    assert res_wk[0][3] == date(2026, 11, 18)


def test_relative_embedded_sentences(ref_sunday: datetime):
    engine = RelativeRuleEngine()

    res1 = engine.parse_relative_span("เจอกันพรุ่งนี้นะ", ref_time=ref_sunday)
    assert len(res1) == 1
    assert res1[0][0] == "พรุ่งนี้"
    assert res1[0][1] == 6
    assert res1[0][2] == 14
    assert res1[0][3] == date(2026, 11, 9)

    res2 = engine.parse_relative_span("ส่งรายงานภายในวันมะรืนนี้", ref_time=ref_sunday)
    assert len(res2) == 1
    assert res2[0][0] == "มะรืนนี้"
    assert res2[0][3] == date(2026, 11, 10)
