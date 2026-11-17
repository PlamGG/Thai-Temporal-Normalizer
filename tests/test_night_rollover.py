"""Unit tests for night-time day-rollover normalization engine."""

from datetime import datetime
import pytest
from thai_temporal_normalizer.rules.constants import BANGKOK_TZ
from thai_temporal_normalizer.rules.night_rollover import NightRolloverRuleEngine


@pytest.fixture
def ref_sunday() -> datetime:
    # 2026-11-08 (Sunday 12:00:00)
    return datetime(2026, 11, 8, 12, 0, 0, tzinfo=BANGKOK_TZ)


@pytest.fixture
def rollover_engine() -> NightRolloverRuleEngine:
    return NightRolloverRuleEngine()


def test_positive_night_rollovers(rollover_engine: NightRolloverRuleEngine, ref_sunday: datetime):
    # คืนวันศุกร์ (Nov 13) + ตีสอง -> Saturday Nov 14 02:00:00
    res1 = rollover_engine.parse_rollover_span("คืนวันศุกร์ ตีสอง", ref_sunday)
    assert len(res1) == 1
    assert res1[0].text_span == "คืนวันศุกร์ ตีสอง"
    assert res1[0].iso_value == "2026-11-14T02:00:00+07:00"

    # ดึกวันจันทร์ (Nov 9) + ตีหนึ่งครึ่ง -> Tuesday Nov 10 01:30:00
    res2 = rollover_engine.parse_rollover_span("ดึกวันจันทร์ ตีหนึ่งครึ่ง", ref_sunday)
    assert len(res2) == 1
    assert res2[0].iso_value == "2026-11-10T01:30:00+07:00"

    # คืนนี้ (Nov 8) + ตีสาม -> Monday Nov 9 03:00:00
    res3 = rollover_engine.parse_rollover_span("คืนนี้ ตีสาม", ref_sunday)
    assert len(res3) == 1
    assert res3[0].iso_value == "2026-11-09T03:00:00+07:00"

    # เมื่อคืน (Nov 7) + ตีสอง -> Sunday Nov 8 02:00:00
    res4 = rollover_engine.parse_rollover_span("เมื่อคืน ตีสอง", ref_sunday)
    assert len(res4) == 1
    assert res4[0].iso_value == "2026-11-08T02:00:00+07:00"

    # คืนวันพุธ (Nov 11) + เที่ยงคืนครึ่ง -> Thursday Nov 12 00:30:00
    res5 = rollover_engine.parse_rollover_span("คืนวันพุธ เที่ยงคืนครึ่ง", ref_sunday)
    assert len(res5) == 1
    assert res5[0].iso_value == "2026-11-12T00:30:00+07:00"


def test_negative_controls_no_rollover(rollover_engine: NightRolloverRuleEngine, ref_sunday: datetime):
    # Evening time: คืนวันศุกร์ สี่ทุ่ม -> Friday Nov 13 22:00:00 (No shift)
    res1 = rollover_engine.parse_rollover_span("คืนวันศุกร์ สี่ทุ่ม", ref_sunday)
    assert len(res1) == 1
    assert res1[0].iso_value == "2026-11-13T22:00:00+07:00"

    # Evening time: ดึกวันจันทร์ ห้าทุ่ม -> Monday Nov 9 23:00:00 (No shift)
    res2 = rollover_engine.parse_rollover_span("ดึกวันจันทร์ ห้าทุ่ม", ref_sunday)
    assert len(res2) == 1
    assert res2[0].iso_value == "2026-11-09T23:00:00+07:00"

    # Morning anchor prefix: เช้าวันเสาร์ ตีสอง -> Saturday Nov 14 02:00:00 (No shift)
    res3 = rollover_engine.parse_rollover_span("เช้าวันเสาร์ ตีสอง", ref_sunday)
    assert len(res3) == 1
    assert res3[0].iso_value == "2026-11-14T02:00:00+07:00"

    # Missing night prefix: วันศุกร์ ตีสอง -> Friday Nov 13 02:00:00 (No shift)
    res4 = rollover_engine.parse_rollover_span("วันศุกร์ ตีสอง", ref_sunday)
    assert len(res4) == 1
    assert res4[0].iso_value == "2026-11-13T02:00:00+07:00"
