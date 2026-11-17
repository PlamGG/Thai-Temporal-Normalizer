"""Unit tests for Thai 6-Hour Clock and official 24-hour time parsing."""

from datetime import time
import pytest
from thai_temporal_normalizer.rules.clock import ClockRuleEngine


@pytest.fixture
def clock_engine() -> ClockRuleEngine:
    return ClockRuleEngine()


def test_clock_early_morning_ti(clock_engine: ClockRuleEngine):
    res1 = clock_engine.parse_time_span("ตื่นตอนตีหนึ่ง")
    assert len(res1) == 1
    assert res1[0][0] == "ตีหนึ่ง"
    assert res1[0][3] == time(1, 0)
    assert res1[0][4] == "hour"

    res2 = clock_engine.parse_time_span("ตี 3 ครึ่ง")
    assert len(res2) == 1
    assert res2[0][3] == time(3, 30)
    assert res2[0][4] == "minute"

    res3 = clock_engine.parse_time_span("ตีสองสิบห้า")
    assert len(res3) == 1
    assert res3[0][3] == time(2, 15)
    assert res3[0][4] == "minute"

    res4 = clock_engine.parse_time_span("ตีห้า")
    assert len(res4) == 1
    assert res4[0][3] == time(5, 0)


def test_clock_morning_mong(clock_engine: ClockRuleEngine):
    cases = [
        ("โมงเช้า", time(7, 0), "hour"),
        ("แปดโมงเช้า", time(8, 0), "hour"),
        ("เก้าโมง", time(9, 0), "hour"),
        ("สิบโมงเช้า", time(10, 0), "hour"),
        ("สิบเอ็ดโมง", time(11, 0), "hour"),
        ("เก้าโมงครึ่ง", time(9, 30), "minute"),
        ("สิบโมงสี่สิบห้า", time(10, 45), "minute"),
    ]
    for text, exp_time, exp_grain in cases:
        res = clock_engine.parse_time_span(text)
        assert len(res) == 1, f"Failed on '{text}'"
        assert res[0][3] == exp_time
        assert res[0][4] == exp_grain


def test_clock_noon_and_midnight(clock_engine: ClockRuleEngine):
    assert clock_engine.parse_time_span("เที่ยงตรง")[0][3] == time(12, 0)
    assert clock_engine.parse_time_span("เที่ยงวัน")[0][3] == time(12, 0)
    assert clock_engine.parse_time_span("เที่ยงครึ่ง")[0][3] == time(12, 30)

    assert clock_engine.parse_time_span("เที่ยงคืน")[0][3] == time(0, 0)
    assert clock_engine.parse_time_span("หกทุ่ม")[0][3] == time(0, 0)
    assert clock_engine.parse_time_span("เที่ยงคืนครึ่ง")[0][3] == time(0, 30)


def test_clock_afternoon_bai(clock_engine: ClockRuleEngine):
    cases = [
        ("บ่ายโมง", time(13, 0), "hour"),
        ("บ่ายสอง", time(14, 0), "hour"),
        ("บ่ายสาม", time(15, 0), "hour"),
        ("บ่ายโมงครึ่ง", time(13, 30), "minute"),
        ("บ่ายสองสี่สิบห้า", time(14, 45), "minute"),
    ]
    for text, exp_time, exp_grain in cases:
        res = clock_engine.parse_time_span(text)
        assert len(res) == 1, f"Failed on '{text}'"
        assert res[0][3] == exp_time
        assert res[0][4] == exp_grain


def test_clock_evening_and_night(clock_engine: ClockRuleEngine):
    # Evening
    assert clock_engine.parse_time_span("สี่โมงเย็น")[0][3] == time(16, 0)
    assert clock_engine.parse_time_span("ห้าโมงเย็น")[0][3] == time(17, 0)
    assert clock_engine.parse_time_span("หกโมงเย็น")[0][3] == time(18, 0)
    assert clock_engine.parse_time_span("ห้าโมงครึ่ง")[0][3] == time(17, 30)

    # Night
    assert clock_engine.parse_time_span("หนึ่งทุ่ม")[0][3] == time(19, 0)
    assert clock_engine.parse_time_span("ทุ่มนึง")[0][3] == time(19, 0)
    assert clock_engine.parse_time_span("สองทุ่ม")[0][3] == time(20, 0)
    assert clock_engine.parse_time_span("สามทุ่ม")[0][3] == time(21, 0)
    assert clock_engine.parse_time_span("สี่ทุ่ม")[0][3] == time(22, 0)
    assert clock_engine.parse_time_span("ห้าทุ่ม")[0][3] == time(23, 0)
    assert clock_engine.parse_time_span("ทุ่มครึ่ง")[0][3] == time(19, 30)
    assert clock_engine.parse_time_span("สี่ทุ่มครึ่ง")[0][3] == time(22, 30)


def test_clock_official_and_subtractive(clock_engine: ClockRuleEngine):
    # 24-hour official format
    res_24 = clock_engine.parse_time_span("เริ่มประชุม 14.45 น.")
    assert len(res_24) == 1
    assert res_24[0][3] == time(14, 45)

    res_colon = clock_engine.parse_time_span("08:00 น.")
    assert len(res_colon) == 1
    assert res_colon[0][3] == time(8, 0)

    res_nalika = clock_engine.parse_time_span("สิบสี่นาฬิกาสามสิบนาที")
    assert len(res_nalika) == 1
    assert res_nalika[0][3] == time(14, 30)

    # Subtractive minute
    res_sub = clock_engine.parse_time_span("อีกสิบห้านาทีบ่ายสอง")
    assert len(res_sub) == 1
    assert res_sub[0][3] == time(13, 45)
