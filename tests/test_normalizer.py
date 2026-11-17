"""Integration unit tests for main parse_thai_temporal public API."""

from datetime import datetime
import pytest
from thai_temporal_normalizer import parse_thai_temporal
from thai_temporal_normalizer.rules.constants import BANGKOK_TZ


@pytest.fixture
def ref_time() -> datetime:
    return datetime(2026, 11, 8, 12, 0, 0, tzinfo=BANGKOK_TZ)


def test_normalizer_default_call_no_ref_time():
    res = parse_thai_temporal("พรุ่งนี้")
    assert len(res) == 1
    assert res[0].text_span == "พรุ่งนี้"
    assert res[0].type == "POINT"
    assert res[0].iso_value is not None


def test_normalizer_compound_date_and_time(ref_time: datetime):
    res = parse_thai_temporal("พรุ่งนี้ บ่ายสอง", reference_time=ref_time)
    assert len(res) == 1
    assert res[0].text_span == "พรุ่งนี้ บ่ายสอง"
    assert res[0].start_char == 0
    assert res[0].end_char == 16
    assert res[0].iso_value == "2026-11-09T14:00:00+07:00"


def test_normalizer_multiple_entities_in_sentence(ref_time: datetime):
    # Multiple distinct temporal entities
    text = "นัดประชุมพรุ่งนี้ บ่ายสอง ถึง สี่โมงเย็น"
    res = parse_thai_temporal(text, reference_time=ref_time)
    assert len(res) >= 2
    # First: 'พรุ่งนี้ บ่ายสอง'
    assert res[0].text_span == "พรุ่งนี้ บ่ายสอง"
    assert res[0].iso_value == "2026-11-09T14:00:00+07:00"
    # Second: 'สี่โมงเย็น'
    assert res[1].text_span == "สี่โมงเย็น"
    assert res[1].iso_value == "2026-11-08T16:00:00+07:00"


def test_normalizer_conversational_noise_returns_empty(ref_time: datetime):
    assert parse_thai_temporal("สวัสดีครับ สบายดีไหมครับ", reference_time=ref_time) == []
    assert parse_thai_temporal("กินข้าวกันเถอะ หิวแล้ว", reference_time=ref_time) == []
    assert parse_thai_temporal("ขอบคุณมากๆ เลยนะ", reference_time=ref_time) == []


def test_normalizer_impossible_calendar_dates(ref_time: datetime):
    # 30 ก.พ. is physically impossible
    assert parse_thai_temporal("เจอกัน 30 ก.พ. 67", reference_time=ref_time, strict_mode=False) == []
    with pytest.raises(ValueError, match="Invalid calendar date"):
        parse_thai_temporal("เจอกัน 30 ก.พ. 67", reference_time=ref_time, strict_mode=True)


def test_normalizer_impossible_clock_times(ref_time: datetime):
    # 25:70 น. or ตี 8 are impossible
    assert parse_thai_temporal("เวลา 25:70 น.", reference_time=ref_time, strict_mode=False) == []
    with pytest.raises(ValueError, match="Malformed or impossible clock time"):
        parse_thai_temporal("เวลา 25:70 น.", reference_time=ref_time, strict_mode=True)

    assert parse_thai_temporal("นัดตี 8", reference_time=ref_time, strict_mode=False) == []
    with pytest.raises(ValueError, match="Malformed or impossible clock time"):
        parse_thai_temporal("นัดตี 8", reference_time=ref_time, strict_mode=True)


def test_normalizer_strict_mode_coarse_clarification(ref_time: datetime):
    # Coarse range requires clarification in strict_mode
    with pytest.raises(ValueError, match="coarse time expression requires clarification"):
        parse_thai_temporal("เจอกันสายๆ นะ", reference_time=ref_time, strict_mode=True)

    with pytest.raises(ValueError, match="coarse time expression requires clarification"):
        parse_thai_temporal("เจอกันช่วงดึก", reference_time=ref_time, strict_mode=True)


def test_normalizer_custom_config_override(ref_time: datetime):
    custom = {"สายๆ": ("08:15:00", "10:45:00")}
    res = parse_thai_temporal("เจอกันสายๆ", reference_time=ref_time, custom_config=custom)
    assert len(res) == 1
    assert res[0].range_start == "2026-11-08T08:15:00+07:00"
    assert res[0].range_end == "2026-11-08T10:45:00+07:00"


def test_normalizer_timezone_parameter(ref_time: datetime):
    # UTC timezone outputs UTC ISO string
    res_utc = parse_thai_temporal("วันนี้", reference_time=ref_time, timezone="UTC")
    assert len(res_utc) == 1
    assert "+00:00" in res_utc[0].iso_value

    # Asia/Bangkok timezone outputs +07:00
    res_bkk = parse_thai_temporal("วันนี้", reference_time=ref_time, timezone="Asia/Bangkok")
    assert len(res_bkk) == 1
    assert "+07:00" in res_bkk[0].iso_value

    # Invalid timezone in non-strict mode falls back gracefully
    res_fallback = parse_thai_temporal("วันนี้", reference_time=ref_time, timezone="INVALID_TZ", strict_mode=False)
    assert len(res_fallback) == 1

    # Invalid timezone in strict mode raises ValueError
    with pytest.raises(ValueError, match="Invalid timezone"):
        parse_thai_temporal("วันนี้", reference_time=ref_time, timezone="INVALID_TZ", strict_mode=True)


def test_normalizer_strict_mode_feb29_non_leap(ref_time: datetime):
    # 29 ก.พ. 68 (2568 BE = 2025 CE, non-leap year): non-strict returns [], strict raises ValueError
    assert parse_thai_temporal("29 ก.พ. 68", reference_time=ref_time, strict_mode=False) == []
    with pytest.raises(ValueError, match="Invalid calendar date"):
        parse_thai_temporal("29 ก.พ. 68", reference_time=ref_time, strict_mode=True)

    # 29 ก.พ. 67 (2567 BE = 2024 CE, leap year): valid in both non-strict and strict
    res_leap = parse_thai_temporal("29 ก.พ. 67", reference_time=ref_time, strict_mode=True)
    assert len(res_leap) == 1
    assert res_leap[0].iso_value == "2024-02-29T00:00:00+07:00"
