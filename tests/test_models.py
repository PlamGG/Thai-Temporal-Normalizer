"""Unit tests for TemporalEntity data models and type definitions."""

import pytest
from thai_temporal_normalizer.models import TemporalEntity, WeekdayAnchorPolicy


def test_temporal_entity_initialization_and_properties():
    ent = TemporalEntity(
        text_span="บ่ายสาม",
        start_char=5,
        end_char=12,
        type="POINT",
        iso_value="2026-11-08T15:00:00+07:00",
        precision="exact",
        confidence=1.0,
        grain="hour",
    )

    assert ent.text_span == "บ่ายสาม"
    assert ent.start_char == 5
    assert ent.end_char == 12
    assert ent.type == "POINT"
    assert ent.iso_value == "2026-11-08T15:00:00+07:00"
    assert ent.precision == "exact"
    assert ent.confidence == 1.0
    assert ent.grain == "hour"

    # Properties
    assert ent.span == (5, 12)
    assert ent.value == "2026-11-08T15:00:00+07:00"
    assert ent.normalized_range == (None, None)


def test_range_temporal_entity_properties():
    ent = TemporalEntity(
        text_span="สายๆ",
        start_char=0,
        end_char=4,
        type="RANGE",
        range_start="2026-11-08T09:00:00+07:00",
        range_end="2026-11-08T11:30:00+07:00",
        precision="coarse",
        confidence=0.8,
    )

    assert ent.span == (0, 4)
    assert ent.value is None
    assert ent.normalized_range == (
        "2026-11-08T09:00:00+07:00",
        "2026-11-08T11:30:00+07:00",
    )


def test_temporal_entity_validation_errors():
    # Invalid type
    with pytest.raises(ValueError, match="Invalid type"):
        TemporalEntity(text_span="x", start_char=0, end_char=1, type="INVALID")

    # Invalid precision
    with pytest.raises(ValueError, match="Invalid precision"):
        TemporalEntity(text_span="x", start_char=0, end_char=1, type="POINT", precision="super_exact")

    # Negative start_char
    with pytest.raises(ValueError, match="start_char must be non-negative"):
        TemporalEntity(text_span="x", start_char=-1, end_char=1, type="POINT")

    # end_char < start_char
    with pytest.raises(ValueError, match="end_char .* cannot be less than start_char"):
        TemporalEntity(text_span="x", start_char=10, end_char=5, type="POINT")

    # Confidence out of bounds
    with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
        TemporalEntity(text_span="x", start_char=0, end_char=1, type="POINT", confidence=1.5)


def test_dict_serialization_roundtrip():
    original = TemporalEntity(
        text_span="คืนนี้ดึกๆ",
        start_char=0,
        end_char=10,
        type="RANGE",
        range_start="2026-11-08T22:00:00+07:00",
        range_end="2026-11-09T02:00:00+07:00",
        precision="coarse",
        confidence=0.8,
        grain="auto",
    )
    d = original.to_dict()
    assert isinstance(d, dict)
    assert d["text_span"] == "คืนนี้ดึกๆ"
    assert d["type"] == "RANGE"

    restored = TemporalEntity.from_dict(d)
    assert restored == original


def test_weekday_anchor_policy_enum():
    assert WeekdayAnchorPolicy.NEXT_OCCURRENCE == "next_occurrence"
    assert WeekdayAnchorPolicy.NEXT_WEEK == "next_week"
