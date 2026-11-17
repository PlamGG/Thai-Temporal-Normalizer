"""Unit tests for evaluation metrics: Range IoU, Span F1, and Value Accuracy."""

from datetime import datetime
import pytest
from thai_temporal_normalizer.metrics import (
    compute_range_iou,
    compute_span_f1,
    compute_value_accuracy,
    decompose_midnight_interval,
)
from thai_temporal_normalizer.models import TemporalEntity


def test_continuous_range_iou_standard():
    # Exact match
    t_start = "2026-11-08T09:00:00+07:00"
    t_end = "2026-11-08T11:30:00+07:00"
    assert compute_range_iou(t_start, t_end, t_start, t_end) == 1.0

    # Disjoint intervals -> 0.0
    s1, e1 = "2026-11-08T09:00:00+07:00", "2026-11-08T11:30:00+07:00"
    s2, e2 = "2026-11-08T13:00:00+07:00", "2026-11-08T15:30:00+07:00"
    assert compute_range_iou(s1, e1, s2, e2) == 0.0

    # Partial overlap: 09:00-11:00 vs 10:00-12:00 -> intersection 1h, union 3h -> 1/3
    s_a, e_a = datetime(2026, 11, 8, 9, 0), datetime(2026, 11, 8, 11, 0)
    s_b, e_b = datetime(2026, 11, 8, 10, 0), datetime(2026, 11, 8, 12, 0)
    assert pytest.approx(compute_range_iou(s_a, e_a, s_b, e_b), rel=1e-3) == 1.0 / 3.0


def test_continuous_range_iou_midnight_crossing():
    # Midnight crossing case from PRD §7.2:
    # Pred: 22:00 to 02:00 (4h duration)
    # True: 23:00 to 01:00 (2h duration)
    # True is completely inside Pred: Intersection = 2h, Union = 4h -> IoU = 2/4 = 0.50
    pred_s = "2026-11-08T22:00:00+07:00"
    pred_e = "2026-11-09T02:00:00+07:00"
    gt_s = "2026-11-08T23:00:00+07:00"
    gt_e = "2026-11-09T01:00:00+07:00"

    iou = compute_range_iou(pred_s, pred_e, gt_s, gt_e)
    assert pytest.approx(iou, rel=1e-3) == 0.50

    # Also test via time strings '22:00' and '02:00'
    iou_time_str = compute_range_iou("22:00", "02:00", "23:00", "01:00")
    assert pytest.approx(iou_time_str, rel=1e-3) == 0.50


def test_decompose_midnight_interval():
    # 22:00 (79200s) to 02:00 (7200s)
    decomp = decompose_midnight_interval("22:00", "02:00")
    assert len(decomp) == 2
    assert decomp[0] == (79200.0, 86400.0)  # [22:00, 24:00)
    assert decomp[1] == (0.0, 7200.0)       # [00:00, 02:00)


def test_character_span_f1_strict():
    gold = [
        TemporalEntity(text_span="พรุ่งนี้", start_char=0, end_char=8, type="POINT"),
        TemporalEntity(text_span="บ่ายสอง", start_char=9, end_char=16, type="POINT"),
    ]

    # Perfect prediction
    res_perfect = compute_span_f1(gold, gold, relaxed=False)
    assert res_perfect["precision"] == 1.0
    assert res_perfect["recall"] == 1.0
    assert res_perfect["f1"] == 1.0
    assert res_perfect["tp"] == 2.0

    # Discrepant boundary
    pred_shifted = [
        TemporalEntity(text_span="ตอนพรุ่งนี้", start_char=0, end_char=11, type="POINT"),
        TemporalEntity(text_span="บ่ายสอง", start_char=9, end_char=16, type="POINT"),
    ]
    res_strict = compute_span_f1(pred_shifted, gold, relaxed=False)
    assert res_strict["tp"] == 1.0
    assert res_strict["fp"] == 1.0
    assert res_strict["fn"] == 1.0
    assert res_strict["precision"] == 0.5
    assert res_strict["recall"] == 0.5
    assert res_strict["f1"] == 0.5

    # Relaxed overlap matches the overlapping span
    res_relaxed = compute_span_f1(pred_shifted, gold, relaxed=True)
    assert res_relaxed["tp"] == 2.0
    assert res_relaxed["f1"] == 1.0


def test_span_f1_empty_edge_cases():
    assert compute_span_f1([], []) == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
    }

    ent = [TemporalEntity(text_span="x", start_char=0, end_char=1, type="POINT")]
    assert compute_span_f1(ent, [])["precision"] == 0.0
    assert compute_span_f1([], ent)["recall"] == 0.0


def test_value_accuracy():
    gold = [
        TemporalEntity(text_span="วันนี้", start_char=0, end_char=6, type="POINT", iso_value="2026-11-08T12:00:00+07:00"),
        TemporalEntity(text_span="สายๆ", start_char=7, end_char=11, type="RANGE", range_start="2026-11-08T09:00:00+07:00", range_end="2026-11-08T11:30:00+07:00"),
    ]
    pred = [
        TemporalEntity(text_span="วันนี้", start_char=0, end_char=6, type="POINT", iso_value="2026-11-08T12:00:00+07:00"),
        TemporalEntity(text_span="สายๆ", start_char=7, end_char=11, type="RANGE", range_start="2026-11-08T09:00:00+07:00", range_end="2026-11-08T11:30:00+07:00"),
    ]

    assert compute_value_accuracy(pred, gold) == 1.0

    # Wrong point value
    pred_wrong = [
        TemporalEntity(text_span="วันนี้", start_char=0, end_char=6, type="POINT", iso_value="2026-11-09T12:00:00+07:00"),
        TemporalEntity(text_span="สายๆ", start_char=7, end_char=11, type="RANGE", range_start="2026-11-08T09:00:00+07:00", range_end="2026-11-08T11:30:00+07:00"),
    ]
    assert compute_value_accuracy(pred_wrong, gold) == 0.5


def test_continuous_range_iou_mixed_naive_aware():
    # Naive datetime mixed with tz-aware string must not crash with TypeError
    d1 = datetime(2026, 11, 8, 9, 0)
    d1_end = datetime(2026, 11, 8, 11, 0)
    d2 = "2026-11-08T09:00:00+07:00"
    d3 = "2026-11-08T11:00:00+07:00"

    iou = compute_range_iou(d1, d1_end, d2, d3)
    assert pytest.approx(iou) == 1.0

    # Disjoint point and range returns 0.0 without error
    iou_zero = compute_range_iou(d1, d1, d2, d3)
    assert iou_zero == 0.0


def test_continuous_range_iou_2400_string():
    # '24:00' representation normalized cleanly to next day
    iou = compute_range_iou("22:00", "24:00", "22:00", "24:00")
    assert pytest.approx(iou) == 1.0
