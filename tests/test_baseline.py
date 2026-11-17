"""Unit tests for Phase 0 Baseline Dataset and Benchmark Harness."""

import json
from datetime import datetime
from pathlib import Path
from collections import Counter
import pytest

from benchmarks.run_baseline import compute_wilson_ci, compute_iou_continuous

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "tests" / "data" / "baseline_150.jsonl"
RESULTS_PATH = PROJECT_ROOT / "benchmarks" / "baseline_results.json"
PRED_RAW_PATH = PROJECT_ROOT / "predictions_raw.jsonl"


def test_baseline_dataset_count_and_distribution():
    assert DATA_PATH.exists(), f"Missing dataset file at {DATA_PATH}"
    items = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    assert len(items) == 150, f"Expected 150 items, got {len(items)}"

    counts = Counter(x["category"] for x in items)
    expected_categories = {
        "relative_day_week": 30,
        "thai_6hour_clock": 30,
        "coarse_time_periods": 30,
        "short_buddhist_era": 30,
        "night_date_rollover": 30,
    }
    assert dict(counts) == expected_categories


def test_baseline_dataset_schema_and_span_integrity():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            item = json.loads(line)
            assert item["id"] == idx
            assert "raw_input" in item and len(item["raw_input"]) <= 120
            assert "ground_truth" in item
            gt = item["ground_truth"]

            # Span check
            span = gt["text_span"]
            raw = item["raw_input"]
            start_c = gt["start_char"]
            end_c = gt["end_char"]
            assert raw[start_c:end_c] == span, f"Span slice mismatch at id={item['id']}"

            # Datetime check
            ref_dt = datetime.fromisoformat(item["metadata"]["reference_time"])
            assert ref_dt is not None

            if gt["type"] == "POINT":
                assert gt["iso_value"] is not None
                dt = datetime.fromisoformat(gt["iso_value"])
                assert dt is not None
            elif gt["type"] == "RANGE":
                assert gt["range_start"] is not None
                assert gt["range_end"] is not None
                s_dt = datetime.fromisoformat(gt["range_start"])
                e_dt = datetime.fromisoformat(gt["range_end"])
                assert s_dt is not None and e_dt is not None


def test_wilson_ci_boundary_values():
    # Boundary Go: k=45, n=150
    lower_go, upper_go = compute_wilson_ci(45, 150)
    assert lower_go == 0.2324
    assert upper_go == 0.3776

    # Boundary No-Go: k=15, n=150
    lower_nogo, upper_nogo = compute_wilson_ci(15, 150)
    assert lower_nogo == 0.0615
    assert upper_nogo == 0.1584


def test_continuous_iou_calculation():
    t0 = datetime(2026, 11, 8, 9, 0, 0)
    t1 = datetime(2026, 11, 8, 11, 30, 0)
    # Exact match: IoU == 1.0
    assert compute_iou_continuous(t0, t1, t0, t1) == 1.0

    # Half overlap: 09:00-11:00 vs 10:00-12:00
    # intersection: 1h, union: 3h -> 1/3
    s1, e1 = datetime(2026, 11, 8, 9, 0, 0), datetime(2026, 11, 8, 11, 0, 0)
    s2, e2 = datetime(2026, 11, 8, 10, 0, 0), datetime(2026, 11, 8, 12, 0, 0)
    iou = compute_iou_continuous(s1, e1, s2, e2)
    assert pytest.approx(iou, rel=1e-3) == 1.0 / 3.0


def test_benchmark_results_and_gate_decision():
    assert RESULTS_PATH.exists(), f"Missing results file at {RESULTS_PATH}"
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["final_gate_verdict"] == "GO"
    systems = data["systems"]
    assert "dateparser" in systems
    assert "pythainlp_wrapper" in systems
    assert "simple_regex" in systems

    for sname, sdata in systems.items():
        assert sdata["total"] == 150
        assert sdata["error_rate"] > 0.30
        assert sdata["gate_decision"] == "GO"


def test_predictions_raw_logged():
    if not PRED_RAW_PATH.exists():
        pytest.skip('predictions_raw.jsonl cleaned for release')
    assert PRED_RAW_PATH.exists(), f"Missing raw predictions log at {PRED_RAW_PATH}"
    with open(PRED_RAW_PATH, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) >= 450, f"Expected at least 450 raw prediction logs, got {len(lines)}"
    baseline_records = [line_str for line_str in lines if '"system":' in line_str]
    assert len(baseline_records) == 450, f"Expected exactly 450 baseline records, got {len(baseline_records)}"
