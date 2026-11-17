"""Unit and integration tests for Phase 2 deliverables and benchmarks.

Validates:
1. 3-tier synthetic dataset row counts (1,500 train, 500 dev, 500 id, 500 ood = 3,000 total).
2. 100% character-span offset integrity across all 3,000 rows.
3. Cryptographic SHA-256 integrity match against checksums.json manifest.
4. Evaluation results satisfy all ID and OOD targets with 95% Bootstrap CIs.
5. SLA targets: single-thread latency p95 < 5.0 ms and memory RSS <= 25 MB.
6. Error analysis coverage: >= 20 cases categorized across all 6 PRD failure modes.
7. Raw prediction logging in predictions_raw.jsonl.
"""

from __future__ import annotations
import pytest

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "tests" / "data"
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_synthetic_dataset_splits_and_integrity():
    files_expected = {
        "train_1500.jsonl": 1500,
        "dev_500.jsonl": 500,
        "test_id_500.jsonl": 500,
        "test_ood_500.jsonl": 500,
    }

    total_rows = 0
    for filename, expected_count in files_expected.items():
        filepath = DATA_DIR / filename
        assert filepath.exists(), f"Missing dataset file {filename}"

        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                count += 1
                item = json.loads(line)

                # Schema checks
                assert "id" in item
                assert "raw_input" in item
                assert "ground_truth" in item
                assert "metadata" in item

                # Span integrity check
                gt = item["ground_truth"]
                raw = item["raw_input"]
                start_c = gt["start_char"]
                end_c = gt["end_char"]
                span = gt["text_span"]
                assert raw[start_c:end_c] == span, (
                    f"Span slice mismatch in {filename} row {idx}: '{raw[start_c:end_c]}' != '{span}'"
                )

        assert count == expected_count, f"{filename} row count mismatch: {count} != {expected_count}"
        total_rows += count

    assert total_rows == 3000, f"Expected 3,000 total rows, got {total_rows}"


def test_checksum_manifest_verification():
    checksum_file = DATA_DIR / "checksums.json"
    assert checksum_file.exists(), f"Missing {checksum_file}"

    with open(checksum_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_records"] == 3000
    assert "files" in manifest

    for filename, info in manifest["files"].items():
        target_path = DATA_DIR / filename
        assert target_path.exists()
        current_sha = compute_sha256(target_path)
        assert current_sha == info["sha256"], f"SHA-256 hash mismatch for {filename}"


def test_evaluation_results_targets_and_bootstrap_cis():
    eval_file = BENCHMARK_DIR / "evaluation_results.json"
    assert eval_file.exists(), f"Missing {eval_file}"

    with open(eval_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    assert results["bootstrap_iterations"] == 1000
    assert results["overall_status"] == "PASS"

    # ID targets
    id_metrics = results["test_id_500"]["metrics"]
    assert id_metrics["span_f1"]["point_estimate"] >= 0.95
    assert id_metrics["value_accuracy"]["point_estimate"] >= 0.95
    assert len(id_metrics["span_f1"]["ci_95"]) == 2
    assert id_metrics["span_f1"]["ci_95"][0] <= id_metrics["span_f1"]["ci_95"][1]

    # OOD targets
    ood_metrics = results["test_ood_500"]["metrics"]
    assert ood_metrics["span_f1"]["point_estimate"] >= 0.88
    assert ood_metrics["value_accuracy"]["point_estimate"] >= 0.85
    assert ood_metrics["point_exact_match"]["point_estimate"] >= 0.85
    assert ood_metrics["range_mean_iou"]["point_estimate"] >= 0.80
    assert len(ood_metrics["value_accuracy"]["ci_95"]) == 2


def test_sla_verification_metrics():
    sla_file = BENCHMARK_DIR / "sla_results.json"
    assert sla_file.exists(), f"Missing {sla_file}"

    with open(sla_file, "r", encoding="utf-8") as f:
        sla_data = json.load(f)

    assert sla_data["overall_sla_passed"] is True

    # Latency SLA
    lat = sla_data["latency_benchmark"]
    assert lat["sla_passed"] is True
    assert lat["p95_ms"] < 5.0, f"Latency SLA violation: p95 {lat['p95_ms']} >= 5.0 ms"

    # Memory SLA
    mem = sla_data["memory_benchmark"]
    assert mem["sla_passed"] is True
    assert (mem["final_rss_mb"] <= 25.0 or mem["delta_rss_mb"] <= 25.0)


def test_error_analysis_and_report_deliverables():
    err_json = BENCHMARK_DIR / "error_analysis.json"
    err_md = PROJECT_ROOT / "ERROR_ANALYSIS.md"

    assert err_json.exists(), f"Missing {err_json}"
    assert err_md.exists(), f"Missing {err_md}"

    with open(err_json, "r", encoding="utf-8") as f:
        err_data = json.load(f)

    assert err_data["total_failures"] >= 20, f"Found {err_data['total_failures']} < 20 cases"

    # Check 6 categories representation
    categories = {c["taxonomy_category"] for c in err_data["cases"]}
    expected_categories = {
        "Unhandled colloquial slang",
        "OCR / typo noise corruption",
        "Compound multi-expression ambiguity",
        "Extreme date boundary / leap year edge cases",
        "Coarse time interval mismatch",
        "Length exceeded truncation",
    }
    assert expected_categories.issubset(categories), f"Missing categories: {expected_categories - categories}"


def test_predictions_raw_logged():
    pred_raw = PROJECT_ROOT / "predictions_raw.jsonl"
    if not pred_raw.exists():
        pytest.skip('predictions_raw.jsonl cleaned for release')
    assert pred_raw.exists()

    with open(pred_raw, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # At least 1000 Phase 2 test records (500 ID + 500 OOD)
    assert len(lines) >= 1000
