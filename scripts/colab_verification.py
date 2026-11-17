#!/usr/bin/env python3
"""Colab & Local 1-Click Verification Script for Thai-Temporal-Normalizer.

Runs comprehensive end-to-end verification:
1. Environment and import validation
2. 5 Core Linguistic Phenomena validation with assertions
3. Evaluation against tests/data/baseline_150.jsonl (150 stratified cases)
4. Latency SLA benchmarking (1,000 iterations, asserts p95 < 5.0ms)
5. Security & ReDoS guardrail verification
6. Outputs clean summary report and exits with 0 on complete pass.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure UTF-8 output encoding across platforms
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Resolve project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import library components
try:
    import thai_temporal_normalizer  # noqa: F401
    from thai_temporal_normalizer import TemporalEntity, parse_thai_temporal
    from thai_temporal_normalizer.metrics.range_iou import compute_range_iou  # noqa: F401
    from thai_temporal_normalizer.metrics.span_f1 import compute_span_f1  # noqa: F401
    from thai_temporal_normalizer.metrics.value_accuracy import compute_value_accuracy
except ImportError as err:
    print(f"[FATAL] Failed to import thai_temporal_normalizer: {err}")
    print("Please install package with: pip install dist/*.whl or pip install -e .")
    sys.exit(1)

BANGKOK_TZ = timezone(timedelta(hours=7))
DEFAULT_REF = datetime(2026, 11, 8, 12, 0, 0, tzinfo=BANGKOK_TZ)  # Sunday BE 2569


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def verify_linguistic_phenomena() -> bool:
    """Part 1: Verify all 5 linguistic phenomena with unit assertions."""
    print_header("Part 1: Linguistic Phenomena Assertions")
    all_passed = True

    tests = [
        # 1. Thai 6-Hour Clock
        (
            "6-Hour Clock (ตี / Early Morning)",
            "ตื่นตอนตีสาม",
            DEFAULT_REF,
            lambda ents: len(ents) == 1 and ents[0].iso_value == "2026-11-08T03:00:00+07:00",
            "Expected 2026-11-08T03:00:00+07:00",
        ),
        (
            "6-Hour Clock (บ่าย / Afternoon)",
            "เจอกันบ่ายสองโมงครึ่ง",
            DEFAULT_REF,
            lambda ents: len(ents) == 1 and ents[0].iso_value == "2026-11-08T14:30:00+07:00",
            "Expected 2026-11-08T14:30:00+07:00",
        ),
        (
            "6-Hour Clock (ทุ่ม / Night)",
            "นัดกินข้าวสองทุ่ม",
            DEFAULT_REF,
            lambda ents: len(ents) == 1 and ents[0].iso_value == "2026-11-08T20:00:00+07:00",
            "Expected 2026-11-08T20:00:00+07:00",
        ),
        (
            "6-Hour Clock (หกทุ่ม / Midnight)",
            "นอนตอนหกทุ่ม",
            DEFAULT_REF,
            lambda ents: len(ents) == 1 and ents[0].iso_value == "2026-11-08T00:00:00+07:00",
            "Expected 2026-11-08T00:00:00+07:00",
        ),
        # 2. Relative Day & Week
        (
            "Relative Day (พรุ่งนี้)",
            "ส่งเอกสารพรุ่งนี้",
            DEFAULT_REF,
            lambda ents: len(ents) == 1 and ents[0].iso_value == "2026-11-09T12:00:00+07:00",
            "Expected 2026-11-09T12:00:00+07:00",
        ),
        (
            "Relative Week (พุธหน้า)",
            "มีสอบพุธหน้า",
            DEFAULT_REF,
            lambda ents: len(ents) == 1 and ents[0].iso_value == "2026-11-11T12:00:00+07:00",
            "Expected 2026-11-11T12:00:00+07:00",
        ),
        (
            "Relative Past Day (เมื่อวานซืน)",
            "ไปหาหมอเมื่อวานซืน",
            DEFAULT_REF,
            lambda ents: len(ents) == 1 and ents[0].iso_value == "2026-11-06T12:00:00+07:00",
            "Expected 2026-11-06T12:00:00+07:00",
        ),
        # 3. Dynamic Short BE Century Pivot
        (
            "Short BE Pivot (ปี 69 -> BE 2569 / CE 2026)",
            "เริ่มงานปี 69",
            DEFAULT_REF,
            lambda ents: len(ents) == 1 and ents[0].iso_value == "2026-01-01T00:00:00+07:00",
            "Expected 2026-01-01T00:00:00+07:00",
        ),
        (
            "Short BE Pivot (ปี 79 -> BE 2579 / CE 2036)",
            "หมดอายุปี 79",
            DEFAULT_REF,
            lambda ents: len(ents) == 1 and ents[0].iso_value == "2036-01-01T00:00:00+07:00",
            "Expected 2036-01-01T00:00:00+07:00",
        ),
        (
            "Short BE Pivot (ปี 80 -> BE 2480 / CE 1937)",
            "สร้างเมื่อปี 80",
            DEFAULT_REF,
            lambda ents: len(ents) == 1 and ents[0].iso_value == "1937-01-01T00:00:00+07:00",
            "Expected 1937-01-01T00:00:00+07:00",
        ),
        (
            "Full Thai Date Short BE (15 มี.ค. 67)",
            "วันที่ 15 มี.ค. 67",
            DEFAULT_REF,
            lambda ents: len(ents) == 1 and ents[0].iso_value == "2024-03-15T00:00:00+07:00",
            "Expected 2024-03-15T00:00:00+07:00",
        ),
        # 4. Night-Time Day Rollover
        (
            "Night Rollover (คืนวันศุกร์ ตีสอง -> เสาร์ 02:00)",
            "คืนวันศุกร์ ตีสอง",
            DEFAULT_REF,
            lambda ents: len(ents) == 1 and ents[0].iso_value == "2026-11-14T02:00:00+07:00",
            "Expected 2026-11-14T02:00:00+07:00 (Saturday morning)",
        ),
        (
            "Night Rollover (คืนนี้ ตีหนึ่ง)",
            "คืนนี้ ตีหนึ่ง",
            DEFAULT_REF,
            lambda ents: len(ents) == 1 and ents[0].iso_value == "2026-11-09T01:00:00+07:00",
            "Expected 2026-11-09T01:00:00+07:00 (Next day morning)",
        ),
        # 5. Coarse Time Ranges
        (
            "Coarse Period (พรุ่งนี้ช่วงสายๆ)",
            "พรุ่งนี้ช่วงสายๆ",
            DEFAULT_REF,
            lambda ents: (
                len(ents) == 1
                and ents[0].type == "RANGE"
                and ents[0].range_start == "2026-11-09T09:00:00+07:00"
                and ents[0].range_end == "2026-11-09T11:30:00+07:00"
            ),
            "Expected RANGE 09:00 to 11:30 on 2026-11-09",
        ),
        (
            "Coarse Midnight Wrap (ดึกๆ 22:00-02:00 next day)",
            "พรุ่งนี้ดึกๆ",
            DEFAULT_REF,
            lambda ents: (
                len(ents) == 1
                and ents[0].type == "RANGE"
                and ents[0].range_start == "2026-11-09T22:00:00+07:00"
                and ents[0].range_end == "2026-11-10T02:00:00+07:00"
            ),
            "Expected RANGE 22:00 to 02:00 next day",
        ),
    ]

    for name, text, ref, validator, err_msg in tests:
        ents = parse_thai_temporal(text, reference_time=ref)
        passed = validator(ents)
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name:<46} | Input: '{text}'")
        if not passed:
            print(f"         Reason: {err_msg}")
            if ents:
                print(f"         Actual: {ents[0].iso_value or (ents[0].range_start + ' to ' + ents[0].range_end)}")
            else:
                print("         Actual: [] (no entity returned)")
            all_passed = False

    return all_passed


def evaluate_baseline_150() -> bool:
    """Part 2: Evaluate against tests/data/baseline_150.jsonl."""
    print_header("Part 2: Evaluation on Stratified Baseline (N=150)")
    data_path = PROJECT_ROOT / "tests" / "data" / "baseline_150.jsonl"
    if not data_path.exists():
        print(f"[FAIL] Baseline dataset not found at {data_path}")
        return False

    records: List[Dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} test cases across 5 categories.")

    correct = 0
    errors = 0
    category_stats: Dict[str, Dict[str, int]] = {}

    for row in records:
        cat = row.get("category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0, "errors": 0}
        category_stats[cat]["total"] += 1

        raw_input = row["raw_input"]
        ref_time_str = (
            row.get("reference_time")
            or row.get("metadata", {}).get("reference_time")
            or "2026-11-08T12:00:00+07:00"
        )
        ref_time = datetime.fromisoformat(ref_time_str)

        # Build gold entities from ground_truth or entities field
        gold_entities: List[TemporalEntity] = []
        if "ground_truth" in row and isinstance(row["ground_truth"], dict):
            gt = row["ground_truth"]
            gold_entities.append(
                TemporalEntity(
                    text_span=gt["text_span"],
                    start_char=gt["start_char"],
                    end_char=gt["end_char"],
                    type=gt["type"],
                    iso_value=gt.get("iso_value"),
                    range_start=gt.get("range_start"),
                    range_end=gt.get("range_end"),
                    precision=gt.get("precision", "exact"),
                    confidence=gt.get("confidence", 1.0),
                )
            )
        elif "entities" in row and isinstance(row["entities"], list):
            for e in row["entities"]:
                gold_entities.append(
                    TemporalEntity(
                        text_span=e["text_span"],
                        start_char=e["start_char"],
                        end_char=e["end_char"],
                        type=e["type"],
                        iso_value=e.get("iso_value"),
                        range_start=e.get("range_start"),
                        range_end=e.get("range_end"),
                        precision=e.get("precision", "exact"),
                        confidence=e.get("confidence", 1.0),
                    )
                )

        pred_entities = parse_thai_temporal(raw_input, reference_time=ref_time)

        # Compute accuracy via formal evaluation metric
        acc = compute_value_accuracy(pred_entities, gold_entities)
        if acc >= 1.0:
            correct += 1
            category_stats[cat]["correct"] += 1
        else:
            errors += 1
            category_stats[cat]["errors"] += 1

    total = len(records)
    acc_pct = (correct / total) * 100.0
    err_pct = (errors / total) * 100.0

    print(f"\nOverall Result: {correct}/{total} correct ({acc_pct:.2f}% Accuracy, {err_pct:.2f}% Error Rate)")
    print("\nPer-Category Breakdown:")
    for cat, stats in sorted(category_stats.items()):
        cat_acc = (stats["correct"] / stats["total"]) * 100.0
        print(f"  - {cat:<24}: {stats['correct']}/{stats['total']} ({cat_acc:.1f}%)")

    passed = (correct == total)
    print(f"\nBaseline Verification Status: {'[PASS]' if passed else '[FAIL]'}")
    return passed


def benchmark_latency_sla() -> bool:
    """Part 3: Measure Latency SLA (1,000 runs, assert p95 < 5.0ms)."""
    print_header("Part 3: Latency SLA Benchmark (Single-Thread CPU)")

    sample_expressions = [
        "เจอกันพรุ่งนี้ บ่ายสองโมงครึ่งนะ",
        "คืนวันศุกร์ ตีสอง",
        "โครงการปี 69 จะเริ่มช่วงสายๆ",
        "อีกสิบห้านาทีบ่ายสอง",
        "15 มี.ค. 67 ตอนสองทุ่ม",
        "มะรืนนี้ ช่วงหัวค่ำ",
    ]

    # Warmup runs (200 passes)
    for _ in range(200):
        for text in sample_expressions:
            parse_thai_temporal(text, reference_time=DEFAULT_REF)

    # 1,000 timed runs
    durations_ms: List[float] = []
    runs = 1000
    for i in range(runs):
        text = sample_expressions[i % len(sample_expressions)]
        t0 = time.perf_counter_ns()
        parse_thai_temporal(text, reference_time=DEFAULT_REF)
        t1 = time.perf_counter_ns()
        durations_ms.append((t1 - t0) / 1_000_000.0)

    durations_ms.sort()
    mean_ms = sum(durations_ms) / len(durations_ms)
    p50_ms = durations_ms[int(0.50 * len(durations_ms))]
    p90_ms = durations_ms[int(0.90 * len(durations_ms))]
    p95_ms = durations_ms[int(0.95 * len(durations_ms))]
    p99_ms = durations_ms[int(0.99 * len(durations_ms))]
    max_ms = durations_ms[-1]

    print(f"  Iterations: {runs}")
    print(f"  Mean:       {mean_ms:.4f} ms")
    print(f"  p50:        {p50_ms:.4f} ms")
    print(f"  p90:        {p90_ms:.4f} ms")
    print(f"  p95:        {p95_ms:.4f} ms (Target SLA: < 5.0 ms)")
    print(f"  p99:        {p99_ms:.4f} ms")
    print(f"  Max:        {max_ms:.4f} ms")

    passed = (p95_ms < 5.0)
    print(f"\nLatency SLA Status: {'[PASS]' if passed else '[FAIL]'} ({5.0 / p95_ms:.1f}x headroom)")
    return passed


def verify_security_guardrails() -> bool:
    """Part 4: Security & ReDoS guardrails."""
    print_header("Part 4: Security & ReDoS Guardrails")
    all_passed = True

    # 1. Hard input length bound (<= 120 chars)
    short_text = "พรุ่งนี้" * 15  # length = 120 chars
    long_text = "พรุ่งนี้" * 16   # length = 128 chars (> 120)

    res_short = parse_thai_temporal(short_text, reference_time=DEFAULT_REF)
    t0 = time.perf_counter_ns()
    res_long = parse_thai_temporal(long_text, reference_time=DEFAULT_REF)
    t1 = time.perf_counter_ns()
    rej_time_ms = (t1 - t0) / 1_000_000.0

    len_check_passed = (len(res_short) > 0 and len(res_long) == 0 and rej_time_ms < 0.5)
    print(f"  {'[PASS]' if len_check_passed else '[FAIL]'} Input > 120 chars rejected immediately ({rej_time_ms:.4f} ms)")
    if not len_check_passed:
        all_passed = False

    # 2. Strict mode raises ValueError on > 120 chars
    strict_passed = False
    try:
        parse_thai_temporal(long_text, reference_time=DEFAULT_REF, strict_mode=True)
    except ValueError:
        strict_passed = True
    print(f"  {'[PASS]' if strict_passed else '[FAIL]'} Strict mode raises ValueError on > 120 chars")
    if not strict_passed:
        all_passed = False

    # 3. Pathological backtracking payload safety
    pathological = "สายๆ" * 20  # Repetitive token
    t0 = time.perf_counter_ns()
    parse_thai_temporal(pathological, reference_time=DEFAULT_REF)
    t1 = time.perf_counter_ns()
    backtrack_ms = (t1 - t0) / 1_000_000.0

    bt_passed = (backtrack_ms < 2.0)
    print(f"  {'[PASS]' if bt_passed else '[FAIL]'} Catastrophic backtracking defense ({backtrack_ms:.4f} ms)")
    if not bt_passed:
        all_passed = False

    return all_passed


def main() -> int:
    print("\n" + "#" * 70)
    print("# Thai-Temporal-Normalizer: Colab & Local End-to-End Verification")
    print(f"# Python Runtime: {sys.version.split()[0]} on {sys.platform}")
    print("#" * 70)

    part1_ok = verify_linguistic_phenomena()
    part2_ok = evaluate_baseline_150()
    part3_ok = benchmark_latency_sla()
    part4_ok = verify_security_guardrails()

    print_header("FINAL VERIFICATION SUMMARY")
    print(f"  1. Linguistic Phenomena:      {'PASS' if part1_ok else 'FAIL'}")
    print(f"  2. Stratified Baseline (N=150):{'PASS' if part2_ok else 'FAIL'}")
    print(f"  3. Latency SLA Benchmark:     {'PASS' if part3_ok else 'FAIL'}")
    print(f"  4. Security & ReDoS Guard:    {'PASS' if part4_ok else 'FAIL'}")

    all_ok = part1_ok and part2_ok and part3_ok and part4_ok
    if all_ok:
        print("\n>>> ALL VERIFICATION CHECKS PASSED SUCCESSFULLY (Exit Code 0) <<<\n")
        return 0
    else:
        print("\n>>> ONE OR MORE VERIFICATION CHECKS FAILED (Exit Code 1) <<<\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
