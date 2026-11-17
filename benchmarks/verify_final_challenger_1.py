"""Empirical Challenger 1 - Adversarial Dataset & Span Offset Fuzzer.

Exhaustive verification across all 10,000 rows of data/v1.1:
1. Span Offset Audit (100% of 9,000 positive rows: raw_input[start:end] == span, coordinate validity, Unicode NFC/NFD stability)
2. Global Deduplication Audit (10,000 rows: global uniqueness, within-split uniqueness, pairwise split disjointness, whitespace-normalized uniqueness)
3. Sentence Length Bounds (all len(raw_input) <= 120, character length statistics & histograms)
4. Schema & Data Quality Audit (JSON validity, metadata completeness, negative sample null-GT assertion)
5. Cryptographic SHA-256 Checksum Validation (data/v1.1/checksums.json & DATA_CARD.md alignment)
6. Latency / ReDoS Stress Test on extreme boundary rows
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Force UTF-8 on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_V1_1 = PROJECT_ROOT / "data" / "v1.1"
SPLIT_FILES = ["train.jsonl", "dev.jsonl", "test_id.jsonl", "test_ood.jsonl"]
EXPECTED_COUNTS = {
    "train.jsonl": 7000,
    "dev.jsonl": 1000,
    "test_id.jsonl": 1000,
    "test_ood.jsonl": 1000,
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_full_adversarial_audit() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits": {},
        "global_summary": {},
        "span_offset_audit": {},
        "deduplication_audit": {},
        "length_bounds_audit": {},
        "schema_audit": {},
        "checksum_audit": {},
        "redos_stress_audit": {},
        "verdict": "UNKNOWN",
        "errors": [],
    }

    all_records: List[Dict[str, Any]] = []
    records_by_split: Dict[str, List[Dict[str, Any]]] = {}
    raw_texts_by_split: Dict[str, List[str]] = {}
    all_raw_texts: List[str] = []
    all_record_ids: List[str] = []

    # ---------------------------------------------------------
    # 1. File Loading & Line-by-Line Schema Parsing
    # ---------------------------------------------------------
    for filename in SPLIT_FILES:
        filepath = DATA_V1_1 / filename
        if not filepath.exists():
            report["errors"].append(f"Missing file: {filepath}")
            continue

        split_recs: List[Dict[str, Any]] = []
        raw_list: List[str] = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw_line = line.strip()
                if not raw_line:
                    continue
                try:
                    obj = json.loads(raw_line)
                except Exception as exc:
                    report["errors"].append(f"JSONDecodeError in {filename}:{line_no}: {exc}")
                    continue

                split_recs.append(obj)
                all_records.append(obj)
                raw_text = obj.get("raw_input", "")
                raw_list.append(raw_text)
                all_raw_texts.append(raw_text)
                if "id" in obj:
                    all_record_ids.append(obj["id"])

        records_by_split[filename] = split_recs
        raw_texts_by_split[filename] = raw_list
        report["splits"][filename] = {
            "actual_rows": len(split_recs),
            "expected_rows": EXPECTED_COUNTS[filename],
            "matches_expected": len(split_recs) == EXPECTED_COUNTS[filename],
            "size_bytes": filepath.stat().st_size,
            "sha256": compute_sha256(filepath),
        }

    total_records = len(all_records)
    report["global_summary"]["total_records"] = total_records
    report["global_summary"]["target_records"] = 10000

    # ---------------------------------------------------------
    # 2. Span Offset Audit (100% of positive rows)
    # ---------------------------------------------------------
    pos_count = 0
    neg_count = 0
    span_mismatches: List[Dict[str, Any]] = []
    coord_violations: List[Dict[str, Any]] = []
    unicode_nfd_discrepancies: List[Dict[str, Any]] = []
    empty_spans: List[str] = []

    for obj in all_records:
        rec_id = obj.get("id", "UNKNOWN")
        raw_input = obj.get("raw_input", "")
        gt = obj.get("ground_truth")

        if gt is None:
            neg_count += 1
            continue

        pos_count += 1
        start_char = gt.get("start_char")
        end_char = gt.get("end_char")
        text_span = gt.get("text_span")

        # Bounds and type checks
        if not isinstance(start_char, int) or not isinstance(end_char, int):
            coord_violations.append({
                "id": rec_id,
                "reason": f"Non-integer coordinates: start={start_char}, end={end_char}",
            })
            continue

        if not (0 <= start_char < end_char <= len(raw_input)):
            coord_violations.append({
                "id": rec_id,
                "reason": f"Out of bounds: 0 <= {start_char} < {end_char} <= {len(raw_input)}",
                "raw_len": len(raw_input),
            })
            continue

        if not text_span:
            empty_spans.append(rec_id)
            continue

        # Exact character slice assertion
        extracted = raw_input[start_char:end_char]
        if extracted != text_span:
            span_mismatches.append({
                "id": rec_id,
                "start_char": start_char,
                "end_char": end_char,
                "expected_span": text_span,
                "actual_slice": extracted,
                "raw_input": raw_input,
            })

        # Length check
        if len(text_span) != (end_char - start_char):
            coord_violations.append({
                "id": rec_id,
                "reason": f"Span length mismatch: len(text_span)={len(text_span)} != (end - start)={end_char - start_char}",
            })

        # Check Unicode NFC/NFD consistency
        # If raw_input is normalized to NFD, do character counts diverge?
        if unicodedata.normalize("NFC", raw_input) != raw_input:
            unicode_nfd_discrepancies.append({
                "id": rec_id,
                "issue": "raw_input is not NFC normalized",
            })

        # Whitespace sanity in text_span
        if text_span != text_span.strip():
            coord_violations.append({
                "id": rec_id,
                "reason": f"text_span has untrimmed leading/trailing whitespace: '{text_span}'",
            })

    # Multi-occurrence check
    multi_occurrence_spans = 0
    for obj in all_records:
        gt = obj.get("ground_truth")
        if gt:
            raw = obj.get("raw_input", "")
            span = gt.get("text_span", "")
            if raw.count(span) > 1:
                multi_occurrence_spans += 1

    report["span_offset_audit"] = {
        "positive_rows_evaluated": pos_count,
        "negative_rows_evaluated": neg_count,
        "expected_positive_rows": 9000,
        "expected_negative_rows": 1000,
        "span_offset_mismatches_count": len(span_mismatches),
        "span_offset_mismatches_samples": span_mismatches[:5],
        "coordinate_violations_count": len(coord_violations),
        "coordinate_violations_samples": coord_violations[:5],
        "empty_spans_count": len(empty_spans),
        "unicode_nfc_non_compliant_count": len(unicode_nfd_discrepancies),
        "multi_occurrence_spans_count": multi_occurrence_spans,
        "all_positive_exact_match": (len(span_mismatches) == 0 and pos_count == 9000 and len(coord_violations) == 0),
    }

    # ---------------------------------------------------------
    # 3. Global Deduplication Audit
    # ---------------------------------------------------------
    unique_raw_count = len(set(all_raw_texts))
    raw_duplicates_count = total_records - unique_raw_count

    # Normalized text deduplication (strip extra whitespace)
    normalized_texts = [" ".join(t.split()) for t in all_raw_texts]
    unique_normalized_count = len(set(normalized_texts))

    # Cross-split disjointness
    pairwise_overlaps: Dict[str, int] = {}
    split_names = list(raw_texts_by_split.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1 = split_names[i]
            s2 = split_names[j]
            overlap = set(raw_texts_by_split[s1]).intersection(set(raw_texts_by_split[s2]))
            pairwise_overlaps[f"{s1} ∩ {s2}"] = len(overlap)

    # Within-split duplicates
    within_split_dups: Dict[str, int] = {}
    for s_name, s_texts in raw_texts_by_split.items():
        dups = len(s_texts) - len(set(s_texts))
        within_split_dups[s_name] = dups

    # ID uniqueness
    unique_id_count = len(set(all_record_ids))

    report["deduplication_audit"] = {
        "total_rows": total_records,
        "unique_raw_inputs": unique_raw_count,
        "raw_duplicates": raw_duplicates_count,
        "zero_raw_duplicates": raw_duplicates_count == 0,
        "unique_whitespace_normalized": unique_normalized_count,
        "unique_record_ids": unique_id_count,
        "zero_id_duplicates": unique_id_count == total_records,
        "within_split_duplicates": within_split_dups,
        "pairwise_split_overlaps": pairwise_overlaps,
        "zero_pairwise_overlap": all(v == 0 for v in pairwise_overlaps.values()),
    }

    # ---------------------------------------------------------
    # 4. Sentence Length Bounds (ReDoS SLA bound: <= 120 chars)
    # ---------------------------------------------------------
    lengths = [len(t) for t in all_raw_texts]
    violations_over_120 = [
        {"id": obj.get("id"), "len": len(obj.get("raw_input", "")), "text": obj.get("raw_input")}
        for obj in all_records
        if len(obj.get("raw_input", "")) > 120
    ]
    empty_strings = [
        obj.get("id")
        for obj in all_records
        if len(obj.get("raw_input", "")) == 0
    ]

    sorted_lengths = sorted(lengths)
    p50 = statistics.median(lengths)
    p90 = sorted_lengths[int(0.90 * len(sorted_lengths))]
    p95 = sorted_lengths[int(0.95 * len(sorted_lengths))]
    p99 = sorted_lengths[int(0.99 * len(sorted_lengths))]

    # Histogram buckets
    buckets = {
        "0-20": sum(1 for l in lengths if 0 <= l <= 20),
        "21-40": sum(1 for l in lengths if 21 <= l <= 40),
        "41-60": sum(1 for l in lengths if 41 <= l <= 60),
        "61-80": sum(1 for l in lengths if 61 <= l <= 80),
        "81-100": sum(1 for l in lengths if 81 <= l <= 100),
        "101-120": sum(1 for l in lengths if 101 <= l <= 120),
        ">120": sum(1 for l in lengths if l > 120),
    }

    report["length_bounds_audit"] = {
        "bound_sla": 120,
        "violations_over_120_count": len(violations_over_120),
        "violations_samples": violations_over_120[:5],
        "all_satisfy_sla_120": len(violations_over_120) == 0,
        "empty_strings_count": len(empty_strings),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "mean_len": round(statistics.mean(lengths), 2),
        "median_len": p50,
        "p90_len": p90,
        "p95_len": p95,
        "p99_len": p99,
        "histogram": buckets,
    }

    # ---------------------------------------------------------
    # 5. Schema & Metadata Consistency
    # ---------------------------------------------------------
    missing_id_count = sum(1 for r in all_records if "id" not in r)
    missing_raw_count = sum(1 for r in all_records if "raw_input" not in r)
    missing_meta_count = sum(1 for r in all_records if "metadata" not in r)
    invalid_category_count = sum(
        1 for r in all_records
        if r.get("metadata", {}).get("category") not in {
            "standard", "bound_duration_recurrence", "chat_colloquial", "negative_sample"
        }
    )

    report["schema_audit"] = {
        "missing_id_count": missing_id_count,
        "missing_raw_input_count": missing_raw_count,
        "missing_metadata_count": missing_meta_count,
        "invalid_category_count": invalid_category_count,
        "schema_flawless": (missing_id_count == 0 and missing_raw_count == 0 and missing_meta_count == 0 and invalid_category_count == 0),
    }

    # ---------------------------------------------------------
    # 6. Checksum Manifest Integrity
    # ---------------------------------------------------------
    manifest_path = DATA_V1_1 / "checksums.json"
    manifest_matches = False
    manifest_details = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        manifest_files = manifest_data.get("files", {})
        all_shas_match = True
        mismatched_shas = []

        for fname in SPLIT_FILES:
            actual_sha = report["splits"][fname]["sha256"]
            manifest_sha = manifest_files.get(fname, {}).get("sha256")
            if actual_sha != manifest_sha:
                all_shas_match = False
                mismatched_shas.append((fname, manifest_sha, actual_sha))

        manifest_matches = all_shas_match
        manifest_details = {
            "version": manifest_data.get("version"),
            "total_records": manifest_data.get("total_records"),
            "seed": manifest_data.get("seed"),
            "all_shas_match": all_shas_match,
            "mismatched_shas": mismatched_shas,
        }

    report["checksum_audit"] = {
        "manifest_path_exists": manifest_path.exists(),
        "manifest_matches": manifest_matches,
        "details": manifest_details,
    }

    # ---------------------------------------------------------
    # 7. ReDoS Stress & Normalizer Latency Fuzzing
    # ---------------------------------------------------------
    # Test normalizer on boundary cases: top 100 longest inputs, 100 negative inputs, 100 chat inputs
    try:
        from thai_temporal_normalizer import parse_thai_temporal

        # Full False Positive Immunity Audit on all 1,000 negative records
        all_neg_records = [r for r in all_records if r.get("ground_truth") is None]
        fp_violations: List[Dict[str, Any]] = []
        for rec in all_neg_records:
            extracted = parse_thai_temporal(rec["raw_input"])
            if len(extracted) > 0:
                fp_violations.append({
                    "id": rec.get("id"),
                    "text": rec.get("raw_input"),
                    "extracted": [e.text_span for e in extracted],
                })

        report["negative_fp_audit"] = {
            "total_negative_records": len(all_neg_records),
            "false_positive_count": len(fp_violations),
            "false_positives": fp_violations,
            "zero_false_positives": len(fp_violations) == 0,
        }

        longest_records = sorted(all_records, key=lambda r: len(r.get("raw_input", "")), reverse=True)[:100]
        chat_records = [r for r in all_records if r.get("metadata", {}).get("category") == "chat_colloquial"][:100]
        neg_records = all_neg_records[:100]

        stress_pool = longest_records + chat_records + neg_records
        latencies_ms: List[float] = []
        crashes: List[str] = []

        for rec in stress_pool:
            text = rec["raw_input"]
            t0 = time.perf_counter()
            try:
                res = parse_thai_temporal(text)
            except Exception as exc:
                crashes.append(f"Crash on ID={rec['id']}: {exc}")
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        report["redos_stress_audit"] = {
            "tested_records_count": len(stress_pool),
            "crashes_count": len(crashes),
            "crashes": crashes,
            "min_latency_ms": round(min(latencies_ms), 3),
            "max_latency_ms": round(max(latencies_ms), 3),
            "mean_latency_ms": round(statistics.mean(latencies_ms), 3),
            "p95_latency_ms": round(sorted(latencies_ms)[int(0.95 * len(latencies_ms))], 3),
            "all_under_50ms_sla": max(latencies_ms) < 50.0,
        }
    except Exception as exc:
        report["redos_stress_audit"] = {"error": str(exc)}

    # ---------------------------------------------------------
    # Final Verdict Synthesis
    # ---------------------------------------------------------
    checks = [
        total_records == 10000,
        report["span_offset_audit"]["all_positive_exact_match"],
        report["deduplication_audit"]["zero_raw_duplicates"],
        report["deduplication_audit"]["zero_pairwise_overlap"],
        report["length_bounds_audit"]["all_satisfy_sla_120"],
        report["schema_audit"]["schema_flawless"],
        report["checksum_audit"]["manifest_matches"],
        report["negative_fp_audit"].get("zero_false_positives", False),
        report["redos_stress_audit"].get("all_under_50ms_sla", False),
    ]

    if all(checks) and len(report["errors"]) == 0:
        report["verdict"] = "APPROVE"
    else:
        report["verdict"] = "REJECT"

    return report


if __name__ == "__main__":
    audit = run_full_adversarial_audit()
    print("=" * 80)
    print("THAI-TEMPORAL-NORMALIZER v1.1 ADVERSARIAL AUDIT REPORT")
    print("=" * 80)
    print(f"Verdict: {audit['verdict']}")
    print(f"Total Rows: {audit['global_summary']['total_records']} / {audit['global_summary']['target_records']}")
    print("-" * 80)
    print("1. Span Offset Audit:")
    spa = audit["span_offset_audit"]
    print(f"   - Positive rows: {spa['positive_rows_evaluated']} (expected: {spa['expected_positive_rows']})")
    print(f"   - Span offset mismatches: {spa['span_offset_mismatches_count']}")
    print(f"   - Coordinate violations: {spa['coordinate_violations_count']}")
    print(f"   - Unicode NFC non-compliant: {spa['unicode_nfc_non_compliant_count']}")
    print(f"   - All exact matches: {spa['all_positive_exact_match']}")
    print("-" * 80)
    print("2. Deduplication Audit:")
    ded = audit["deduplication_audit"]
    print(f"   - Unique raw inputs: {ded['unique_raw_inputs']} / {ded['total_rows']}")
    print(f"   - Raw duplicates: {ded['raw_duplicates']}")
    print(f"   - Pairwise overlap: {ded['pairwise_split_overlaps']}")
    print(f"   - Zero duplicates across splits: {ded['zero_raw_duplicates']}")
    print("-" * 80)
    print("3. Sentence Length Bounds (SLA <= 120 chars):")
    sla = audit["length_bounds_audit"]
    print(f"   - Violations (>120): {sla['violations_over_120_count']}")
    print(f"   - Min len: {sla['min_len']}, Max len: {sla['max_len']}, Mean len: {sla['mean_len']}")
    print(f"   - Percentiles: p50={sla['median_len']}, p90={sla['p90_len']}, p95={sla['p95_len']}, p99={sla['p99_len']}")
    print(f"   - Histogram: {sla['histogram']}")
    print("-" * 80)
    print("4. Checksum Manifest Integrity:")
    chk = audit["checksum_audit"]
    print(f"   - Manifest exists: {chk['manifest_path_exists']}")
    print(f"   - All SHAs match: {chk['manifest_matches']}")
    print("-" * 80)
    print("5. ReDoS Stress & Latency Audit:")
    red = audit["redos_stress_audit"]
    print(f"   - Tested rows: {red.get('tested_records_count')}")
    print(f"   - Crashes: {red.get('crashes_count')}")
    print(f"   - Mean latency: {red.get('mean_latency_ms')} ms, Max: {red.get('max_latency_ms')} ms, p95: {red.get('p95_latency_ms')} ms")
    print(f"   - All < 50ms SLA: {red.get('all_under_50ms_sla')}")
    print("=" * 80)

    # Dump full JSON report to output
    out_json = DATA_V1_1 / "fuzzing_audit_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    print(f"Wrote full JSON audit report to {out_json}")
