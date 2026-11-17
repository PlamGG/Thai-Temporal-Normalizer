"""Systematic Error Analysis for Thai-Temporal-Normalizer (Phase 2).

Extracts and categorizes failure cases from the Out-of-Distribution (OOD) blind test
across the 6-category taxonomy (PRD Section 10 & 11):
1. Unhandled colloquial slang
2. OCR / typo noise corruption
3. Compound multi-expression ambiguity
4. Extreme date boundary / leap year edge cases
5. Coarse time interval mismatch
6. Length exceeded truncation

Outputs:
- benchmarks/error_analysis.json
- ERROR_ANALYSIS.md (comprehensive markdown report in project root)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from thai_temporal_normalizer import TemporalEntity, parse_thai_temporal  # noqa: E402
from thai_temporal_normalizer.metrics.span_f1 import compute_span_f1  # noqa: E402
from thai_temporal_normalizer.metrics.value_accuracy import compute_value_accuracy  # noqa: E402


TAXONOMY_MAP = {
    "slang_abbr_period": "Unhandled colloquial slang",
    "slang_particle_kwa": "Unhandled colloquial slang",
    "slang_particle_nid": "Unhandled colloquial slang",
    "slang_coarse_idiom": "Unhandled colloquial slang",
    "slang_exact_noon": "Unhandled colloquial slang",
    "clock_night_slang": "Unhandled colloquial slang",
    "typo_double_tone": "OCR / typo noise corruption",
    "typo_inverted_vowel": "OCR / typo noise corruption",
    "typo_shifted_key": "OCR / typo noise corruption",
    "typo_mixed_digits": "OCR / typo noise corruption",
    "typo_elongated_vowel": "OCR / typo noise corruption",
    "multi_ambiguity_choice": "Compound multi-expression ambiguity",
    "range_compound_explicit": "Compound multi-expression ambiguity",
    "coarse_compound_daylong": "Compound multi-expression ambiguity",
    "calendar_invalid_leap": "Extreme date boundary / leap year edge cases",
    "calendar_invalid_30day_month": "Extreme date boundary / leap year edge cases",
    "calendar_invalid_feb30": "Extreme date boundary / leap year edge cases",
    "century_pivot_overflow": "Extreme date boundary / leap year edge cases",
    "century_pivot_distant": "Extreme date boundary / leap year edge cases",
    "coarse_custom_convention": "Coarse time interval mismatch",
    "coarse_shifted_morning": "Coarse time interval mismatch",
    "coarse_truncated_midnight": "Coarse time interval mismatch",
    "coarse_shifted_evening": "Coarse time interval mismatch",
    "length_exceeded_rejection": "Length exceeded truncation",
}


def classify_error(item: Dict[str, Any], preds: List[TemporalEntity]) -> str:
    """Classify failure case into one of the 6 taxonomy categories."""
    ling = item.get("metadata", {}).get("linguistic_type", "")
    if ling in TAXONOMY_MAP:
        return TAXONOMY_MAP[ling]

    raw = item["raw_input"]
    if len(raw) > 120:
        return "Length exceeded truncation"

    gt_span = item["ground_truth"]["text_span"]
    if any(s in gt_span for s in ["หรือ", "ถึง", "จรด"]):
        return "Compound multi-expression ambiguity"

    if any(s in gt_span for s in ["ก.พ.", "เม.ย."]) and not preds:
        return "Extreme date boundary / leap year edge cases"

    if any(s in gt_span for s in ["ช่วง", "ตอน", "สายๆ", "หัวค่ำ", "ดึกๆ"]):
        return "Coarse time interval mismatch"

    if any(c in raw for c in [":", "...", "->", "!", "[", "]"]):
        return "OCR / typo noise corruption"

    return "Unhandled colloquial slang"


def run_error_analysis(ood_file: Path) -> List[Dict[str, Any]]:
    """Identify and analyze all failure cases in OOD test split."""
    with open(ood_file, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    analyzed_failures: List[Dict[str, Any]] = []

    for item in items:
        raw_input = item["raw_input"]
        gt_dict = item["ground_truth"]
        ref_time = datetime.fromisoformat(item["metadata"]["reference_time"])

        preds = parse_thai_temporal(raw_input, reference_time=ref_time)

        gt_ent = TemporalEntity(
            text_span=gt_dict["text_span"],
            start_char=gt_dict["start_char"],
            end_char=gt_dict["end_char"],
            type=gt_dict["type"],
            iso_value=gt_dict.get("iso_value"),
            range_start=gt_dict.get("range_start"),
            range_end=gt_dict.get("range_end"),
            precision=gt_dict.get("precision", "exact"),
            confidence=float(gt_dict.get("confidence", 1.0)),
            grain=gt_dict.get("grain", "auto"),
        )

        span_res = compute_span_f1(preds, [gt_ent], relaxed=False)
        val_acc = compute_value_accuracy(preds, [gt_ent], range_iou_threshold=0.80)

        # Check if failure
        if span_res["f1"] < 1.0 or val_acc < 1.0:
            category = classify_error(item, preds)
            analyzed_failures.append({
                "id": item["id"],
                "raw_input": raw_input,
                "text_length": len(raw_input),
                "taxonomy_category": category,
                "ground_truth": gt_dict,
                "model_predictions": [p.to_dict() for p in preds],
                "span_f1": span_res["f1"],
                "value_accuracy": val_acc,
                "notes": item.get("notes", ""),
            })

    return analyzed_failures


def generate_markdown_report(failures: List[Dict[str, Any]], output_md: Path) -> None:
    """Generate detailed ERROR_ANALYSIS.md."""
    cat_counts: Dict[str, int] = {}
    for f in failures:
        cat = f["taxonomy_category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    lines = [
        "# Error Analysis Report: Thai-Temporal-Normalizer",
        "",
        "> **Phase 2 Milestone Deliverable**: Diagnostic Error Analysis across 6-Category Taxonomy",
        f"> **Total Failure Cases Analyzed**: {len(failures)} cases (Strict Target: >= 20 cases)",
        "> **Evaluation Source**: Out-of-Distribution (OOD) Blind Test Set (`test_ood_500.jsonl`)",
        "",
        "---",
        "",
        "## 1. Executive Summary & Category Breakdown",
        "",
        "| # | Taxonomy Category | Frequency | Proportion | Key Root Cause |",
        "|---|---|:---:|:---:|---|",
    ]

    remediation_map = {
        "Unhandled colloquial slang": "Expand chat slang lexicons (e.g. trailing particles 'กว่าๆ', 'นิดๆ', abbreviations with periods 'พน.').",
        "OCR / typo noise corruption": "Integrate Unicode canonical normalization (NFC/NFD) and fuzzy regex for common Thai keyboard shift typos.",
        "Compound multi-expression ambiguity": "Implement multi-entity disjunctive resolver or return candidates under candidate-return policy.",
        "Extreme date boundary / leap year edge cases": "Valid rejection behavior for impossible dates (29 ก.พ. non-leap, 30 ก.พ.); adjust token boundary around 'ประจำปี'.",
        "Coarse time interval mismatch": "Support user-injected custom convention table to override default coarse interval definitions.",
        "Length exceeded truncation": "By design under ReDoS protection SLA (< 5.0ms for <= 120 chars); document chunking recommendation for long documents.",
    }

    cat_idx = 1
    for cat_name, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(failures)) * 100
        rem = remediation_map.get(cat_name, "Rule extension")
        lines.append(f"| {cat_idx} | **{cat_name}** | {count} | {pct:.1f}% | {rem} |")
        cat_idx += 1

    lines.extend([
        "",
        "---",
        "",
        "## 2. Deep-Dive Diagnostic Case Studies",
        "",
    ])

    # Group by category and print top cases per category
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for f in failures:
        grouped.setdefault(f["taxonomy_category"], []).append(f)

    for cat_name, cat_failures in grouped.items():
        lines.extend([
            f"### Category: {cat_name} ({len(cat_failures)} cases)",
            "",
            f"**Remediation Architecture**: {remediation_map.get(cat_name, '')}",
            "",
            "| ID | Raw Input Text | Expected Ground Truth | Actual Model Output | Root Cause & Diagnostic |",
            "|:---:|---|---|---|---|",
        ])

        for f in cat_failures:
            gt_span = f["ground_truth"]["text_span"]
            gt_val = f["ground_truth"].get("iso_value") or f"{f['ground_truth'].get('range_start')}..{f['ground_truth'].get('range_end')}"
            preds = f["model_predictions"]
            if preds:
                pred_span = ", ".join(p["text_span"] for p in preds)
                pred_val = ", ".join(str(p.get("iso_value") or f"{p.get('range_start')}..{p.get('range_end')}") for p in preds)
                pred_str = f"`{pred_span}` ({pred_val})"
            else:
                pred_str = "*[] (Empty / Rejected)*"

            notes = f["notes"]
            lines.append(
                f"| {f['id']} | `{f['raw_input']}` | `{gt_span}` ({gt_val}) | {pred_str} | {notes} |"
            )

        lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. Engineering Recommendations for Next Sprint (v1.1.0)",
        "",
        "1. **Unicode NFC Preprocessor**: Add `unicodedata.normalize('NFC', text)` before regex matching to resolve reversed vowel codepoints (e.g. `พรุ่งนี้`).",
        "2. **Chat Slang Tokenizer**: Support trailing chat particles (`กว่าๆ`, `นิดๆ`, `เป๊ง`, `คล้อย`) and periods in abbreviations (`พน.`, `มรน.`).",
        "3. **Disjunctive Splitter**: Add explicit support for `A หรือ B` returning two candidate entities or candidate return list.",
        "4. **Long Document Chunking**: For texts > 120 characters, recommend sliding-window chunking (stride=100, window=120) before normalizer ingestion.",
        "",
    ])

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Generated comprehensive error analysis report at {output_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run error analysis on OOD test split.")
    parser.add_argument("--ood-file", type=Path, default=PROJECT_ROOT / "tests" / "data" / "test_ood_500.jsonl")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "benchmarks" / "error_analysis.json")
    parser.add_argument("--output-md", type=Path, default=PROJECT_ROOT / "ERROR_ANALYSIS.md")
    args = parser.parse_args()

    failures = run_error_analysis(args.ood_file)
    print(f"Identified {len(failures)} concrete failure cases in {args.ood_file.name}")
    assert len(failures) >= 20, f"Mandatory Requirement Violation: Found {len(failures)} < 20 failure cases!"

    # Save JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({
            "total_failures": len(failures),
            "source_file": str(args.ood_file.name),
            "cases": failures,
        }, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved error analysis data to {args.output_json}")

    # Save Markdown
    generate_markdown_report(failures, args.output_md)


if __name__ == "__main__":
    main()
