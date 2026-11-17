"""Benchmark evaluation harness for Phase 0 Baseline Systems.

Evaluates 3 baseline systems against tests/data/baseline_150.jsonl:
1. dateparser (with Thai locale and relative base)
2. pythainlp composite wrapper (thaiword_to_date + thaiword_to_time)
3. Simple Regex Baseline (naive regex fallback)

Computes Error Rate (k/n), 95% Wilson Score Interval, per-category breakdown,
and evaluates the Go / Pivot / No-Go Decision Gate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dateparser
from pythainlp.util import thaiword_to_date, thaiword_to_time

BANGKOK_TZ = timezone(timedelta(hours=7))


# =============================================================================
# Mathematical Utilities
# =============================================================================

def compute_wilson_ci(
    k: int,
    n: int,
    z: float = 1.959964,
) -> Tuple[float, float]:
    """Compute 95% Wilson Score Confidence Interval for binomial proportion.

    Args:
        k: Number of errors (or successes).
        n: Total sample size.
        z: Normal distribution quantile (1.959964 for 95% two-sided).

    Returns:
        (lower_bound, upper_bound) rounded to 4 decimal places.
    """
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    den = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2.0 * n)) / den
    margin = (z / den) * math.sqrt((p * (1.0 - p)) / n + (z ** 2) / (4.0 * (n ** 2)))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return round(lower, 4), round(upper, 4)


def compute_iou_continuous(
    pred_start: datetime,
    pred_end: datetime,
    gt_start: datetime,
    gt_end: datetime,
) -> float:
    """Compute continuous interval Intersection over Union (IoU) in seconds."""
    int_start = max(pred_start, gt_start)
    int_end = min(pred_end, gt_end)
    intersection = max(0.0, (int_end - int_start).total_seconds())

    duration_pred = max(0.0, (pred_end - pred_start).total_seconds())
    duration_gt = max(0.0, (gt_end - gt_start).total_seconds())
    union = duration_pred + duration_gt - intersection

    if union <= 0.0:
        return 0.0
    return intersection / union


# =============================================================================
# Baseline Systems
# =============================================================================

class DateparserBaseline:
    """Baseline 1: dateparser with Thai language settings."""

    name = "dateparser"
    version = getattr(dateparser, "__version__", "1.5.0")

    def predict(self, text: str, ref_time: datetime) -> Optional[datetime]:
        try:
            res = dateparser.parse(
                text,
                languages=["th"],
                settings={
                    "RELATIVE_BASE": ref_time,
                    "TIMEZONE": "Asia/Bangkok",
                    "RETURN_AS_TIMEZONE_AWARE": True,
                },
            )
            return res
        except Exception:
            return None


class PyThaiNLPBaseline:
    """Baseline 2: PyThaiNLP composite wrapper combining date and time parsing."""

    name = "pythainlp_wrapper"
    version = "5.1.2"

    def predict(self, text: str, ref_time: datetime) -> Optional[datetime]:
        d_val: Optional[datetime] = None
        try:
            d_val = thaiword_to_date(text, date=ref_time)
        except Exception:
            d_val = None

        t_val: Optional[time] = None
        try:
            t_str = thaiword_to_time(text)
            if t_str and ":" in t_str:
                parts = t_str.split(":")
                t_val = time(hour=int(parts[0]), minute=int(parts[1]))
        except Exception:
            t_val = None

        if d_val is not None and t_val is not None:
            return d_val.replace(hour=t_val.hour, minute=t_val.minute, second=0)
        elif d_val is not None:
            return d_val
        elif t_val is not None:
            return ref_time.replace(hour=t_val.hour, minute=t_val.minute, second=0)
        return None


class SimpleRegexBaseline:
    """Baseline 3: Naive regular expression heuristics representing ad-hoc scripts."""

    name = "simple_regex"
    version = "1.0.0 (Native Python regex)"

    THAI_NUM_MAP = {
        "ศูนย์": 0, "หนึ่ง": 1, "เอ็ด": 1, "นึง": 1, "สอง": 2, "สาม": 3,
        "สี่": 4, "ห้า": 5, "หก": 6, "เจ็ด": 7, "แปด": 8, "เก้า": 9,
        "สิบ": 10, "สิบเอ็ด": 11, "สิบสอง": 12,
    }

    THAI_MONTH_MAP = {
        "ม.ค.": 1, "ก.พ.": 2, "มี.ค.": 3, "เม.ย.": 4, "พ.ค.": 5, "มิ.ย.": 6,
        "ก.ค.": 7, "ส.ค.": 8, "ก.ย.": 9, "ต.ค.": 10, "พ.ย.": 11, "ธ.ค.": 12,
    }

    def _parse_num(self, s: str) -> Optional[int]:
        s = s.strip()
        if s.isdigit():
            return int(s)
        return self.THAI_NUM_MAP.get(s)

    def predict(self, text: str, ref_time: datetime) -> Optional[datetime]:
        text = text.strip()

        # 1. Simple deictic days
        if "วันนี้" in text:
            return ref_time
        if "มะรืนนี้" in text:
            return ref_time + timedelta(days=2)
        if "พรุ่งนี้" in text:
            return ref_time + timedelta(days=1)
        if "วานซืน" in text:
            return ref_time - timedelta(days=2)
        if "เมื่อวาน" in text:
            return ref_time - timedelta(days=1)

        # 2. Thai 6-hour clock patterns
        # Noon
        if "เที่ยง" in text and "เที่ยงคืน" not in text:
            minute = 30 if "ครึ่ง" in text else 0
            return ref_time.replace(hour=12, minute=minute, second=0)
        if "เที่ยงคืน" in text:
            minute = 30 if "ครึ่ง" in text else 0
            return ref_time.replace(hour=0, minute=minute, second=0)

        # Early morning: ตี X
        m_ti = re.search(r"ตี\s*([0-9]+|หนึ่ง|สอง|สาม|สี่|ห้า)", text)
        if m_ti:
            h = self._parse_num(m_ti.group(1))
            if h is not None and 1 <= h <= 5:
                minute = 30 if "ครึ่ง" in text else 0
                return ref_time.replace(hour=h, minute=minute, second=0)

        # Night: X ทุ่ม
        m_thum = re.search(r"([0-9]+|หนึ่ง|นึง|สอง|สาม|สี่|ห้า)?\s*ทุ่ม", text)
        if m_thum:
            num_str = m_thum.group(1) or "หนึ่ง"
            h_raw = self._parse_num(num_str)
            if h_raw is not None and 1 <= h_raw <= 5:
                h = 18 + h_raw  # 1 ทุ่ม = 19:00
                minute = 30 if "ครึ่ง" in text else 0
                return ref_time.replace(hour=h, minute=minute, second=0)

        # Afternoon: บ่ายโมง / บ่าย X
        if "บ่าย" in text:
            m_bai = re.search(r"บ่าย\s*([0-9]+|โมง|สอง|สาม)?", text)
            if m_bai:
                tok = m_bai.group(1) or "โมง"
                if tok == "โมง" or tok == "หนึ่ง":
                    h = 13
                else:
                    h_val = self._parse_num(tok)
                    h = 12 + h_val if h_val else 13
                minute = 30 if "ครึ่ง" in text else 0
                return ref_time.replace(hour=h, minute=minute, second=0)

        # Morning: X โมง (เช้า)
        m_mong = re.search(r"([0-9]+|โมง|แปด|เก้า|สิบ|สิบเอ็ด)\s*โมง", text)
        if m_mong:
            tok = m_mong.group(1)
            h = 7 if tok == "โมง" else self._parse_num(tok)
            if h and 6 <= h <= 11:
                minute = 30 if "ครึ่ง" in text else 0
                return ref_time.replace(hour=h, minute=minute, second=0)

        # Evening: X โมงเย็น
        m_yen = re.search(r"([0-9]+|สี่|ห้า|หก)\s*โมงเย็น", text)
        if m_yen:
            h_val = self._parse_num(m_yen.group(1))
            if h_val:
                h = 12 + h_val
                minute = 30 if "ครึ่ง" in text else 0
                return ref_time.replace(hour=h, minute=minute, second=0)

        # 3. Calendar date format: DD Month ShortBE (without dynamic century pivot)
        m_date = re.search(r"(\d{1,2})\s*([ก-๙\.]+)\s*(\d{2,4})?", text)
        if m_date:
            day = int(m_date.group(1))
            m_str = m_date.group(2)
            y_str = m_date.group(3)
            m_num = self.THAI_MONTH_MAP.get(m_str)
            if m_num:
                # Naive regex assumes Gregorian year or passes raw 2-digit without BE pivot
                year = 2000 + int(y_str) if y_str and len(y_str) == 2 else (int(y_str) if y_str else ref_time.year)
                try:
                    return datetime(year, m_num, day, 0, 0, 0, tzinfo=ref_time.tzinfo)
                except ValueError:
                    return None

        return None


# =============================================================================
# Benchmark Evaluator
# =============================================================================

@dataclass
class SystemCategoryResult:
    category: str
    total: int
    correct: int
    errors: int
    error_rate: float
    wilson_ci: Tuple[float, float]


@dataclass
class SystemOverallResult:
    system_name: str
    version: str
    total: int
    correct: int
    errors: int
    error_rate: float
    wilson_ci: Tuple[float, float]
    per_category: Dict[str, SystemCategoryResult]
    gate_decision: str


class BenchmarkEvaluator:
    """Evaluates baseline systems against annotated dataset."""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.items: List[Dict[str, Any]] = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    @staticmethod
    def _parse_iso(iso_str: Optional[str]) -> Optional[datetime]:
        if not iso_str:
            return None
        return datetime.fromisoformat(iso_str)

    def evaluate_item(
        self,
        system: Any,
        item: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Evaluate a single test item.

        Returns:
            (is_correct, pred_str, error_reason)
        """
        raw_text = item["raw_input"]
        gt = item["ground_truth"]
        meta = item["metadata"]
        ref_time = datetime.fromisoformat(meta["reference_time"])

        # Invoke system
        pred = system.predict(raw_text, ref_time)
        pred_str = pred.isoformat() if isinstance(pred, datetime) else str(pred)

        if pred is None:
            return False, None, "system_returned_none"

        gt_type = gt.get("type", "POINT")

        # 1. POINT Entity Evaluation
        if gt_type == "POINT":
            if not isinstance(pred, datetime):
                return False, pred_str, "expected_datetime_point"

            gt_iso = gt.get("iso_value")
            if not gt_iso:
                return False, pred_str, "missing_ground_truth_iso"

            gt_dt = datetime.fromisoformat(gt_iso)
            grain = gt.get("grain", "minute")

            if grain == "year":
                matched = (pred.year == gt_dt.year)
            elif grain == "day":
                matched = (pred.year == gt_dt.year and pred.month == gt_dt.month and pred.day == gt_dt.day)
            else:  # minute, hour, second
                matched = (
                    pred.year == gt_dt.year
                    and pred.month == gt_dt.month
                    and pred.day == gt_dt.day
                    and pred.hour == gt_dt.hour
                    and pred.minute == gt_dt.minute
                )

            if matched:
                return True, pred_str, None
            else:
                return False, pred_str, f"point_mismatch (pred={pred.isoformat()}, gt={gt_iso})"

        # 2. RANGE Entity Evaluation
        elif gt_type == "RANGE":
            gt_start = self._parse_iso(gt.get("range_start"))
            gt_end = self._parse_iso(gt.get("range_end"))
            if not gt_start or not gt_end:
                return False, pred_str, "missing_ground_truth_range"

            # If system returned single datetime, IoU with continuous interval is 0.0
            if isinstance(pred, datetime):
                return False, pred_str, "point_output_for_range_ground_truth (IoU=0.0 < 0.80)"

            # If system returned a tuple/dict of range
            if isinstance(pred, tuple) and len(pred) == 2:
                iou = compute_iou_continuous(pred[0], pred[1], gt_start, gt_end)
                if iou >= 0.80:
                    return True, pred_str, None
                return False, pred_str, f"low_iou ({iou:.4f} < 0.80)"

            return False, pred_str, "unsupported_range_output_format"

        return False, pred_str, f"unknown_gt_type_{gt_type}"

    def evaluate_system(
        self,
        system: Any,
        raw_pred_file: Optional[Path] = None,
    ) -> SystemOverallResult:
        """Run benchmark evaluation on one system across all items."""
        categories = sorted(list({item["category"] for item in self.items}))
        cat_stats: Dict[str, Dict[str, int]] = {c: {"total": 0, "correct": 0, "errors": 0} for c in categories}
        overall_total = 0
        overall_correct = 0
        overall_errors = 0

        raw_logs: List[Dict[str, Any]] = []

        for item in self.items:
            cat = item["category"]
            is_correct, pred_str, reason = self.evaluate_item(system, item)

            cat_stats[cat]["total"] += 1
            overall_total += 1

            if is_correct:
                cat_stats[cat]["correct"] += 1
                overall_correct += 1
            else:
                cat_stats[cat]["errors"] += 1
                overall_errors += 1

            raw_logs.append({
                "item_id": item["id"],
                "category": cat,
                "raw_input": item["raw_input"],
                "system": getattr(system, "name", "unknown"),
                "is_correct": is_correct,
                "prediction": pred_str,
                "ground_truth": item["ground_truth"].get("iso_value") or {
                    "start": item["ground_truth"].get("range_start"),
                    "end": item["ground_truth"].get("range_end"),
                },
                "error_reason": reason,
            })

        if raw_pred_file:
            with open(raw_pred_file, "a", encoding="utf-8") as f:
                for log in raw_logs:
                    f.write(json.dumps(log, ensure_ascii=False) + "\n")

        # Per category results
        per_cat_results: Dict[str, SystemCategoryResult] = {}
        for cat in categories:
            t = cat_stats[cat]["total"]
            c = cat_stats[cat]["correct"]
            e = cat_stats[cat]["errors"]
            rate = round(e / t, 4) if t > 0 else 0.0
            ci = compute_wilson_ci(e, t)
            per_cat_results[cat] = SystemCategoryResult(
                category=cat,
                total=t,
                correct=c,
                errors=e,
                error_rate=rate,
                wilson_ci=ci,
            )

        overall_rate = round(overall_errors / overall_total, 4) if overall_total > 0 else 0.0
        overall_ci = compute_wilson_ci(overall_errors, overall_total)

        # Gate Decision: Go threshold > 30% error rate
        if overall_rate > 0.30:
            decision = "GO"
        elif overall_rate >= 0.10:
            decision = "PIVOT"
        else:
            decision = "NO-GO"

        return SystemOverallResult(
            system_name=getattr(system, "name", "unknown"),
            version=getattr(system, "version", "unknown"),
            total=overall_total,
            correct=overall_correct,
            errors=overall_errors,
            error_rate=overall_rate,
            wilson_ci=overall_ci,
            per_category=per_cat_results,
            gate_decision=decision,
        )


# =============================================================================
# Main CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 Baseline Benchmark Harness")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "data" / "baseline_150.jsonl",
        help="Path to baseline 150 JSONL dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "baseline_results.json",
        help="Path to output baseline results JSON",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "predictions_raw.jsonl",
        help="Path to raw prediction logs JSONL",
    )
    args = parser.parse_args()

    print(f"Loading benchmark dataset from: {args.data}")
    evaluator = BenchmarkEvaluator(args.data)
    print(f"Loaded {len(evaluator.items)} items across categories.")

    # Clear raw log file if exists
    if args.raw.exists():
        args.raw.unlink()

    systems = [
        DateparserBaseline(),
        PyThaiNLPBaseline(),
        SimpleRegexBaseline(),
    ]

    all_results: Dict[str, Any] = {
        "benchmark_metadata": {
            "date": datetime.now(BANGKOK_TZ).isoformat(),
            "sample_size": len(evaluator.items),
            "go_threshold": "Error Rate > 30%",
        },
        "systems": {},
    }

    print("\n" + "=" * 80)
    print("PHASE 0 BASELINE BENCHMARK EVALUATION")
    print("=" * 80)

    for sys in systems:
        print(f"\nEvaluating: {sys.name} (version: {sys.version}) ...")
        result = evaluator.evaluate_system(sys, raw_pred_file=args.raw)
        all_results["systems"][sys.name] = asdict(result)

        print(f"  Overall Total:      {result.total}")
        print(f"  Correct:            {result.correct}")
        print(f"  Errors:             {result.errors}")
        print(f"  Error Rate (p_hat): {result.error_rate * 100:.2f}%")
        print(f"  95% Wilson CI:      [{result.wilson_ci[0] * 100:.2f}%, {result.wilson_ci[1] * 100:.2f}%]")
        print(f"  Gate Verdict:       {result.gate_decision}")
        print("  Per-Category Breakdown:")
        for cat, cres in result.per_category.items():
            print(f"    - {cat:22s}: {cres.errors:2d}/{cres.total:2d} errors ({cres.error_rate * 100:.1f}%), CI: [{cres.wilson_ci[0]*100:.1f}%, {cres.wilson_ci[1]*100:.1f}%]")

    # Check overall gate decision
    all_go = all(res["gate_decision"] == "GO" for res in all_results["systems"].values())
    all_results["final_gate_verdict"] = "GO" if all_go else "REVIEW"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"Final Decision Gate: {all_results['final_gate_verdict']}")
    print(f"Results saved to: {args.output}")
    print(f"Raw prediction logs saved to: {args.raw}")
    print("=" * 80)


if __name__ == "__main__":
    main()
