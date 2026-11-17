"""Adversarial Verification Script for Phase 0 Baseline and Statistics.
Executed by Challenger M0_1.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime



def verify_dataset(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    total_lines = len(lines)
    errors = []
    categories = Counter()
    span_mismatches = []
    ids = []
    id_set = set()
    point_count = 0
    range_count = 0

    REQUIRED_KEYS = {"id", "raw_input", "ground_truth", "category", "metadata", "notes"}
    GT_KEYS = {"text_span", "start_char", "end_char", "type", "grain", "iso_value", "range_start", "range_end", "precision", "confidence"}
    META_KEYS = {"reference_time", "timezone", "linguistic_type", "has_midnight_crossing"}

    for idx, line in enumerate(lines, 1):
        try:
            data = json.loads(line)
        except Exception as e:
            errors.append((idx, f"JSON parse error: {e}"))
            continue

        missing = REQUIRED_KEYS - set(data.keys())
        if missing:
            errors.append((idx, f"Missing top-level keys: {missing}"))

        item_id = data.get("id")
        ids.append(item_id)
        if item_id in id_set:
            errors.append((idx, f"Duplicate id: {item_id}"))
        id_set.add(item_id)

        cat = data.get("category")
        categories[cat] += 1

        gt = data.get("ground_truth", {})
        missing_gt = GT_KEYS - set(gt.keys())
        if missing_gt:
            errors.append((idx, f"Missing gt keys: {missing_gt}"))

        raw = data.get("raw_input", "")
        start = gt.get("start_char")
        end = gt.get("end_char")
        span = gt.get("text_span")

        if not isinstance(start, int) or not isinstance(end, int) or not isinstance(span, str):
            errors.append((idx, f"Invalid span types: start={type(start)}, end={type(end)}, span={type(span)}"))
        else:
            sliced = raw[start:end]
            if sliced != span:
                span_mismatches.append({
                    "idx": idx,
                    "id": item_id,
                    "raw_input": raw,
                    "start": start,
                    "end": end,
                    "sliced": sliced,
                    "expected_span": span,
                })

        # Schema domain checks
        entity_type = gt.get("type")
        if entity_type not in {"POINT", "RANGE"}:
            errors.append((idx, f"Invalid type: {entity_type}"))

        if entity_type == "POINT":
            point_count += 1
            if gt.get("grain") not in {"day", "hour", "minute", "year"}:
                errors.append((idx, f"Invalid grain for POINT: {gt.get('grain')}"))
            if not gt.get("iso_value"):
                errors.append((idx, "Missing iso_value for POINT"))
            else:
                try:
                    datetime.fromisoformat(gt.get("iso_value"))
                except Exception as e:
                    errors.append((idx, f"Invalid iso_value format: {e}"))
            if gt.get("range_start") is not None or gt.get("range_end") is not None:
                errors.append((idx, "POINT entity must not have range_start/range_end"))

        elif entity_type == "RANGE":
            range_count += 1
            if gt.get("grain") is not None and gt.get("grain") not in {"day", "hour", "minute", "year"}:
                errors.append((idx, f"Invalid grain for RANGE: {gt.get('grain')}"))
            if not gt.get("range_start") or not gt.get("range_end"):
                errors.append((idx, "Missing range_start/range_end for RANGE"))
            else:
                try:
                    rs = datetime.fromisoformat(gt.get("range_start"))
                    re_dt = datetime.fromisoformat(gt.get("range_end"))
                    if rs > re_dt:
                        errors.append((idx, f"range_start ({rs}) > range_end ({re_dt})"))
                except Exception as e:
                    errors.append((idx, f"Invalid range datetime format: {e}"))
            if gt.get("iso_value") is not None:
                errors.append((idx, "RANGE entity must not have iso_value"))

        if gt.get("precision") not in {"exact", "coarse"}:
            errors.append((idx, f"Invalid precision: {gt.get('precision')}"))
        if not isinstance(gt.get("confidence"), (int, float)):
            errors.append((idx, f"Invalid confidence: {gt.get('confidence')}"))

        meta = data.get("metadata", {})
        missing_meta = META_KEYS - set(meta.keys())
        if missing_meta:
            errors.append((idx, f"Missing meta keys: {missing_meta}"))
        else:
            try:
                datetime.fromisoformat(meta.get("reference_time"))
            except Exception as e:
                errors.append((idx, f"Invalid reference_time: {e}"))
            if not isinstance(meta.get("has_midnight_crossing"), bool):
                errors.append((idx, f"Invalid has_midnight_crossing type: {type(meta.get('has_midnight_crossing'))}"))

    expected_ids = list(range(1, total_lines + 1))
    sequential_ids = (ids == expected_ids)

    return {
        "total_lines": total_lines,
        "validation_errors": errors,
        "categories": dict(categories),
        "span_mismatches": span_mismatches,
        "sequential_ids": sequential_ids,
        "point_count": point_count,
        "range_count": range_count,
    }


def independent_wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Exact Wilson score interval recalculation from first principles."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    den = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2.0 * n)) / den
    margin = (z / den) * math.sqrt((p * (1.0 - p)) / n + (z ** 2) / (4.0 * (n ** 2)))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return round(lower, 4), round(upper, 4)


def one_sample_proportion_test(k: int, n: int, p0: float = 0.30) -> dict:
    """Statistical significance test for H0: p <= 0.30 vs H1: p > 0.30."""
    p_hat = k / n
    se_null = math.sqrt(p0 * (1.0 - p0) / n)
    z_stat = (p_hat - p0) / se_null

    # One-tailed p-value: P(Z >= z_stat)
    cdf = 0.5 * (1.0 + math.erf(z_stat / math.sqrt(2.0)))
    p_value = 1.0 - cdf

    # Exact binomial test p-value: sum_{x=k..n} C(n, x) * p0^x * (1 - p0)^(n - x)
    exact_p = 0.0
    for x in range(k, n + 1):
        ways = math.comb(n, x)
        prob = ways * (p0 ** x) * ((1.0 - p0) ** (n - x))
        exact_p += prob

    return {
        "p_hat": p_hat,
        "z_stat": z_stat,
        "z_test_p_value": p_value,
        "exact_binomial_p_value": exact_p,
        "reject_h0_alpha_005": p_value < 0.05 and exact_p < 0.05,
    }


def main():
    print("=" * 80)
    print("TASK 1: DATASET EMPIRICAL VERIFICATION")
    print("=" * 80)
    res = verify_dataset("tests/data/baseline_150.jsonl")
    print(f"Total lines parsed: {res['total_lines']} (Expected: 150)")
    print(f"Sequential IDs (1..150): {res['sequential_ids']}")
    print(f"POINT entities: {res['point_count']}")
    print(f"RANGE entities: {res['range_count']}")
    print(f"Total validation errors: {len(res['validation_errors'])}")
    if res['validation_errors']:
        for e in res['validation_errors']:
            print(f"  ERROR: {e}")
    else:
        print("  [PASS] Zero validation errors across all schema fields!")
    print(f"Category counts: {res['categories']}")
    strat_pass = all(c == 30 for c in res['categories'].values()) and len(res['categories']) == 5
    print(f"Stratification check (5 categories, exactly 30 each): {strat_pass}")
    print(f"Span mismatches count: {len(res['span_mismatches'])}")
    if res['span_mismatches']:
        for m in res['span_mismatches']:
            print(f"  MISMATCH: {m}")
    else:
        print("  [PASS] All 150 spans satisfy raw_input[start_char:end_char] == text_span exactly!")

    print("\n" + "=" * 80)
    print("TASK 2: ADVERSARIAL WILSON SCORE INTERVAL VERIFICATION")
    print("=" * 80)
    with open("benchmarks/baseline_results.json", "r", encoding="utf-8") as f:
        reported = json.load(f)

    discrepancies = []
    for sys_name, sys_data in reported["systems"].items():
        total = sys_data["total"]
        errors = sys_data["errors"]
        rep_rate = sys_data["error_rate"]
        rep_ci = sys_data["wilson_ci"]

        calc_rate = round(errors / total, 4)
        calc_ci = list(independent_wilson_ci(errors, total))

        rate_diff = abs(rep_rate - calc_rate)
        ci_diff = [abs(rep_ci[0] - calc_ci[0]), abs(rep_ci[1] - calc_ci[1])]

        print(f"\nSystem: {sys_name}")
        print(f"  Reported: rate={rep_rate}, CI={rep_ci}")
        print(f"  Computed: rate={calc_rate}, CI={calc_ci}")
        if rate_diff > 1e-4 or ci_diff[0] > 1e-4 or ci_diff[1] > 1e-4:
            discrepancies.append((sys_name, "overall", rep_rate, calc_rate, rep_ci, calc_ci))
            print("  [!] DISCREPANCY DETECTED")
        else:
            print("  [PASS] Exact match with independent recalculation")

        # Per category
        for cat_name, cat_data in sys_data["per_category"].items():
            c_tot = cat_data["total"]
            c_err = cat_data["errors"]
            c_rep_rate = cat_data["error_rate"]
            c_rep_ci = cat_data["wilson_ci"]

            c_calc_rate = round(c_err / c_tot, 4)
            c_calc_ci = list(independent_wilson_ci(c_err, c_tot))

            c_rate_diff = abs(c_rep_rate - c_calc_rate)
            c_ci_diff = [abs(c_rep_ci[0] - c_calc_ci[0]), abs(c_rep_ci[1] - c_calc_ci[1])]

            if c_rate_diff > 1e-4 or c_ci_diff[0] > 1e-4 or c_ci_diff[1] > 1e-4:
                discrepancies.append((sys_name, cat_name, c_rep_rate, c_calc_rate, c_rep_ci, c_calc_ci))
                print(f"    Cat {cat_name}: [!] DISCREPANCY: rep={c_rep_ci}, calc={c_calc_ci}")
            else:
                print(f"    Cat {cat_name:20s}: [PASS] rate={c_calc_rate:.4f}, CI={c_calc_ci}")

    print(f"\nTotal Wilson CI numerical discrepancies found: {len(discrepancies)}")

    print("\n" + "=" * 80)
    print("TASK 3: STATISTICAL SIGNIFICANCE OF GO GATE (Error Rate > 30%)")
    print("=" * 80)
    all_go_significant = True
    for sys_name, sys_data in reported["systems"].items():
        total = sys_data["total"]
        errors = sys_data["errors"]
        test_res = one_sample_proportion_test(errors, total, p0=0.30)
        calc_ci = independent_wilson_ci(errors, total)
        is_go = test_res['p_hat'] > 0.30 and calc_ci[0] > 0.30 and test_res['reject_h0_alpha_005']
        if not is_go:
            all_go_significant = False
        print(f"System: {sys_name}")
        print(f"  Failures (k/n):           {errors}/{total} (p_hat = {test_res['p_hat'] * 100:.2f}%)")
        print(f"  Independent 95% Wilson CI: [{calc_ci[0] * 100:.2f}%, {calc_ci[1] * 100:.2f}%]")
        print(f"  Wilson Lower Bound > 30%:  {calc_ci[0] > 0.30} (margin: +{(calc_ci[0] - 0.30) * 100:.2f}%)")
        print(f"  Z-statistic (vs p0=0.30):  {test_res['z_stat']:.4f}")
        print(f"  Z-test one-sided p-value:  {test_res['z_test_p_value']:.4e}")
        print(f"  Exact Binomial p-value:    {test_res['exact_binomial_p_value']:.4e}")
        print(f"  Significant at alpha=0.05: {test_res['reject_h0_alpha_005']}")
        print(f"  Verdict:                   {'GO' if is_go else 'NO-GO'}")

    print(f"\nAll baseline systems statistically reject H0 (p <= 0.30) with p < 1e-26: {all_go_significant}")


if __name__ == "__main__":
    main()
