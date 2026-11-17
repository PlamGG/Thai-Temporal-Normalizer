"""Rigorous Evaluation Protocol for Thai-Temporal-Normalizer (Phase 2).

Evaluates parse_thai_temporal against:
1. tests/data/test_id_500.jsonl (In-Distribution test set)
2. tests/data/test_ood_500.jsonl (Out-of-Distribution blind test set)

Calculates:
- Character-Span Offset F1 (Exact start_char, end_char, type)
- Value Accuracy (Exact match for POINT, continuous Range IoU >= 0.80 for RANGE)
- Point Exact Match (Point EM)
- Range Mean IoU (with midnight wrapping decomposition)
- 95% Bootstrap Confidence Intervals (B = 1,000 iterations via empirical percentile method)

Outputs:
- benchmarks/evaluation_results.json (comprehensive metrics + CIs)
- predictions_raw.jsonl (appends per-item prediction records)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from thai_temporal_normalizer import TemporalEntity, parse_thai_temporal  # noqa: E402
from thai_temporal_normalizer.metrics.range_iou import compute_range_iou  # noqa: E402
from thai_temporal_normalizer.metrics.span_f1 import compute_span_f1  # noqa: E402
from thai_temporal_normalizer.metrics.value_accuracy import compute_value_accuracy  # noqa: E402


# =============================================================================
# Evaluation Logic
# =============================================================================

def evaluate_sample(
    raw_input: str,
    gt_dict: Dict[str, Any],
    ref_time: datetime,
) -> Dict[str, Any]:
    """Evaluate a single test instance."""
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

    t0 = time.perf_counter()
    preds = parse_thai_temporal(raw_input, reference_time=ref_time)
    t1 = time.perf_counter()
    lat_ms = (t1 - t0) * 1000.0

    # Span metrics
    span_res = compute_span_f1(preds, [gt_ent], relaxed=False)
    span_f1 = span_res["f1"]
    tp = span_res["tp"]
    fp = span_res["fp"]
    fn = span_res["fn"]

    # Value accuracy
    val_acc = compute_value_accuracy(preds, [gt_ent], range_iou_threshold=0.80)
    is_val_correct = (val_acc >= 1.0)

    # Specific metrics
    point_match = None
    range_iou = None

    if gt_ent.type == "POINT":
        if preds and preds[0].type == "POINT" and preds[0].iso_value and gt_ent.iso_value:
            point_match = (preds[0].iso_value == gt_ent.iso_value)
        else:
            point_match = False
    elif gt_ent.type == "RANGE":
        if preds and preds[0].type == "RANGE" and preds[0].range_start and preds[0].range_end and gt_ent.range_start and gt_ent.range_end:
            p_s = preds[0].range_start
            p_e = preds[0].range_end
            g_s = gt_ent.range_start
            g_e = gt_ent.range_end
            range_iou = compute_range_iou(p_s, p_e, g_s, g_e)
        else:
            range_iou = 0.0

    return {
        "predictions": preds,
        "latency_ms": lat_ms,
        "span_f1": span_f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "value_match": is_val_correct,
        "point_match": point_match,
        "range_iou": range_iou,
    }


def compute_bootstrap_ci(
    values: np.ndarray,
    b_iterations: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Compute empirical percentile 95% bootstrap confidence interval.

    Returns:
        (mean_estimate, ci_lower_2.5, ci_upper_97.5, standard_error)
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    rng = np.random.RandomState(seed)
    bootstrap_means = np.empty(b_iterations, dtype=float)

    for b in range(b_iterations):
        indices = rng.randint(0, n, size=n)
        bootstrap_means[b] = np.mean(values[indices])

    mean_est = float(np.mean(values))
    ci_lower = float(np.percentile(bootstrap_means, 2.5))
    ci_upper = float(np.percentile(bootstrap_means, 97.5))
    std_err = float(np.std(bootstrap_means))

    return round(mean_est, 4), round(ci_lower, 4), round(ci_upper, 4), round(std_err, 4)


def evaluate_dataset(
    dataset_path: Path,
    split_name: str,
    b_iterations: int = 1000,
    seed: int = 42,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run full evaluation on a dataset split with Bootstrap CIs."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    print(f"\nEvaluating {split_name} ({len(items)} samples) from {dataset_path.name}...")

    raw_prediction_records: List[Dict[str, Any]] = []
    sample_span_f1s: List[float] = []
    sample_val_matches: List[float] = []
    sample_point_matches: List[float] = []
    sample_range_ious: List[float] = []
    latencies: List[float] = []

    for item in items:
        raw_input = item["raw_input"]
        gt_dict = item["ground_truth"]
        ref_time = datetime.fromisoformat(item["metadata"]["reference_time"])

        eval_res = evaluate_sample(raw_input, gt_dict, ref_time)

        sample_span_f1s.append(eval_res["span_f1"])
        sample_val_matches.append(1.0 if eval_res["value_match"] else 0.0)
        latencies.append(eval_res["latency_ms"])

        if eval_res["point_match"] is not None:
            sample_point_matches.append(1.0 if eval_res["point_match"] else 0.0)

        if eval_res["range_iou"] is not None:
            sample_range_ious.append(eval_res["range_iou"])

        # Format record for predictions_raw.jsonl
        raw_prediction_records.append({
            "id": item["id"],
            "split": split_name,
            "text": raw_input,
            "reference_time": item["metadata"]["reference_time"],
            "category": item.get("category", "unknown"),
            "ground_truth": [gt_dict],
            "model_prediction": [p.to_dict() for p in eval_res["predictions"]],
            "span_f1": eval_res["span_f1"],
            "value_match": eval_res["value_match"],
            "point_match": eval_res["point_match"],
            "range_iou": eval_res["range_iou"],
            "latency_ms": round(eval_res["latency_ms"], 4),
        })

    # Compute Bootstrap Confidence Intervals
    span_mean, span_l, span_u, span_se = compute_bootstrap_ci(np.array(sample_span_f1s), b_iterations, seed)
    val_mean, val_l, val_u, val_se = compute_bootstrap_ci(np.array(sample_val_matches), b_iterations, seed)

    point_em_mean, point_l, point_u, point_se = compute_bootstrap_ci(
        np.array(sample_point_matches) if sample_point_matches else np.array([0.0]),
        b_iterations,
        seed,
    )
    range_iou_mean, range_l, range_u, range_se = compute_bootstrap_ci(
        np.array(sample_range_ious) if sample_range_ious else np.array([0.0]),
        b_iterations,
        seed,
    )

    metrics_summary = {
        "split_name": split_name,
        "sample_count": len(items),
        "metrics": {
            "span_f1": {
                "point_estimate": span_mean,
                "ci_95": [span_l, span_u],
                "std_err": span_se,
            },
            "value_accuracy": {
                "point_estimate": val_mean,
                "ci_95": [val_l, val_u],
                "std_err": val_se,
            },
            "point_exact_match": {
                "point_estimate": point_em_mean,
                "sample_count": len(sample_point_matches),
                "ci_95": [point_l, point_u],
                "std_err": point_se,
            },
            "range_mean_iou": {
                "point_estimate": range_iou_mean,
                "sample_count": len(sample_range_ious),
                "ci_95": [range_l, range_u],
                "std_err": range_se,
            },
        },
        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 4),
            "p50": round(float(np.percentile(latencies, 50)), 4),
            "p95": round(float(np.percentile(latencies, 95)), 4),
            "p99": round(float(np.percentile(latencies, 99)), 4),
        },
    }

    print(f"Results for {split_name}:")
    print(f"  Span F1:        {span_mean:.4f} [95% CI: {span_l:.4f} - {span_u:.4f}]")
    print(f"  Value Accuracy: {val_mean:.4f} [95% CI: {val_l:.4f} - {val_u:.4f}]")
    print(f"  Point EM:       {point_em_mean:.4f} [95% CI: {point_l:.4f} - {point_u:.4f}]")
    print(f"  Range Mean IoU: {range_iou_mean:.4f} [95% CI: {range_l:.4f} - {range_u:.4f}]")
    print(f"  Latency p95:    {metrics_summary['latency_ms']['p95']:.4f} ms")

    return metrics_summary, raw_prediction_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Thai-Temporal-Normalizer on test splits.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "tests" / "data")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "benchmarks" / "evaluation_results.json")
    parser.add_argument("--raw-predictions", type=Path, default=PROJECT_ROOT / "predictions_raw.jsonl")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    id_path = data_dir / "test_id_500.jsonl"
    ood_path = data_dir / "test_ood_500.jsonl"

    if not id_path.exists():
        raise FileNotFoundError(f"Missing {id_path}")
    if not ood_path.exists():
        raise FileNotFoundError(f"Missing {ood_path}")

    # Evaluate ID Test set
    id_summary, id_raw_records = evaluate_dataset(
        id_path, "test_id_500", b_iterations=args.bootstrap_iterations, seed=args.seed
    )

    # Evaluate OOD Blind Test set
    ood_summary, ood_raw_records = evaluate_dataset(
        ood_path, "test_ood_500", b_iterations=args.bootstrap_iterations, seed=args.seed
    )

    # Verify Target SLAs
    # In-Distribution targets: Span F1 >= 0.95, Value Acc >= 0.95
    id_targets_met = {
        "span_f1_ge_0_95": id_summary["metrics"]["span_f1"]["point_estimate"] >= 0.95,
        "value_acc_ge_0_95": id_summary["metrics"]["value_accuracy"]["point_estimate"] >= 0.95,
    }

    # OOD Blind Test targets: Span F1 >= 0.88, Value Acc >= 0.85, Point EM >= 0.85, Range IoU >= 0.80
    ood_targets_met = {
        "span_f1_ge_0_88": ood_summary["metrics"]["span_f1"]["point_estimate"] >= 0.88,
        "value_acc_ge_0_85": ood_summary["metrics"]["value_accuracy"]["point_estimate"] >= 0.85,
        "point_em_ge_0_85": ood_summary["metrics"]["point_exact_match"]["point_estimate"] >= 0.85,
        "range_iou_ge_0_80": ood_summary["metrics"]["range_mean_iou"]["point_estimate"] >= 0.80,
    }

    id_summary["targets_met"] = id_targets_met
    ood_summary["targets_met"] = ood_targets_met

    full_results = {
        "timestamp": datetime.now().isoformat(),
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "targets": {
            "test_id": {
                "span_f1": ">= 0.95",
                "value_accuracy": ">= 0.95",
            },
            "test_ood": {
                "span_f1": ">= 0.88",
                "value_accuracy": ">= 0.85",
                "point_exact_match": ">= 0.85",
                "range_mean_iou": ">= 0.80",
            },
        },
        "test_id_500": id_summary,
        "test_ood_500": ood_summary,
        "overall_status": "PASS" if (all(id_targets_met.values()) and all(ood_targets_met.values())) else "FAIL",
    }

    # Save evaluation_results.json
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved evaluation results to {args.output_json}")

    # Read existing records from predictions_raw.jsonl to preserve Phase 0 baseline
    existing_phase0_records: List[Dict[str, Any]] = []
    if args.raw_predictions.exists():
        with open(args.raw_predictions, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if "system" in item:
                        existing_phase0_records.append(item)

    # Write back Phase 0 + Phase 2 idempotently
    with open(args.raw_predictions, "w", encoding="utf-8") as f:
        for r in existing_phase0_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in id_raw_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in ood_raw_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote {len(existing_phase0_records)} Phase 0 + {len(id_raw_records) + len(ood_raw_records)} Phase 2 records to {args.raw_predictions}")

    print("\n" + "=" * 60)
    print(f"OVERALL EVALUATION STATUS: {full_results['overall_status']}")
    print("=" * 60)
    for k, v in id_targets_met.items():
        print(f"ID Target  {k:<20}: {'[PASS]' if v else '[FAIL]'}")
    for k, v in ood_targets_met.items():
        print(f"OOD Target {k:<20}: {'[PASS]' if v else '[FAIL]'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
