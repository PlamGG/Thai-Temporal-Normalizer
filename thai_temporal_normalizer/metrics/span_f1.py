"""Character-Span Offset F1 evaluation metric (Strict and Relaxed Overlap)."""

from __future__ import annotations

from typing import Dict, List, Set

from thai_temporal_normalizer.models import TemporalEntity


def compute_span_f1(
    pred_entities: List[TemporalEntity],
    gold_entities: List[TemporalEntity],
    relaxed: bool = False,
) -> Dict[str, float]:
    """Calculate Character-Span Offset Precision, Recall, and F1.

    Args:
        pred_entities: List of predicted TemporalEntity objects.
        gold_entities: List of ground-truth TemporalEntity objects.
        relaxed: If True, considers partial character overlap as a match.
                 If False, requires exact (start_char, end_char, type) match.

    Returns:
        Dict with keys: 'precision', 'recall', 'f1', 'tp', 'fp', 'fn'.
    """
    if not pred_entities and not gold_entities:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
        }

    matched_gold_indices: Set[int] = set()
    tp = 0

    for p in pred_entities:
        match_idx = None
        for idx, g in enumerate(gold_entities):
            if idx in matched_gold_indices:
                continue
            if p.type != g.type:
                continue

            if relaxed:
                # Overlap check
                overlap = max(0, min(p.end_char, g.end_char) - max(p.start_char, g.start_char))
                if overlap > 0:
                    match_idx = idx
                    break
            else:
                # Strict check
                if p.start_char == g.start_char and p.end_char == g.end_char:
                    match_idx = idx
                    break

        if match_idx is not None:
            matched_gold_indices.add(match_idx)
            tp += 1

    fp = len(pred_entities) - tp
    fn = len(gold_entities) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not gold_entities else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if not pred_entities else 0.0)
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }
