"""Value Accuracy metric for normalized ISO-8601 temporal representations."""

from __future__ import annotations

from typing import List

from thai_temporal_normalizer.metrics.range_iou import compute_range_iou
from thai_temporal_normalizer.models import TemporalEntity


def compute_value_accuracy(
    pred_entities: List[TemporalEntity],
    gold_entities: List[TemporalEntity],
    range_iou_threshold: float = 0.80,
) -> float:
    """Compute value accuracy proportion for matched temporal entities.

    For POINT entities: Requires exact match on iso_value string.
    For RANGE entities: Requires Range IoU >= range_iou_threshold or exact start/end match.
    """
    if not pred_entities and not gold_entities:
        return 1.0
    if not pred_entities or not gold_entities:
        return 0.0

    total_gold = len(gold_entities)
    correct_values = 0
    used_gold = set()

    for p in pred_entities:
        for idx, g in enumerate(gold_entities):
            if idx in used_gold:
                continue
            if p.type != g.type:
                continue

            if p.type == "POINT":
                if p.iso_value and g.iso_value and p.iso_value == g.iso_value:
                    correct_values += 1
                    used_gold.add(idx)
                    break
            elif p.type == "RANGE":
                # Check exact range or high IoU
                if p.range_start == g.range_start and p.range_end == g.range_end:
                    correct_values += 1
                    used_gold.add(idx)
                    break
                if p.range_start and p.range_end and g.range_start and g.range_end:
                    iou = compute_range_iou(p.range_start, p.range_end, g.range_start, g.range_end)
                    if iou >= range_iou_threshold:
                        correct_values += 1
                        used_gold.add(idx)
                        break

    return round(correct_values / total_gold, 4)
