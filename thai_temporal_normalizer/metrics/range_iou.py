"""Continuous Range IoU metric with midnight-crossing canonical decomposition."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import List, Tuple, Union

from thai_temporal_normalizer.rules.constants import BANGKOK_TZ


def _to_datetime(val: Union[datetime, str, time], base_date: datetime) -> datetime:
    """Normalize input into an absolute datetime object with consistent timezone awareness."""
    tz = base_date.tzinfo or BANGKOK_TZ
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=tz)
        return val
    if isinstance(val, str):
        # Check if full ISO datetime or HH:MM(:SS) time
        if "T" in val or " " in val:
            dt = datetime.fromisoformat(val)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=tz)
            return dt
        parts = list(map(int, val.split(":")))
        h = parts[0]
        m = parts[1] if len(parts) > 1 else 0
        s = parts[2] if len(parts) > 2 else 0
        if h == 24:
            return (base_date + timedelta(days=1)).replace(hour=0, minute=m, second=s, microsecond=0, tzinfo=tz)
        return base_date.replace(hour=h, minute=m, second=s, microsecond=0, tzinfo=tz)
    if isinstance(val, time):
        val_tz = val.tzinfo or tz
        return base_date.replace(hour=val.hour, minute=val.minute, second=val.second, microsecond=0, tzinfo=val_tz)
    raise TypeError(f"Unsupported datetime/time type: {type(val)}")


def decompose_midnight_interval(
    start: Union[datetime, str, time],
    end: Union[datetime, str, time],
) -> List[Tuple[float, float]]:
    """Decompose an interval into canonical daily sub-intervals in seconds [0.0, 86400.0].

    For example, [22:00, 02:00) decomposes into:
        [79200.0, 86400.0) and [0.0, 7200.0)
    """
    ref_d = datetime(2026, 1, 1, tzinfo=BANGKOK_TZ)
    dt_s = _to_datetime(start, ref_d)
    dt_e = _to_datetime(end, ref_d)

    # Convert to seconds from midnight of respective day
    s_sec = dt_s.hour * 3600.0 + dt_s.minute * 60.0 + dt_s.second
    e_sec = dt_e.hour * 3600.0 + dt_e.minute * 60.0 + dt_e.second

    if dt_e < dt_s or (isinstance(start, (str, time)) and e_sec < s_sec):
        # Midnight crossing: [s_sec, 86400.0) and [0.0, e_sec)
        return [(s_sec, 86400.0), (0.0, e_sec)]
    return [(s_sec, e_sec)]


def compute_range_iou(
    pred_start: Union[datetime, str, time],
    pred_end: Union[datetime, str, time],
    gt_start: Union[datetime, str, time],
    gt_end: Union[datetime, str, time],
) -> float:
    """Compute continuous interval Intersection over Union (IoU) with midnight wrapping.

    Formula:
        IoU = Duration(R_pred ∩ R_true) / Duration(R_pred ∪ R_true)

    Handles:
        - Regular within-day intervals (e.g. 09:00 to 11:30)
        - Midnight-crossing intervals (e.g. 22:00 to 02:00 next day)
        - Absolute datetime ranges with dates
    """
    # If both inputs are explicit full datetimes across distinct dates:
    ref_d = datetime(2026, 1, 1, tzinfo=BANGKOK_TZ)
    dt_ps = _to_datetime(pred_start, ref_d)
    dt_pe = _to_datetime(pred_end, ref_d)
    dt_gs = _to_datetime(gt_start, ref_d)
    dt_ge = _to_datetime(gt_end, ref_d)

    # If both pred and gt have explicit date differences (e.g. across midnight)
    if dt_pe < dt_ps:
        dt_pe += timedelta(days=1)
    if dt_ge < dt_gs:
        dt_ge += timedelta(days=1)

    # If the anchors are aligned on the same timeline:
    # Check if we should align dates if only times were provided
    if isinstance(pred_start, (time, str)) and ("T" not in str(pred_start)):
        # Decomposed interval intersection in seconds of day
        pred_intervals = decompose_midnight_interval(pred_start, pred_end)
        gt_intervals = decompose_midnight_interval(gt_start, gt_end)

        # Total intersection duration
        intersection = 0.0
        for p_s, p_e in pred_intervals:
            for g_s, g_e in gt_intervals:
                overlap_s = max(p_s, g_s)
                overlap_e = min(p_e, g_e)
                if overlap_e > overlap_s:
                    intersection += (overlap_e - overlap_s)

        # Durations
        dur_p = sum(p_e - p_s for p_s, p_e in pred_intervals)
        dur_g = sum(g_e - g_s for g_s, g_e in gt_intervals)
        union = dur_p + dur_g - intersection

        if union <= 0.0:
            return 0.0
        return intersection / union

    # Continuous calculation using absolute timestamps
    int_start = max(dt_ps, dt_gs)
    int_end = min(dt_pe, dt_ge)
    intersection = max(0.0, (int_end - int_start).total_seconds())

    dur_pred = max(0.0, (dt_pe - dt_ps).total_seconds())
    dur_gt = max(0.0, (dt_ge - dt_gs).total_seconds())
    union = dur_pred + dur_gt - intersection

    if union <= 0.0:
        return 0.0
    return intersection / union
