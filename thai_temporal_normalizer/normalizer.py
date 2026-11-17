"""Main pipeline entrypoint for Thai Temporal Normalizer."""

from __future__ import annotations

import calendar
import re
import zoneinfo
from datetime import datetime, tzinfo
from datetime import timezone as dt_timezone
from typing import Any, Dict, List, Optional, Tuple

from thai_temporal_normalizer.models import TemporalEntity
from thai_temporal_normalizer.redos_guard import TimeoutGuard, check_input_length
from thai_temporal_normalizer.rules import (
    BANGKOK_TZ,
    BuddhistEraRuleEngine,
    ClockRuleEngine,
    CoarseRangeRuleEngine,
    NightRolloverRuleEngine,
    RelativeRuleEngine,
    normalize_thai_digits,
    resolve_be_year,
)

# Impossible calendar date detector (e.g. 30 ก.พ., 31 เม.ย.)
P_IMPOSSIBLE_DATE = re.compile(
    r"(?:3[01]\s*(?:ก\.พ\.|กุมภาพันธ์)|31\s*(?:เม\.ย\.|มิ\.ย\.|ก\.ย\.|พ\.ย\.|เมษายน|มิถุนายน|กันยายน|พฤศจิกายน))"
)

# Impossible clock time detector (e.g. 25:70 น., ตี 7, ตี 8, สิบสองทุ่ม)
P_IMPOSSIBLE_TIME = re.compile(
    r"(?:(?:2[4-9]|[3-9]\d)[:.][6-9]\d|ตี\s*(?:[6-9]|0|เจ็ด|แปด|เก้า)|สิบสองทุ่ม)"
)

# 29 Feb detector for leap-year validation
P_FEB_29 = re.compile(
    r"29\s*(?:ก\.พ\.|กุมภาพันธ์)(?:\s*(?:(?:พ\.ศ\.|ปี)?\s*['\"]?(\d{2,4}))|\b)|29[/.-]0?2[/.-](\d{2,4})"
)


def _has_invalid_feb29(text: str, current_be: int) -> bool:
    """Check if input text references 29 Feb on a non-leap year."""
    norm = normalize_thai_digits(text)
    for m in P_FEB_29.finditer(norm):
        yr_str = m.group(1) or m.group(2)
        if yr_str:
            yr_ce = resolve_be_year(int(yr_str), current_be)
        else:
            yr_ce = current_be - 543
        if not calendar.isleap(yr_ce):
            return True
    return False


def _resolve_timezone(tz_name: str, strict_mode: bool = False) -> tzinfo:
    """Resolve timezone string to tzinfo with graceful fallback."""
    try:
        return zoneinfo.ZoneInfo(tz_name)
    except Exception as exc:
        if tz_name in ("Asia/Bangkok", "Bangkok"):
            return BANGKOK_TZ
        if tz_name in ("UTC", "Etc/UTC"):
            return dt_timezone.utc
        if strict_mode:
            raise ValueError(f"Invalid timezone: {tz_name}") from exc
        return BANGKOK_TZ


# Module-level static engine singletons to eliminate instantiation overhead
_GLOBAL_CLOCK_ENGINE = ClockRuleEngine()
_GLOBAL_RELATIVE_ENGINE = RelativeRuleEngine()
_GLOBAL_BE_ENGINE = BuddhistEraRuleEngine()
_GLOBAL_NIGHT_ENGINE = NightRolloverRuleEngine(clock_engine=_GLOBAL_CLOCK_ENGINE)
_GLOBAL_COARSE_ENGINE = CoarseRangeRuleEngine(relative_engine=_GLOBAL_RELATIVE_ENGINE)


def parse_thai_temporal(
    text: str,
    reference_time: Optional[datetime] = None,
    timezone: str = "Asia/Bangkok",
    strict_mode: bool = False,
    custom_config: Optional[Dict[str, Any]] = None,
) -> List[TemporalEntity]:
    """Parse colloquial Thai temporal expressions into ISO-8601 standardized representations.

    Args:
        text: Input Thai text containing temporal expressions (max 120 chars).
        reference_time: Base reference datetime for deictic resolution (default: datetime.now(Asia/Bangkok)).
        timezone: Target timezone name (default: 'Asia/Bangkok').
        strict_mode: If True, raises ValueError on length violation, impossible dates/times,
                     or coarse expressions requiring clarification.
        custom_config: Optional mapping overriding coarse interval start/end times.

    Returns:
        List of parsed TemporalEntity instances sorted by start_char.

    Raises:
        ValueError: Under strict_mode for invalid lengths, impossible dates/times,
                    or ambiguous coarse expressions requiring clarification.
    """
    # 1. Early length check (ReDoS Guard)
    if not check_input_length(text, strict_mode=strict_mode):
        return []

    # 2. Setup target timezone and reference time
    target_tz = _resolve_timezone(timezone, strict_mode=strict_mode)
    if reference_time is None:
        ref_dt = datetime.now(target_tz)
    else:
        if reference_time.tzinfo is None:
            ref_dt = reference_time.replace(tzinfo=target_tz)
        else:
            ref_dt = reference_time.astimezone(target_tz)
    current_be = ref_dt.year + 543

    # 3. Guard against impossible dates & clock times (including 29 Feb non-leap years)
    if P_IMPOSSIBLE_DATE.search(text) or _has_invalid_feb29(text, current_be):
        if strict_mode:
            raise ValueError("Invalid calendar date detected in input text.")
        return []

    if P_IMPOSSIBLE_TIME.search(text):
        if strict_mode:
            raise ValueError("Malformed or impossible clock time detected in input text.")
        return []

    # 4. Engine selection (using module-level cached singletons)
    clock_engine = _GLOBAL_CLOCK_ENGINE
    relative_engine = _GLOBAL_RELATIVE_ENGINE
    be_engine = _GLOBAL_BE_ENGINE
    night_engine = _GLOBAL_NIGHT_ENGINE
    if custom_config is not None:
        coarse_engine = CoarseRangeRuleEngine(custom_config=custom_config, relative_engine=relative_engine)
    else:
        coarse_engine = _GLOBAL_COARSE_ENGINE

    # 5. Execute normalization engines under timeout guard
    with TimeoutGuard(timeout_sec=0.010) as guard:
        candidates: List[TemporalEntity] = []

        # (A) Coarse range engine
        coarse_entities = coarse_engine.parse_coarse_span(text, ref_time=ref_dt)
        guard.check()
        candidates.extend(coarse_entities)

        # (B) Night-time day-rollover engine
        night_entities = night_engine.parse_rollover_span(text, ref_time=ref_dt)
        guard.check()
        candidates.extend(night_entities)

        # (C) Buddhist Era & Calendar dates
        be_entities = be_engine.parse_be_span(text, ref_time=ref_dt, strict_mode=strict_mode)
        guard.check()
        candidates.extend(be_entities)

        # (D) Relative day/week expressions
        rel_spans = relative_engine.parse_relative_span(text, ref_time=ref_dt)
        guard.check()

        # (E) Clock time expressions
        clock_spans = clock_engine.parse_time_span(text)
        guard.check()

        # (F) Combine adjacent Relative Date + Clock Time
        combined_spans: List[Tuple[int, int]] = []
        for r_span, r_s, r_e, r_date, r_grain in rel_spans:
            adjacent_clock = None
            for c_span, c_s, c_e, c_time, c_grain in clock_spans:
                gap = text[r_e:c_s]
                if r_s < c_s and c_s >= r_e and gap.strip() in ("", "เวลา", "ตอน", "ช่วง"):
                    adjacent_clock = (c_span, c_s, c_e, c_time, c_grain)
                    break

            if adjacent_clock is not None:
                c_span, c_s, c_e, c_time, c_grain = adjacent_clock
                combined_text = text[r_s:c_e]
                combined_dt = datetime(
                    r_date.year,
                    r_date.month,
                    r_date.day,
                    c_time.hour,
                    c_time.minute,
                    c_time.second,
                    tzinfo=target_tz,
                )
                candidates.append(
                    TemporalEntity(
                        text_span=combined_text,
                        start_char=r_s,
                        end_char=c_e,
                        type="POINT",
                        grain=c_grain,
                        iso_value=combined_dt.isoformat(),
                        precision="exact",
                        confidence=1.0,
                    )
                )
                combined_spans.append((r_s, c_e))
            else:
                target_dt = datetime(
                    r_date.year,
                    r_date.month,
                    r_date.day,
                    ref_dt.hour,
                    ref_dt.minute,
                    ref_dt.second,
                    tzinfo=target_tz,
                )
                candidates.append(
                    TemporalEntity(
                        text_span=r_span,
                        start_char=r_s,
                        end_char=r_e,
                        type="POINT",
                        grain=r_grain,
                        iso_value=target_dt.isoformat(),
                        precision="exact",
                        confidence=1.0,
                    )
                )

        # (G) Standalone clock times
        for c_span, c_s, c_e, c_time, c_grain in clock_spans:
            if any(c_s >= cb_s and c_e <= cb_e for cb_s, cb_e in combined_spans):
                continue

            target_dt = datetime(
                ref_dt.year,
                ref_dt.month,
                ref_dt.day,
                c_time.hour,
                c_time.minute,
                c_time.second,
                tzinfo=target_tz,
            )
            candidates.append(
                TemporalEntity(
                    text_span=c_span,
                    start_char=c_s,
                    end_char=c_e,
                    type="POINT",
                    grain=c_grain,
                    iso_value=target_dt.isoformat(),
                    precision="exact",
                    confidence=1.0,
                )
            )

    # 5. Strict mode coarse range check on parsed entities
    if strict_mode:
        for ent in candidates:
            if ent.precision == "coarse":
                raise ValueError("Strict mode: coarse time expression requires clarification.")

    # 6. Deduplicate and resolve overlapping entity spans
    candidates.sort(key=lambda x: (x.start_char, -(x.end_char - x.start_char)))
    filtered: List[TemporalEntity] = []

    for cand in candidates:
        if any(cand.start_char >= chosen.start_char and cand.end_char <= chosen.end_char for chosen in filtered):
            continue
        enclosed_indices = [
            i for i, chosen in enumerate(filtered)
            if chosen.start_char >= cand.start_char and chosen.end_char <= cand.end_char
        ]
        if enclosed_indices:
            for i in reversed(enclosed_indices):
                filtered.pop(i)
        overlap = any(
            max(cand.start_char, chosen.start_char) < min(cand.end_char, chosen.end_char)
            for chosen in filtered
        )
        if not overlap:
            filtered.append(cand)

    filtered.sort(key=lambda x: x.start_char)
    return filtered
