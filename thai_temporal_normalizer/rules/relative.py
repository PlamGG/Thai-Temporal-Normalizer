"""Relative day, week, and weekday normalization engine."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import List, Tuple

from thai_temporal_normalizer.models import WeekdayAnchorPolicy
from thai_temporal_normalizer.rules.constants import (
    THAI_WEEKDAY_MAP,
    parse_thai_number,
)


class RelativeRuleEngine:
    """Parser for relative temporal expressions (วันนี้, พรุ่งนี้, พุธหน้า, อีก N วัน, etc.)."""

    def __init__(self, policy: WeekdayAnchorPolicy = WeekdayAnchorPolicy.NEXT_OCCURRENCE) -> None:
        self.policy = policy

        # 1. Single deictic days (ordered by length to match longest first)
        self.p_single_days = re.compile(
            r"(เมื่อวานซืน|เมื่อวานนี้|เมื่อวาน|วานซืน|มะรืนนี้|พรุ่งนี้|วันนี้|(?<![ก-๙])มรน(?![ก-๙])|(?<![ก-๙])พน(?![ก-๙]))"
        )

        # 2. Numeric day offset: อีก N วัน, อีกสองวัน, N วันก่อน, N วันที่แล้ว
        self.p_numeric_day_plus = re.compile(
            r"อีก\s*([ก-๙0-9]+)\s*วัน"
        )
        self.p_numeric_day_minus = re.compile(
            r"([ก-๙0-9]+)\s*วัน(?:ก่อน|ที่แล้ว)"
        )

        # 3. Numeric week offset: อีก N สัปดาห์ / อาทิตย์, N สัปดาห์ที่แล้ว
        self.p_numeric_week_plus = re.compile(
            r"อีก\s*([ก-๙0-9]+)\s*(?:สัปดาห์|อาทิตย์)"
        )
        self.p_numeric_week_minus = re.compile(
            r"([ก-๙0-9]+)\s*(?:สัปดาห์|อาทิตย์)(?:ก่อน|ที่แล้ว)"
        )

        # 4. Weekday expressions
        # e.g. วันพุธหน้า, พุธหน้า, อังคารที่แล้ว, ศุกร์นี้, จันทร์ที่จะถึงนี้, เสาร์ที่ผ่านมา, วันจันทร์หน้า
        wk_names = r"(?:วัน)?(?:จันทร์|อังคาร|พุธ|พฤหัสบดี|พฤหัส|ศุกร์|เสาร์|อาทิตย์)"
        self.p_weekday = re.compile(
            rf"({wk_names})\s*(ที่จะถึงนี้|ที่จะถึง|ที่ผ่านมา|ที่แล้ว|หน้า|นี้)"
        )

    def parse_relative_span(
        self,
        text: str,
        ref_time: datetime,
    ) -> List[Tuple[str, int, int, date, str]]:
        """Parse all relative day/week instances in text.

        Returns list of tuples: (matched_text, start_char, end_char, target_date, grain)
        """
        results: List[Tuple[str, int, int, date, str]] = []
        ref_date = ref_time.date()
        ref_weekday = ref_date.weekday()  # Monday=0, Sunday=6

        # 1. Single deictic days
        for m in self.p_single_days.finditer(text):
            word = m.group(1)
            offset = 0
            if word in ("เมื่อวานซืน", "วานซืน"):
                offset = -2
            elif word in ("เมื่อวาน", "เมื่อวานนี้"):
                offset = -1
            elif word == "วันนี้":
                offset = 0
            elif word in ("พรุ่งนี้", "พน"):
                offset = 1
            elif word in ("มะรืนนี้", "มรน"):
                offset = 2

            target_date = ref_date + timedelta(days=offset)
            results.append((m.group(0), m.start(), m.end(), target_date, "day"))

        # 2. Numeric day offsets
        for m in self.p_numeric_day_plus.finditer(text):
            num = parse_thai_number(m.group(1))
            if num is not None and num > 0:
                results.append((
                    m.group(0),
                    m.start(),
                    m.end(),
                    ref_date + timedelta(days=num),
                    "day",
                ))

        for m in self.p_numeric_day_minus.finditer(text):
            num = parse_thai_number(m.group(1))
            if num is not None and num > 0:
                results.append((
                    m.group(0),
                    m.start(),
                    m.end(),
                    ref_date - timedelta(days=num),
                    "day",
                ))

        # 3. Numeric week offsets
        for m in self.p_numeric_week_plus.finditer(text):
            num = parse_thai_number(m.group(1))
            if num is not None and num > 0:
                results.append((
                    m.group(0),
                    m.start(),
                    m.end(),
                    ref_date + timedelta(weeks=num),
                    "day",
                ))

        for m in self.p_numeric_week_minus.finditer(text):
            num = parse_thai_number(m.group(1))
            if num is not None and num > 0:
                results.append((
                    m.group(0),
                    m.start(),
                    m.end(),
                    ref_date - timedelta(weeks=num),
                    "day",
                ))

        # 4. Weekday expressions
        for m in self.p_weekday.finditer(text):
            span_text = m.group(0)
            wk_raw = m.group(1).replace("วัน", "")
            mod = m.group(2)
            target_wk = THAI_WEEKDAY_MAP.get(wk_raw)
            if target_wk is None:
                continue

            # Days ahead from ref_date (1 to 7)
            days_ahead = (target_wk - ref_weekday) % 7
            if days_ahead == 0:
                days_ahead = 7

            if mod in ("ที่แล้ว", "ที่ผ่านมา"):
                days_back = (ref_weekday - target_wk) % 7
                if days_back == 0:
                    days_back = 7
                target_date = ref_date - timedelta(days=days_back)
            elif mod in ("ที่จะถึงนี้", "ที่จะถึง"):
                target_date = ref_date + timedelta(days=days_ahead)
            elif mod == "นี้":
                target_date = ref_date + timedelta(days=days_ahead)
            elif mod == "หน้า":
                if self.policy == WeekdayAnchorPolicy.NEXT_WEEK:
                    target_date = ref_date + timedelta(days=days_ahead + 7)
                else:  # NEXT_OCCURRENCE
                    # If explicitly prefixed with 'วัน' and within 2 days (tomorrow/day-after),
                    # colloquial convention refers to the subsequent week
                    if span_text.startswith("วัน") and days_ahead <= 2:
                        target_date = ref_date + timedelta(days=days_ahead + 7)
                    else:
                        target_date = ref_date + timedelta(days=days_ahead)
            else:
                target_date = ref_date + timedelta(days=days_ahead)

            results.append((span_text, m.start(), m.end(), target_date, "day"))

        # Deduplicate overlapping matches (prefer longer match)
        results.sort(key=lambda x: (x[1], -(x[2] - x[1])))
        filtered: List[Tuple[str, int, int, date, str]] = []
        for item in results:
            span, s, e, dt, g = item
            if not any(s >= existing[1] and e <= existing[2] for existing in filtered):
                filtered.append(item)

        return filtered
