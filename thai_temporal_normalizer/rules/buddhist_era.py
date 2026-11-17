"""Buddhist Era (BE) century pivot and calendar date normalization engine."""

from __future__ import annotations

import calendar
import re
from datetime import datetime
from typing import List

from thai_temporal_normalizer.models import TemporalEntity
from thai_temporal_normalizer.rules.constants import (
    BANGKOK_TZ,
    THAI_MONTHS_MAP,
    normalize_thai_digits,
)


def resolve_be_year(year_val: int, current_be_year: int) -> int:
    """Resolve BE year (2-digit short or 4-digit full) to CE year.

    Implements Dynamic Century Pivot Window:
        Pivot = (CurrentYear_BE % 100) + 10
        If YY <= Pivot => 2500 + YY else 2400 + YY
        Year_CE = Year_BE - 543
    """
    if year_val >= 1000:
        if year_val > 2400:
            return year_val - 543
        return year_val

    # 2-digit short BE year
    pivot = (current_be_year % 100) + 10
    if year_val <= pivot:
        be_year = 2500 + year_val
    else:
        be_year = 2400 + year_val

    return be_year - 543


class BuddhistEraRuleEngine:
    """Parser for Buddhist Era years, dates, quarters, and seasons."""

    def __init__(self) -> None:
        m_keys = sorted(THAI_MONTHS_MAP.keys(), key=len, reverse=True)
        m_pattern = "|".join(re.escape(k) for k in m_keys)
        self.p_thai_month_date = re.compile(
            rf"(\d{{1,2}})\s*({m_pattern})\s*(?:(?:พ\.ศ\.|ปี)?\s*['\"]?(\d{{2,4}}))"
        )

        self.p_slash_dash_date = re.compile(
            r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})"
        )

        self.p_seasonal = re.compile(
            r"(ต้นปี|กลางปี|ปลายปี|สิ้นปี)\s*['\"]?(\d{2,4})"
        )

        self.p_quarter = re.compile(
            r"ไตรมาส(?:\s*ที่)?\s*([1-4])\s*(?:ปี)?\s*['\"]?(\d{2,4})"
        )

        self.p_standalone_year = re.compile(
            r"(?:ปี\s*['\"]?|พ\.ศ\.\s*|ปี\s*พ\.ศ\.\s*)(\d{2,4})"
        )

    def parse_be_span(
        self,
        text: str,
        ref_time: datetime,
        strict_mode: bool = False,
    ) -> List[TemporalEntity]:
        """Extract all Buddhist Era and calendar date expressions from text."""
        entities: List[TemporalEntity] = []
        norm_text = normalize_thai_digits(text)
        current_be = (ref_time.year + 543)
        tz = ref_time.tzinfo or BANGKOK_TZ

        # 1. Full date with Thai month name
        for m in self.p_thai_month_date.finditer(norm_text):
            day_str = m.group(1)
            month_str = m.group(2)
            year_str = m.group(3)

            day = int(day_str)
            month = THAI_MONTHS_MAP.get(month_str)
            if not month:
                continue

            year_raw = int(year_str)
            year_ce = resolve_be_year(year_raw, current_be)

            max_days = calendar.monthrange(year_ce, month)[1]
            if not (1 <= day <= max_days):
                if strict_mode:
                    raise ValueError("Invalid calendar date detected in input text.")
                continue

            iso_val = datetime(year_ce, month, day, 0, 0, 0, tzinfo=tz).isoformat()
            entities.append(
                TemporalEntity(
                    text_span=text[m.start():m.end()],
                    start_char=m.start(),
                    end_char=m.end(),
                    type="POINT",
                    grain="day",
                    iso_value=iso_val,
                    precision="exact",
                    confidence=1.0,
                )
            )

        # 2. Slash/Dash numeric dates: "15/03/67", "15-03-67"
        for m in self.p_slash_dash_date.finditer(norm_text):
            day = int(m.group(1))
            month = int(m.group(2))
            year_raw = int(m.group(3))

            if not (1 <= month <= 12):
                if strict_mode:
                    raise ValueError("Invalid calendar date detected in input text.")
                continue

            year_ce = resolve_be_year(year_raw, current_be)
            max_days = calendar.monthrange(year_ce, month)[1]
            if not (1 <= day <= max_days):
                if strict_mode:
                    raise ValueError("Invalid calendar date detected in input text.")
                continue

            iso_val = datetime(year_ce, month, day, 0, 0, 0, tzinfo=tz).isoformat()
            entities.append(
                TemporalEntity(
                    text_span=text[m.start():m.end()],
                    start_char=m.start(),
                    end_char=m.end(),
                    type="POINT",
                    grain="day",
                    iso_value=iso_val,
                    precision="exact",
                    confidence=1.0,
                )
            )

        # 3. Seasonal short BE: "ต้นปี 68", "กลางปี 67", "ปลายปี 69", "สิ้นปี 67"
        for m in self.p_seasonal.finditer(norm_text):
            season_kw = m.group(1)
            year_raw = int(m.group(2))
            year_ce = resolve_be_year(year_raw, current_be)

            if season_kw == "ต้นปี":
                r_start = datetime(year_ce, 1, 1, 0, 0, 0, tzinfo=tz).isoformat()
                r_end = datetime(year_ce, 4, 30, 23, 59, 59, tzinfo=tz).isoformat()
                entities.append(
                    TemporalEntity(
                        text_span=text[m.start():m.end()],
                        start_char=m.start(),
                        end_char=m.end(),
                        type="RANGE",
                        grain="auto",
                        range_start=r_start,
                        range_end=r_end,
                        precision="coarse",
                        confidence=0.85,
                    )
                )
            elif season_kw == "กลางปี":
                r_start = datetime(year_ce, 5, 1, 0, 0, 0, tzinfo=tz).isoformat()
                r_end = datetime(year_ce, 8, 31, 23, 59, 59, tzinfo=tz).isoformat()
                entities.append(
                    TemporalEntity(
                        text_span=text[m.start():m.end()],
                        start_char=m.start(),
                        end_char=m.end(),
                        type="RANGE",
                        grain="auto",
                        range_start=r_start,
                        range_end=r_end,
                        precision="coarse",
                        confidence=0.85,
                    )
                )
            elif season_kw == "ปลายปี":
                r_start = datetime(year_ce, 9, 1, 0, 0, 0, tzinfo=tz).isoformat()
                r_end = datetime(year_ce, 12, 31, 23, 59, 59, tzinfo=tz).isoformat()
                entities.append(
                    TemporalEntity(
                        text_span=text[m.start():m.end()],
                        start_char=m.start(),
                        end_char=m.end(),
                        type="RANGE",
                        grain="auto",
                        range_start=r_start,
                        range_end=r_end,
                        precision="coarse",
                        confidence=0.85,
                    )
                )
            elif season_kw == "สิ้นปี":
                iso_val = datetime(year_ce, 12, 31, 0, 0, 0, tzinfo=tz).isoformat()
                entities.append(
                    TemporalEntity(
                        text_span=text[m.start():m.end()],
                        start_char=m.start(),
                        end_char=m.end(),
                        type="POINT",
                        grain="day",
                        iso_value=iso_val,
                        precision="exact",
                        confidence=1.0,
                    )
                )

        # 4. Quarter: "ไตรมาส 1 ปี 68"
        for m in self.p_quarter.finditer(norm_text):
            q_num = int(m.group(1))
            year_raw = int(m.group(2))
            year_ce = resolve_be_year(year_raw, current_be)

            q_ranges = {
                1: (datetime(year_ce, 1, 1, 0, 0, 0, tzinfo=tz).isoformat(), datetime(year_ce, 3, 31, 23, 59, 59, tzinfo=tz).isoformat()),
                2: (datetime(year_ce, 4, 1, 0, 0, 0, tzinfo=tz).isoformat(), datetime(year_ce, 6, 30, 23, 59, 59, tzinfo=tz).isoformat()),
                3: (datetime(year_ce, 7, 1, 0, 0, 0, tzinfo=tz).isoformat(), datetime(year_ce, 9, 30, 23, 59, 59, tzinfo=tz).isoformat()),
                4: (datetime(year_ce, 10, 1, 0, 0, 0, tzinfo=tz).isoformat(), datetime(year_ce, 12, 31, 23, 59, 59, tzinfo=tz).isoformat()),
            }
            r_start, r_end = q_ranges[q_num]
            entities.append(
                TemporalEntity(
                    text_span=text[m.start():m.end()],
                    start_char=m.start(),
                    end_char=m.end(),
                    type="RANGE",
                    grain="auto",
                    range_start=r_start,
                    range_end=r_end,
                    precision="exact",
                    confidence=1.0,
                )
            )

        # 5. Standalone year: "ปี 69", "ปี'69", "พ.ศ. 2569"
        for m in self.p_standalone_year.finditer(norm_text):
            s, e = m.start(), m.end()
            if any(s >= existing.start_char and e <= existing.end_char for existing in entities):
                continue

            year_raw = int(m.group(1))
            year_ce = resolve_be_year(year_raw, current_be)
            iso_val = datetime(year_ce, 1, 1, 0, 0, 0, tzinfo=tz).isoformat()
            entities.append(
                TemporalEntity(
                    text_span=text[s:e],
                    start_char=s,
                    end_char=e,
                    type="POINT",
                    grain="year",
                    iso_value=iso_val,
                    precision="exact",
                    confidence=1.0,
                )
            )

        entities.sort(key=lambda x: (x.start_char, -(x.end_char - x.start_char)))
        filtered: List[TemporalEntity] = []
        for ent in entities:
            if not any(ent.start_char >= f.start_char and ent.end_char <= f.end_char for f in filtered):
                filtered.append(ent)

        return filtered
