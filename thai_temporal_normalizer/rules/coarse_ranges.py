"""Configurable coarse time ranges engine for colloquial Thai periods."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from thai_temporal_normalizer.models import TemporalEntity
from thai_temporal_normalizer.rules.constants import BANGKOK_TZ, THAI_WEEKDAY_MAP
from thai_temporal_normalizer.rules.relative import RelativeRuleEngine

DEFAULT_COARSE_TABLE: Dict[str, Tuple[str, str, float]] = {
    "เช้ามืด": ("04:00:00", "06:00:00", 0.85),
    "สายๆ": ("09:00:00", "11:30:00", 0.80),
    "บ่ายๆ": ("13:00:00", "15:30:00", 0.80),
    "บ่ายแก่ๆ": ("15:00:00", "17:00:00", 0.85),
    "เย็นๆ": ("16:30:00", "18:30:00", 0.80),
    "หัวค่ำ": ("18:30:00", "20:30:00", 0.85),
    "ค่ำๆ": ("19:00:00", "21:30:00", 0.80),
    "ดึกๆ": ("22:00:00", "02:00:00", 0.80),
}

# Regex to canonical key mapping (longest/most specific patterns first)
COARSE_PATTERNS: List[Tuple[str, str]] = [
    (r"(?:ช่วง|ตอน)?บ่ายแก่ๆ", "บ่ายแก่ๆ"),
    (r"(?:ช่วง|ตอน)?เช้ามืด", "เช้ามืด"),
    (r"(?:ช่วง|ตอน)?หัวค่ำ", "หัวค่ำ"),
    (r"(?:ช่วง|ตอน)?สายๆ|ช่วงสาย|ตอนสาย", "สายๆ"),
    (r"(?:ช่วง|ตอน)?บ่ายๆ|ช่วงบ่าย|ตอนบ่าย", "บ่ายๆ"),
    (r"(?:ช่วง|ตอน)?เย็นๆ|ช่วงเย็น|ตอนเย็น", "เย็นๆ"),
    (r"(?:ช่วง|ตอน)?ค่ำๆ|ช่วงค่ำ|ตอนค่ำ", "ค่ำๆ"),
    (r"(?:ช่วง|ตอน)?ดึกๆ|ช่วงดึก|ตอนดึก", "ดึกๆ"),
]


class CoarseRangeRuleEngine:
    """Parser for coarse Thai temporal ranges with custom configuration injection."""

    def __init__(
        self,
        custom_config: Optional[Dict[str, Any]] = None,
        relative_engine: Optional[RelativeRuleEngine] = None,
    ) -> None:
        self.config: Dict[str, Tuple[str, str, float]] = dict(DEFAULT_COARSE_TABLE)
        if custom_config:
            for k, val in custom_config.items():
                if isinstance(val, dict):
                    s_time = val.get("start", "00:00:00")
                    e_time = val.get("end", "00:00:00")
                elif isinstance(val, (tuple, list)) and len(val) >= 2:
                    s_time, e_time = val[0], val[1]
                else:
                    raise ValueError(
                        f"Invalid custom_config entry for '{k}': expected tuple/list of 2 strings or dict with 'start' and 'end', got {type(val).__name__}"
                    )
                default_conf = DEFAULT_COARSE_TABLE.get(k, ("00:00:00", "00:00:00", 0.80))[2]
                self.config[k] = (s_time, e_time, default_conf)

        self.relative_engine = relative_engine or RelativeRuleEngine()

        # Build compound regex for date + coarse period
        period_union = "|".join(f"(?:{p[0]})" for p in COARSE_PATTERNS)
        self.p_coarse_only = re.compile(rf"({period_union})")

        rel_prefixes = r"(?:พรุ่งนี้|เมื่อวานนี้|เมื่อวาน|วันนี้|คืนนี้|วัน(?:จันทร์|อังคาร|พุธ|พฤหัสบดี|พฤหัส|ศุกร์|เสาร์|อาทิตย์))"
        self.p_compound_coarse = re.compile(
            rf"({rel_prefixes})\s*({period_union})"
        )

    def _get_canonical_key(self, matched_period: str) -> Optional[str]:
        for pat, key in COARSE_PATTERNS:
            if re.fullmatch(pat, matched_period):
                return key
        return None

    def parse_coarse_span(
        self,
        text: str,
        ref_time: datetime,
    ) -> List[TemporalEntity]:
        """Extract all coarse range expressions (standalone or combined with relative date)."""
        entities: List[TemporalEntity] = []
        ref_date = ref_time.date()

        tz = ref_time.tzinfo or BANGKOK_TZ

        # 1. Compound relative + coarse range: e.g. "พรุ่งนี้สายๆ", "เมื่อวานหัวค่ำ"
        for m in self.p_compound_coarse.finditer(text):
            date_part = m.group(1)
            period_part = m.group(2)
            key = self._get_canonical_key(period_part)
            if not key or key not in self.config:
                continue

            if date_part == "พรุ่งนี้":
                target_date = ref_date + timedelta(days=1)
            elif date_part in ("เมื่อวาน", "เมื่อวานนี้"):
                target_date = ref_date - timedelta(days=1)
            elif date_part in ("วันนี้", "คืนนี้"):
                target_date = ref_date
            elif date_part.startswith("วัน"):
                wk_name = date_part.replace("วัน", "")
                target_wk = THAI_WEEKDAY_MAP.get(wk_name)
                if target_wk is not None:
                    days_ahead = (target_wk - ref_date.weekday()) % 7
                    if days_ahead == 0:
                        days_ahead = 7
                    target_date = ref_date + timedelta(days=days_ahead)
                else:
                    target_date = ref_date
            else:
                target_date = ref_date

            start_t_str, end_t_str, conf = self.config[key]
            s_h, s_m, s_s = map(int, start_t_str.split(":"))
            e_h, e_m, e_s = map(int, end_t_str.split(":"))
            start_dt = datetime(target_date.year, target_date.month, target_date.day, s_h, s_m, s_s, tzinfo=tz)

            end_date = target_date + timedelta(days=1) if (e_h, e_m, e_s) < (s_h, s_m, s_s) else target_date
            end_dt = datetime(end_date.year, end_date.month, end_date.day, e_h, e_m, e_s, tzinfo=tz)

            entities.append(
                TemporalEntity(
                    text_span=m.group(0),
                    start_char=m.start(),
                    end_char=m.end(),
                    type="RANGE",
                    range_start=start_dt.isoformat(),
                    range_end=end_dt.isoformat(),
                    precision="coarse",
                    confidence=conf,
                )
            )

        # 2. Standalone coarse range: e.g. "สายๆ", "ช่วงบ่าย", "หัวค่ำ", "ดึกๆ"
        for m in self.p_coarse_only.finditer(text):
            s, e = m.start(), m.end()
            if any(s >= existing.start_char and e <= existing.end_char for existing in entities):
                continue

            period_text = m.group(1)
            key = self._get_canonical_key(period_text)
            if not key or key not in self.config:
                continue

            start_t_str, end_t_str, conf = self.config[key]
            s_h, s_m, s_s = map(int, start_t_str.split(":"))
            e_h, e_m, e_s = map(int, end_t_str.split(":"))

            start_dt = datetime(ref_date.year, ref_date.month, ref_date.day, s_h, s_m, s_s, tzinfo=tz)
            end_date = ref_date + timedelta(days=1) if (e_h, e_m, e_s) < (s_h, s_m, s_s) else ref_date
            end_dt = datetime(end_date.year, end_date.month, end_date.day, e_h, e_m, e_s, tzinfo=tz)

            entities.append(
                TemporalEntity(
                    text_span=m.group(0),
                    start_char=s,
                    end_char=e,
                    type="RANGE",
                    range_start=start_dt.isoformat(),
                    range_end=end_dt.isoformat(),
                    precision="coarse",
                    confidence=conf,
                )
            )

        return entities
