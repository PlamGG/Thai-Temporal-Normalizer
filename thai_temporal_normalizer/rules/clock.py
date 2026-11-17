"""Thai 6-Hour Clock and 24-Hour time normalization engine."""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

from thai_temporal_normalizer.rules.constants import parse_thai_number

_MINUTE_WORDS: Dict[str, int] = {
    "ครึ่ง": 30,
    "สิบห้า": 15,
    "ยี่สิบ": 20,
    "ยี่สิบห้า": 25,
    "สามสิบ": 30,
    "สามสิบห้า": 35,
    "สี่สิบ": 40,
    "สี่สิบห้า": 45,
    "ห้าสิบ": 50,
    "ห้าสิบห้า": 55,
}


def _extract_minute(mod_str: Optional[str]) -> Tuple[int, str]:
    """Extract minute offset and grain from modifier string."""
    if not mod_str:
        return 0, "hour"

    mod = mod_str.strip()
    if mod == "ตรง":
        return 0, "minute"
    if mod == "ครึ่ง":
        return 30, "minute"

    cleaned = re.sub(r"นาที$", "", mod).strip()
    if cleaned in _MINUTE_WORDS:
        return _MINUTE_WORDS[cleaned], "minute"

    num = parse_thai_number(cleaned)
    if num is not None and 0 <= num < 60:
        return num, "minute"

    return 0, "hour"


class ClockRuleEngine:
    """Parser for Thai 6-hour clock (ตี, โมง, บ่าย, ทุ่ม, เที่ยง) and 24h expressions."""

    def __init__(self) -> None:
        min_pat = r"(?:ครึ่ง|ตรง|สี่สิบห้า|สามสิบห้า|ยี่สิบห้า|ห้าสิบห้า|สิบห้า|ยี่สิบ|สามสิบ|สี่สิบ|ห้าสิบ|\d{1,2}\s*นาที|\d{1,2}|(?:[ก-๙]+)\s*นาที)"

        # 1. ตี (01:00-05:00)
        self.p_ti = re.compile(
            rf"ตี\s*([1-5]|หนึ่ง|สอง|สาม|สี่|ห้า)(?:\s*({min_pat}))?"
        )

        # 2. เที่ยง / เที่ยงคืน / หกทุ่ม
        self.p_noon_mid = re.compile(
            rf"(เที่ยงคืน|หกทุ่ม|เที่ยงวัน|เที่ยงตรง|เที่ยง)(?:\s*({min_pat}))?"
        )

        # 3. โมงเย็น (16:00-18:00) - check before morning so 'ห้าโมงเย็น' / 'ห้าโมง' evening is handled
        self.p_evening = re.compile(
            rf"(([4-6]|สี่|ห้า|หก)\s*โมง(?:\s*เย็น)?)(?:\s*({min_pat}))?"
        )

        # 4. โมงเช้า (07:00-11:00)
        self.p_morning = re.compile(
            rf"((?:สิบเอ็ด|สิบ|หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|[6-9]|10|11)?\s*โมง(?:\s*เช้า)?)(?:\s*({min_pat}))?"
        )

        # 5. บ่าย (13:00-15:00)
        self.p_afternoon = re.compile(
            rf"(บ่าย(?:\s*(?:โมง\s*)?([1-3]|หนึ่ง|สอง|สาม))?(?:\s*โมง)?)(?:\s*({min_pat}))?"
        )

        # 6. ทุ่ม (19:00-23:00)
        self.p_night = re.compile(
            rf"(([1-5]|หนึ่ง|สอง|สาม|สี่|ห้า)?\s*ทุ่ม(?:\s*(?:นึง|ตรง))?)(?:\s*({min_pat}))?"
        )

        # 7. Official 24h format
        self.p_official = re.compile(
            r"(\d{1,2})[:.](\d{2})\s*(?:น\.|น(?![ก-๙a-zA-Z]))"
        )
        self.p_official_colon = re.compile(
            r"(\d{1,2}):(\d{2})"
        )
        self.p_official_nalika = re.compile(
            r"(?:(\d{1,2})|([ก-๙]+))\s*นาฬิกา(?:\s*(?:(\d{1,2})|([ก-๙]+))\s*นาที)?"
        )

        # 8. Subtractive minute pattern
        self.p_subtractive = re.compile(
            r"อีก\s*([ก-๙0-9]+)\s*นาที\s*(.+)"
        )

    def parse_time_span(self, text: str) -> List[Tuple[str, int, int, time, str]]:
        """Parse all clock time instances in text."""
        results: List[Tuple[str, int, int, time, str]] = []

        # Check subtractive minute first
        m_sub = self.p_subtractive.search(text)
        if m_sub:
            sub_min_str = m_sub.group(1)
            target_str = m_sub.group(2)
            sub_min = parse_thai_number(sub_min_str)
            if sub_min is not None:
                inner_res = self.parse_time_span(target_str)
                if inner_res:
                    _, _, _, base_t, _ = inner_res[0]
                    dt_base = datetime(2026, 1, 1, base_t.hour, base_t.minute)
                    dt_new = dt_base - timedelta(minutes=sub_min)
                    results.append((
                        m_sub.group(0),
                        m_sub.start(),
                        m_sub.end(),
                        time(dt_new.hour, dt_new.minute),
                        "minute",
                    ))
                    return results

        # 1. ตี (01:00-05:00)
        for m in self.p_ti.finditer(text):
            h_str = m.group(1)
            mod_str = m.group(2)
            h_map = {"1": 1, "หนึ่ง": 1, "2": 2, "สอง": 2, "3": 3, "สาม": 3, "4": 4, "สี่": 4, "5": 5, "ห้า": 5}
            hour = h_map.get(h_str, 1)
            minute, grain = _extract_minute(mod_str)
            results.append((m.group(0), m.start(), m.end(), time(hour, minute), grain))

        # 2. เที่ยง / เที่ยงคืน / หกทุ่ม
        for m in self.p_noon_mid.finditer(text):
            kw = m.group(1)
            mod_str = m.group(2)
            if kw in ("เที่ยงคืน", "หกทุ่ม"):
                hour = 0
                minute, grain = _extract_minute(mod_str)
                if grain == "hour" and kw == "เที่ยงคืน":
                    grain = "minute"
            else:  # เที่ยง, เที่ยงวัน, เที่ยงตรง
                hour = 12
                if kw == "เที่ยงตรง":
                    minute, grain = 0, "minute"
                else:
                    minute, grain = _extract_minute(mod_str)
                    if kw == "เที่ยงวัน":
                        grain = "hour"
            results.append((m.group(0), m.start(), m.end(), time(hour, minute), grain))

        # 3. Evening (โมงเย็น: 16:00-18:00) and colloquial 4/5/6 โมง without 'เช้า'
        for m in self.p_evening.finditer(text):
            full_span = m.group(0)
            prefix_kw = m.group(1)
            mod_str = m.group(3)

            if "เช้า" in full_span:
                continue

            h_str = m.group(2)
            h_map = {"4": 16, "สี่": 16, "5": 17, "ห้า": 17, "6": 18, "หก": 18}
            hour = h_map.get(h_str, 16)
            minute, grain = _extract_minute(mod_str)
            results.append((full_span, m.start(), m.end(), time(hour, minute), grain))

        # 4. Morning (โมงเช้า: 07:00-11:00)
        for m in self.p_morning.finditer(text):
            full_span = m.group(0)
            prefix_kw = m.group(1)
            mod_str = m.group(2)

            if "เย็น" in text[m.start():m.end() + 10] or "บ่าย" in text[max(0, m.start() - 6):m.end()]:
                continue
            if any(prefix_kw.startswith(p) for p in ("สี่โมง", "ห้าโมง", "หกโมง", "4 โมง", "5 โมง", "6 โมง")) and "เช้า" not in prefix_kw:
                continue

            hour = 7
            if "สิบเอ็ด" in prefix_kw or "11" in prefix_kw:
                hour = 11
            elif "สิบ" in prefix_kw or "10" in prefix_kw:
                hour = 10
            elif "เก้า" in prefix_kw or "9" in prefix_kw:
                hour = 9
            elif "แปด" in prefix_kw or "8" in prefix_kw:
                hour = 8
            elif "เจ็ด" in prefix_kw or "7" in prefix_kw or prefix_kw == "โมง":
                hour = 7
            elif "หก" in prefix_kw or "6" in prefix_kw:
                hour = 6
            elif "ห้า" in prefix_kw or "5" in prefix_kw:
                hour = 11
            elif "สี่" in prefix_kw or "4" in prefix_kw:
                hour = 10
            elif "สาม" in prefix_kw:
                hour = 9
            elif "สอง" in prefix_kw:
                hour = 8
            elif "หนึ่ง" in prefix_kw:
                hour = 7

            minute, grain = _extract_minute(mod_str)
            results.append((full_span, m.start(), m.end(), time(hour, minute), grain))

        # 5. บ่าย (13:00-15:00)
        for m in self.p_afternoon.finditer(text):
            full_span = m.group(0)
            h_str = m.group(2)
            mod_str = m.group(3)
            h_map = {None: 13, "": 13, "1": 13, "หนึ่ง": 13, "2": 14, "สอง": 14, "3": 15, "สาม": 15}
            hour = h_map.get(h_str, 13)
            minute, grain = _extract_minute(mod_str)
            results.append((full_span, m.start(), m.end(), time(hour, minute), grain))

        # 6. ทุ่ม (19:00-23:00)
        for m in self.p_night.finditer(text):
            full_span = m.group(0)
            if "หกทุ่ม" in text[max(0, m.start() - 2):m.end()]:
                continue
            h_str = m.group(2)
            mod_str = m.group(3)
            h_map = {None: 19, "": 19, "1": 19, "หนึ่ง": 19, "2": 20, "สอง": 20, "3": 21, "สาม": 21, "4": 22, "สี่": 22, "5": 23, "ห้า": 23}
            hour = h_map.get(h_str, 19)
            if "นึง" in full_span:
                hour = 19
            minute, grain = _extract_minute(mod_str)
            results.append((full_span, m.start(), m.end(), time(hour, minute), grain))

        # 7. Official 24-hour formats
        for m in self.p_official.finditer(text):
            h = int(m.group(1))
            mn = int(m.group(2))
            if 0 <= h <= 24 and 0 <= mn < 60:
                h = 0 if h == 24 else h
                results.append((m.group(0), m.start(), m.end(), time(h, mn), "minute"))

        for m in self.p_official_nalika.finditer(text):
            h_raw = m.group(1) or m.group(2)
            mn_raw = m.group(3) or m.group(4)
            h = parse_thai_number(h_raw) if h_raw else None
            mn = parse_thai_number(mn_raw) if mn_raw else 0
            if h is not None and 0 <= h <= 24 and 0 <= mn < 60:
                h = 0 if h == 24 else h
                grain = "minute" if mn_raw else "hour"
                results.append((m.group(0), m.start(), m.end(), time(h, mn), grain))

        results.sort(key=lambda x: (x[1], -(x[2] - x[1])))
        filtered: List[Tuple[str, int, int, time, str]] = []
        for item in results:
            span, s, e, t, g = item
            if not any(s >= existing[1] and e <= existing[2] for existing in filtered):
                filtered.append(item)

        return filtered
