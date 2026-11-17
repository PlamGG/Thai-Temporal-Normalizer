"""Linguistic constants, numeral tables, and calendar maps for Thai language."""

from __future__ import annotations

from datetime import timedelta, timezone
from typing import Dict, Optional

BANGKOK_TZ = timezone(timedelta(hours=7))

THAI_DIGITS_MAP: Dict[str, str] = {
    "๐": "0",
    "๑": "1",
    "๒": "2",
    "๓": "3",
    "๔": "4",
    "๕": "5",
    "๖": "6",
    "๗": "7",
    "๘": "8",
    "๙": "9",
}

THAI_NUM_WORDS: Dict[str, int] = {
    "ศูนย์": 0,
    "หนึ่ง": 1,
    "เอ็ด": 1,
    "นึง": 1,
    "สอง": 2,
    "สาม": 3,
    "สี่": 4,
    "ห้า": 5,
    "หก": 6,
    "เจ็ด": 7,
    "แปด": 8,
    "เก้า": 9,
    "สิบ": 10,
    "ยี่สิบ": 20,
    "สามสิบ": 30,
    "สี่สิบ": 40,
    "ห้าสิบ": 50,
}

THAI_MONTHS_MAP: Dict[str, int] = {
    # Full names
    "มกราคม": 1,
    "กุมภาพันธ์": 2,
    "มีนาคม": 3,
    "เมษายน": 4,
    "พฤษภาคม": 5,
    "มิถุนายน": 6,
    "กรกฎาคม": 7,
    "สิงหาคม": 8,
    "กันยายน": 9,
    "ตุลาคม": 10,
    "พฤศจิกายน": 11,
    "ธันวาคม": 12,
    # Abbreviated names
    "ม.ค.": 1,
    "ม.ค": 1,
    "ก.พ.": 2,
    "ก.พ": 2,
    "มี.ค.": 3,
    "มี.ค": 3,
    "เม.ย.": 4,
    "เม.ย": 4,
    "พ.ค.": 5,
    "พ.ค": 5,
    "มิ.ย.": 6,
    "มิ.ย": 6,
    "ก.ค.": 7,
    "ก.ค": 7,
    "ส.ค.": 8,
    "ส.ค": 8,
    "ก.ย.": 9,
    "ก.ย": 9,
    "ต.ค.": 10,
    "ต.ค": 10,
    "พ.ย.": 11,
    "พ.ย": 11,
    "ธ.ค.": 12,
    "ธ.ค": 12,
}

# Python weekday: Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6
THAI_WEEKDAY_MAP: Dict[str, int] = {
    "จันทร์": 0,
    "วันจันทร์": 0,
    "อังคาร": 1,
    "วันอังคาร": 1,
    "พุธ": 2,
    "วันพุธ": 2,
    "พฤหัส": 3,
    "พฤหัสบดี": 3,
    "วันพฤหัส": 3,
    "วันพฤหัสบดี": 3,
    "ศุกร์": 4,
    "วันศุกร์": 4,
    "เสาร์": 5,
    "วันเสาร์": 5,
    "อาทิตย์": 6,
    "วันอาทิตย์": 6,
}


def normalize_thai_digits(text: str) -> str:
    """Convert Thai numerals (๐-๙) to Arabic digits (0-9)."""
    return "".join(THAI_DIGITS_MAP.get(ch, ch) for ch in text)


def parse_thai_number(num_str: str) -> Optional[int]:
    """Parse Thai number string into an integer (supports Arabic, Thai digits, and Thai words).

    Examples:
        '15' -> 15
        'สิบห้า' -> 15
        'สาม' -> 3
        'สิบเอ็ด' -> 11
        'สี่สิบห้า' -> 45
    """
    cleaned = normalize_thai_digits(num_str.strip())
    if cleaned.isdigit():
        return int(cleaned)

    # Word-based parsing
    if cleaned in THAI_NUM_WORDS:
        return THAI_NUM_WORDS[cleaned]

    # Two-digit compound numbers like สี่สิบห้า, สิบเอ็ด, ยี่สิบสอง
    val = 0
    s = cleaned
    if s.startswith("ยี่สิบ"):
        val += 20
        s = s[5:]
    elif s.startswith("สามสิบ"):
        val += 30
        s = s[6:]
    elif s.startswith("สี่สิบ"):
        val += 40
        s = s[6:]
    elif s.startswith("ห้าสิบ"):
        val += 50
        s = s[6:]
    elif s.startswith("สิบ"):
        val += 10
        s = s[3:]

    if s:
        unit = THAI_NUM_WORDS.get(s)
        if unit is not None:
            val += unit
        else:
            return None

    return val if val > 0 or cleaned == "ศูนย์" else None
