"""Generate and validate tests/data/baseline_150.jsonl for Phase 0 Baseline Benchmark.

This script constructs 150 stratified Thai temporal items across 5 categories
(30 items each) with exact character spans, ground truth ISO representations,
and reference metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"
OUTPUT_FILE = DATA_DIR / "baseline_150.jsonl"
REF_TIME = "2026-11-08T12:00:00+07:00"
TIMEZONE = "Asia/Bangkok"


def build_raw_items() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Category 1: relative_day_week (30 items)
    # Reference: Sunday 2026-11-08T12:00:00+07:00
    # -------------------------------------------------------------------------
    cat1_defs = [
        # 1-7: Single deictic days
        ("วันนี้", "วันนี้", "POINT", "day", "2026-11-08T12:00:00+07:00", None, None, "exact", "deictic_today", False, "วันนี้ (วันอาทิตย์ที่ 8 พ.ย. 2026)"),
        ("พรุ่งนี้", "พรุ่งนี้", "POINT", "day", "2026-11-09T12:00:00+07:00", None, None, "exact", "deictic_tomorrow", False, "พรุ่งนี้ (+1 วัน = จันทร์ 9 พ.ย.)"),
        ("มะรืนนี้", "มะรืนนี้", "POINT", "day", "2026-11-10T12:00:00+07:00", None, None, "exact", "deictic_day_after_tomorrow", False, "มะรืนนี้ (+2 วัน = อังคาร 10 พ.ย.)"),
        ("เมื่อวาน", "เมื่อวาน", "POINT", "day", "2026-11-07T12:00:00+07:00", None, None, "exact", "deictic_yesterday", False, "เมื่อวาน (-1 วัน = เสาร์ 7 พ.ย.)"),
        ("เมื่อวานนี้", "เมื่อวานนี้", "POINT", "day", "2026-11-07T12:00:00+07:00", None, None, "exact", "deictic_yesterday_full", False, "เมื่อวานนี้ (-1 วัน = เสาร์ 7 พ.ย.)"),
        ("วานซืน", "วานซืน", "POINT", "day", "2026-11-06T12:00:00+07:00", None, None, "exact", "deictic_day_before_yesterday", False, "วานซืน (-2 วัน = ศุกร์ 6 พ.ย.)"),
        ("เมื่อวานซืน", "เมื่อวานซืน", "POINT", "day", "2026-11-06T12:00:00+07:00", None, None, "exact", "deictic_day_before_yesterday_full", False, "เมื่อวานซืน (-2 วัน = ศุกร์ 6 พ.ย.)"),
        # 8-17: Weekday relative shifts
        ("พุธหน้า", "พุธหน้า", "POINT", "day", "2026-11-11T12:00:00+07:00", None, None, "exact", "relative_weekday_next", False, "พุธหน้า (พุธที่จะถึง = 11 พ.ย.)"),
        ("อังคารที่แล้ว", "อังคารที่แล้ว", "POINT", "day", "2026-11-03T12:00:00+07:00", None, None, "exact", "relative_weekday_last", False, "อังคารที่แล้ว (อังคารก่อนหน้า = 3 พ.ย.)"),
        ("ศุกร์นี้", "ศุกร์นี้", "POINT", "day", "2026-11-13T12:00:00+07:00", None, None, "exact", "relative_weekday_this", False, "ศุกร์นี้ (ศุกร์ในสัปดาห์ปัจจุบัน = 13 พ.ย.)"),
        ("จันทร์ที่จะถึงนี้", "จันทร์ที่จะถึงนี้", "POINT", "day", "2026-11-09T12:00:00+07:00", None, None, "exact", "relative_weekday_upcoming", False, "จันทร์ที่จะถึงนี้ (9 พ.ย.)"),
        ("พฤหัสหน้า", "พฤหัสหน้า", "POINT", "day", "2026-11-12T12:00:00+07:00", None, None, "exact", "relative_weekday_next", False, "พฤหัสหน้า (12 พ.ย.)"),
        ("เสาร์ที่ผ่านมา", "เสาร์ที่ผ่านมา", "POINT", "day", "2026-11-07T12:00:00+07:00", None, None, "exact", "relative_weekday_past", False, "เสาร์ที่ผ่านมา (7 พ.ย.)"),
        ("อาทิตย์หน้า", "อาทิตย์หน้า", "POINT", "day", "2026-11-15T12:00:00+07:00", None, None, "exact", "relative_weekday_next_sunday", False, "อาทิตย์หน้า (15 พ.ย.)"),
        ("เสาร์หน้า", "เสาร์หน้า", "POINT", "day", "2026-11-14T12:00:00+07:00", None, None, "exact", "relative_weekday_next_saturday", False, "เสาร์หน้า (14 พ.ย.)"),
        ("พฤหัสที่แล้ว", "พฤหัสที่แล้ว", "POINT", "day", "2026-11-05T12:00:00+07:00", None, None, "exact", "relative_weekday_last_thursday", False, "พฤหัสที่แล้ว (5 พ.ย.)"),
        ("ศุกร์ที่แล้ว", "ศุกร์ที่แล้ว", "POINT", "day", "2026-11-06T12:00:00+07:00", None, None, "exact", "relative_weekday_last_friday", False, "ศุกร์ที่แล้ว (6 พ.ย.)"),
        # 18-25: Numeric offsets and week shifts
        ("อีก 3 วัน", "อีก 3 วัน", "POINT", "day", "2026-11-11T12:00:00+07:00", None, None, "exact", "numeric_offset_plus", False, "อีก 3 วัน (+3 วัน = 11 พ.ย.)"),
        ("อีก 5 วัน", "อีก 5 วัน", "POINT", "day", "2026-11-13T12:00:00+07:00", None, None, "exact", "numeric_offset_plus", False, "อีก 5 วัน (+5 วัน = 13 พ.ย.)"),
        ("2 วันก่อน", "2 วันก่อน", "POINT", "day", "2026-11-06T12:00:00+07:00", None, None, "exact", "numeric_offset_minus", False, "2 วันก่อน (-2 วัน = 6 พ.ย.)"),
        ("3 วันที่แล้ว", "3 วันที่แล้ว", "POINT", "day", "2026-11-05T12:00:00+07:00", None, None, "exact", "numeric_offset_minus", False, "3 วันที่แล้ว (-3 วัน = 5 พ.ย.)"),
        ("อีก 1 สัปดาห์", "อีก 1 สัปดาห์", "POINT", "day", "2026-11-15T12:00:00+07:00", None, None, "exact", "week_offset_plus", False, "อีก 1 สัปดาห์ (+7 วัน = 15 พ.ย.)"),
        ("อีกสองวัน", "อีกสองวัน", "POINT", "day", "2026-11-10T12:00:00+07:00", None, None, "exact", "thai_num_offset_plus", False, "อีกสองวัน (+2 วัน = 10 พ.ย.)"),
        ("วันจันทร์หน้า", "วันจันทร์หน้า", "POINT", "day", "2026-11-16T12:00:00+07:00", None, None, "exact", "relative_weekday_next_full", False, "วันจันทร์หน้า (จันทร์สัปดาห์ถัดไป = 16 พ.ย.)"),
        ("วันอังคารหน้า", "วันอังคารหน้า", "POINT", "day", "2026-11-17T12:00:00+07:00", None, None, "exact", "relative_weekday_next_full", False, "วันอังคารหน้า (อังคารสัปดาห์ถัดไป = 17 พ.ย.)"),
        # 26-30: Natural sentence embedded
        ("เจอกันพรุ่งนี้นะ", "พรุ่งนี้", "POINT", "day", "2026-11-09T12:00:00+07:00", None, None, "exact", "embedded_sentence", False, "พรุ่งนี้ในประโยคสนทนา"),
        ("ส่งรายงานภายในวันมะรืนนี้", "มะรืนนี้", "POINT", "day", "2026-11-10T12:00:00+07:00", None, None, "exact", "embedded_sentence", False, "มะรืนนี้ในประโยคคำสั่ง"),
        ("นัดคุยงานวันพุธหน้า", "วันพุธหน้า", "POINT", "day", "2026-11-11T12:00:00+07:00", None, None, "exact", "embedded_sentence", False, "วันพุธหน้าในประโยคนัดหมาย"),
        ("เคลียร์งานเสร็จตั้งแต่วานซืนแล้ว", "วานซืน", "POINT", "day", "2026-11-06T12:00:00+07:00", None, None, "exact", "embedded_sentence", False, "วานซืนในประโยคบอกเล่า"),
        ("มีประชุมด่วนวันศุกร์นี้", "วันศุกร์นี้", "POINT", "day", "2026-11-13T12:00:00+07:00", None, None, "exact", "embedded_sentence", False, "วันศุกร์นี้ในประโยคแจ้งเตือน"),
    ]

    for raw, span, typ, grain, iso, rstart, rend, prec, ling, mid, notes in cat1_defs:
        items.append({
            "category": "relative_day_week",
            "raw_input": raw,
            "text_span": span,
            "type": typ,
            "grain": grain,
            "iso_value": iso,
            "range_start": rstart,
            "range_end": rend,
            "precision": prec,
            "confidence": 1.0,
            "linguistic_type": ling,
            "has_midnight_crossing": mid,
            "notes": notes,
        })

    # -------------------------------------------------------------------------
    # Category 2: thai_6hour_clock (30 items)
    # Reference date: 2026-11-08
    # -------------------------------------------------------------------------
    cat2_defs = [
        # Morning (โมงเช้า: 07:00-11:00)
        ("โมงเช้า", "โมงเช้า", "POINT", "hour", "2026-11-08T07:00:00+07:00", None, None, "exact", "clock_morning_1st_hour", False, "โมงเช้า = 07:00"),
        ("แปดโมงเช้า", "แปดโมงเช้า", "POINT", "hour", "2026-11-08T08:00:00+07:00", None, None, "exact", "clock_morning", False, "แปดโมงเช้า = 08:00"),
        ("เก้าโมง", "เก้าโมง", "POINT", "hour", "2026-11-08T09:00:00+07:00", None, None, "exact", "clock_morning", False, "เก้าโมง = 09:00"),
        ("สิบโมงเช้า", "สิบโมงเช้า", "POINT", "hour", "2026-11-08T10:00:00+07:00", None, None, "exact", "clock_morning", False, "สิบโมงเช้า = 10:00"),
        ("สิบเอ็ดโมง", "สิบเอ็ดโมง", "POINT", "hour", "2026-11-08T11:00:00+07:00", None, None, "exact", "clock_morning", False, "สิบเอ็ดโมง = 11:00"),
        ("เก้าโมงครึ่ง", "เก้าโมงครึ่ง", "POINT", "minute", "2026-11-08T09:30:00+07:00", None, None, "exact", "clock_morning_half", False, "เก้าโมงครึ่ง = 09:30"),
        ("สิบโมงสี่สิบห้า", "สิบโมงสี่สิบห้า", "POINT", "minute", "2026-11-08T10:45:00+07:00", None, None, "exact", "clock_morning_fraction", False, "สิบโมงสี่สิบห้า = 10:45"),
        # Noon (เที่ยง: 12:00)
        ("เที่ยงตรง", "เที่ยงตรง", "POINT", "minute", "2026-11-08T12:00:00+07:00", None, None, "exact", "clock_noon_exact", False, "เที่ยงตรง = 12:00"),
        ("เที่ยงวัน", "เที่ยงวัน", "POINT", "hour", "2026-11-08T12:00:00+07:00", None, None, "exact", "clock_noon", False, "เที่ยงวัน = 12:00"),
        ("เที่ยงครึ่ง", "เที่ยงครึ่ง", "POINT", "minute", "2026-11-08T12:30:00+07:00", None, None, "exact", "clock_noon_half", False, "เที่ยงครึ่ง = 12:30"),
        # Afternoon (บ่าย: 13:00-15:00)
        ("บ่ายโมง", "บ่ายโมง", "POINT", "hour", "2026-11-08T13:00:00+07:00", None, None, "exact", "clock_afternoon", False, "บ่ายโมง = 13:00"),
        ("บ่ายสอง", "บ่ายสอง", "POINT", "hour", "2026-11-08T14:00:00+07:00", None, None, "exact", "clock_afternoon_colloquial", False, "บ่ายสอง = 14:00 ละคำว่าโมง"),
        ("บ่ายสาม", "บ่ายสาม", "POINT", "hour", "2026-11-08T15:00:00+07:00", None, None, "exact", "clock_afternoon_colloquial", False, "บ่ายสาม = 15:00 ละคำว่าโมง"),
        ("บ่ายโมงครึ่ง", "บ่ายโมงครึ่ง", "POINT", "minute", "2026-11-08T13:30:00+07:00", None, None, "exact", "clock_afternoon_half", False, "บ่ายโมงครึ่ง = 13:30"),
        ("บ่ายสองสี่สิบห้า", "บ่ายสองสี่สิบห้า", "POINT", "minute", "2026-11-08T14:45:00+07:00", None, None, "exact", "clock_afternoon_fraction", False, "บ่ายสองสี่สิบห้า = 14:45"),
        ("บ่ายสามสิบห้า", "บ่ายสามสิบห้า", "POINT", "minute", "2026-11-08T15:15:00+07:00", None, None, "exact", "clock_afternoon_fraction", False, "บ่ายสามสิบห้า = 15:15"),
        # Evening (โมงเย็น: 16:00-18:00)
        ("สี่โมงเย็น", "สี่โมงเย็น", "POINT", "hour", "2026-11-08T16:00:00+07:00", None, None, "exact", "clock_evening", False, "สี่โมงเย็น = 16:00"),
        ("ห้าโมงเย็น", "ห้าโมงเย็น", "POINT", "hour", "2026-11-08T17:00:00+07:00", None, None, "exact", "clock_evening", False, "ห้าโมงเย็น = 17:00"),
        ("หกโมงเย็น", "หกโมงเย็น", "POINT", "hour", "2026-11-08T18:00:00+07:00", None, None, "exact", "clock_evening", False, "หกโมงเย็น = 18:00"),
        ("ห้าโมงครึ่ง", "ห้าโมงครึ่ง", "POINT", "minute", "2026-11-08T17:30:00+07:00", None, None, "exact", "clock_evening_half", False, "ห้าโมงครึ่ง = 17:30"),
        # Night (ทุ่ม: 19:00-23:00)
        ("หนึ่งทุ่ม", "หนึ่งทุ่ม", "POINT", "hour", "2026-11-08T19:00:00+07:00", None, None, "exact", "clock_night", False, "หนึ่งทุ่ม = 19:00"),
        ("ทุ่มนึง", "ทุ่มนึง", "POINT", "hour", "2026-11-08T19:00:00+07:00", None, None, "exact", "clock_night_colloquial", False, "ทุ่มนึง = 19:00 ภาษาพูด"),
        ("สองทุ่ม", "สองทุ่ม", "POINT", "hour", "2026-11-08T20:00:00+07:00", None, None, "exact", "clock_night", False, "สองทุ่ม = 20:00"),
        ("สามทุ่ม", "สามทุ่ม", "POINT", "hour", "2026-11-08T21:00:00+07:00", None, None, "exact", "clock_night", False, "สามทุ่ม = 21:00"),
        ("สี่ทุ่ม", "สี่ทุ่ม", "POINT", "hour", "2026-11-08T22:00:00+07:00", None, None, "exact", "clock_night", False, "สี่ทุ่ม = 22:00"),
        ("ห้าทุ่ม", "ห้าทุ่ม", "POINT", "hour", "2026-11-08T23:00:00+07:00", None, None, "exact", "clock_night", False, "ห้าทุ่ม = 23:00"),
        ("ทุ่มครึ่ง", "ทุ่มครึ่ง", "POINT", "minute", "2026-11-08T19:30:00+07:00", None, None, "exact", "clock_night_half", False, "ทุ่มครึ่ง = 19:30"),
        ("สี่ทุ่มครึ่ง", "สี่ทุ่มครึ่ง", "POINT", "minute", "2026-11-08T22:30:00+07:00", None, None, "exact", "clock_night_half", False, "สี่ทุ่มครึ่ง = 22:30"),
        # Midnight & Early morning (00:00, ตี: 01:00-05:00)
        ("เที่ยงคืน", "เที่ยงคืน", "POINT", "minute", "2026-11-08T00:00:00+07:00", None, None, "exact", "clock_midnight", False, "เที่ยงคืน = 00:00"),
        ("ตีสาม", "ตีสาม", "POINT", "hour", "2026-11-08T03:00:00+07:00", None, None, "exact", "clock_early_morning", False, "ตีสาม = 03:00"),
    ]

    for raw, span, typ, grain, iso, rstart, rend, prec, ling, mid, notes in cat2_defs:
        items.append({
            "category": "thai_6hour_clock",
            "raw_input": raw,
            "text_span": span,
            "type": typ,
            "grain": grain,
            "iso_value": iso,
            "range_start": rstart,
            "range_end": rend,
            "precision": prec,
            "confidence": 1.0,
            "linguistic_type": ling,
            "has_midnight_crossing": mid,
            "notes": notes,
        })

    # -------------------------------------------------------------------------
    # Category 3: coarse_time_periods (30 items)
    # PRD §6: Coarse Time Table definitions
    # -------------------------------------------------------------------------
    cat3_defs = [
        # เช้ามืด: 04:00 - 06:00
        ("เช้ามืด", "เช้ามืด", "RANGE", None, None, "2026-11-08T04:00:00+07:00", "2026-11-08T06:00:00+07:00", "coarse", "coarse_early_morning", False, "เช้ามืด 04:00-06:00"),
        ("ตอนเช้ามืด", "ตอนเช้ามืด", "RANGE", None, None, "2026-11-08T04:00:00+07:00", "2026-11-08T06:00:00+07:00", "coarse", "coarse_early_morning", False, "ตอนเช้ามืด 04:00-06:00"),
        ("ช่วงเช้ามืด", "ช่วงเช้ามืด", "RANGE", None, None, "2026-11-08T04:00:00+07:00", "2026-11-08T06:00:00+07:00", "coarse", "coarse_early_morning", False, "ช่วงเช้ามืด 04:00-06:00"),
        # สายๆ / ช่วงสาย: 09:00 - 11:30
        ("สายๆ", "สายๆ", "RANGE", None, None, "2026-11-08T09:00:00+07:00", "2026-11-08T11:30:00+07:00", "coarse", "coarse_late_morning", False, "สายๆ 09:00-11:30"),
        ("ช่วงสาย", "ช่วงสาย", "RANGE", None, None, "2026-11-08T09:00:00+07:00", "2026-11-08T11:30:00+07:00", "coarse", "coarse_late_morning", False, "ช่วงสาย 09:00-11:30"),
        ("ตอนสายๆ", "ตอนสายๆ", "RANGE", None, None, "2026-11-08T09:00:00+07:00", "2026-11-08T11:30:00+07:00", "coarse", "coarse_late_morning", False, "ตอนสายๆ 09:00-11:30"),
        # บ่ายๆ / ช่วงบ่าย: 13:00 - 15:30
        ("บ่ายๆ", "บ่ายๆ", "RANGE", None, None, "2026-11-08T13:00:00+07:00", "2026-11-08T15:30:00+07:00", "coarse", "coarse_afternoon", False, "บ่ายๆ 13:00-15:30"),
        ("ช่วงบ่าย", "ช่วงบ่าย", "RANGE", None, None, "2026-11-08T13:00:00+07:00", "2026-11-08T15:30:00+07:00", "coarse", "coarse_afternoon", False, "ช่วงบ่าย 13:00-15:30"),
        ("ตอนบ่ายๆ", "ตอนบ่ายๆ", "RANGE", None, None, "2026-11-08T13:00:00+07:00", "2026-11-08T15:30:00+07:00", "coarse", "coarse_afternoon", False, "ตอนบ่ายๆ 13:00-15:30"),
        # บ่ายแก่ๆ: 15:00 - 17:00
        ("บ่ายแก่ๆ", "บ่ายแก่ๆ", "RANGE", None, None, "2026-11-08T15:00:00+07:00", "2026-11-08T17:00:00+07:00", "coarse", "coarse_late_afternoon", False, "บ่ายแก่ๆ 15:00-17:00"),
        ("ช่วงบ่ายแก่ๆ", "ช่วงบ่ายแก่ๆ", "RANGE", None, None, "2026-11-08T15:00:00+07:00", "2026-11-08T17:00:00+07:00", "coarse", "coarse_late_afternoon", False, "ช่วงบ่ายแก่ๆ 15:00-17:00"),
        ("ตอนบ่ายแก่ๆ", "ตอนบ่ายแก่ๆ", "RANGE", None, None, "2026-11-08T15:00:00+07:00", "2026-11-08T17:00:00+07:00", "coarse", "coarse_late_afternoon", False, "ตอนบ่ายแก่ๆ 15:00-17:00"),
        # เย็นๆ / ช่วงเย็น: 16:30 - 18:30
        ("เย็นๆ", "เย็นๆ", "RANGE", None, None, "2026-11-08T16:30:00+07:00", "2026-11-08T18:30:00+07:00", "coarse", "coarse_evening", False, "เย็นๆ 16:30-18:30"),
        ("ช่วงเย็น", "ช่วงเย็น", "RANGE", None, None, "2026-11-08T16:30:00+07:00", "2026-11-08T18:30:00+07:00", "coarse", "coarse_evening", False, "ช่วงเย็น 16:30-18:30"),
        ("ตอนเย็นๆ", "ตอนเย็นๆ", "RANGE", None, None, "2026-11-08T16:30:00+07:00", "2026-11-08T18:30:00+07:00", "coarse", "coarse_evening", False, "ตอนเย็นๆ 16:30-18:30"),
        # หัวค่ำ: 18:30 - 20:30
        ("หัวค่ำ", "หัวค่ำ", "RANGE", None, None, "2026-11-08T18:30:00+07:00", "2026-11-08T20:30:00+07:00", "coarse", "coarse_early_night", False, "หัวค่ำ 18:30-20:30"),
        ("ช่วงหัวค่ำ", "ช่วงหัวค่ำ", "RANGE", None, None, "2026-11-08T18:30:00+07:00", "2026-11-08T20:30:00+07:00", "coarse", "coarse_early_night", False, "ช่วงหัวค่ำ 18:30-20:30"),
        ("ตอนหัวค่ำ", "ตอนหัวค่ำ", "RANGE", None, None, "2026-11-08T18:30:00+07:00", "2026-11-08T20:30:00+07:00", "coarse", "coarse_early_night", False, "ตอนหัวค่ำ 18:30-20:30"),
        # ค่ำๆ / ช่วงค่ำ: 19:00 - 21:30
        ("ค่ำๆ", "ค่ำๆ", "RANGE", None, None, "2026-11-08T19:00:00+07:00", "2026-11-08T21:30:00+07:00", "coarse", "coarse_night", False, "ค่ำๆ 19:00-21:30"),
        ("ช่วงค่ำ", "ช่วงค่ำ", "RANGE", None, None, "2026-11-08T19:00:00+07:00", "2026-11-08T21:30:00+07:00", "coarse", "coarse_night", False, "ช่วงค่ำ 19:00-21:30"),
        # ดึกๆ / ตอนดึก (Midnight Crossing): 22:00 - 02:00
        ("ดึกๆ", "ดึกๆ", "RANGE", None, None, "2026-11-08T22:00:00+07:00", "2026-11-09T02:00:00+07:00", "coarse", "coarse_late_night", True, "ดึกๆ 22:00-02:00 ข้ามเที่ยงคืน"),
        ("ตอนดึก", "ตอนดึก", "RANGE", None, None, "2026-11-08T22:00:00+07:00", "2026-11-09T02:00:00+07:00", "coarse", "coarse_late_night", True, "ตอนดึก 22:00-02:00 ข้ามเที่ยงคืน"),
        ("ช่วงดึก", "ช่วงดึก", "RANGE", None, None, "2026-11-08T22:00:00+07:00", "2026-11-09T02:00:00+07:00", "coarse", "coarse_late_night", True, "ช่วงดึก 22:00-02:00 ข้ามเที่ยงคืน"),
        # Compound combinations
        ("พรุ่งนี้สายๆ", "พรุ่งนี้สายๆ", "RANGE", None, None, "2026-11-09T09:00:00+07:00", "2026-11-09T11:30:00+07:00", "coarse", "compound_relative_coarse", False, "พรุ่งนี้สายๆ (9 พ.ย. 09:00-11:30)"),
        ("พรุ่งนี้บ่ายๆ", "พรุ่งนี้บ่ายๆ", "RANGE", None, None, "2026-11-09T13:00:00+07:00", "2026-11-09T15:30:00+07:00", "coarse", "compound_relative_coarse", False, "พรุ่งนี้บ่ายๆ (9 พ.ย. 13:00-15:30)"),
        ("พรุ่งนี้ตอนเย็นๆ", "พรุ่งนี้ตอนเย็นๆ", "RANGE", None, None, "2026-11-09T16:30:00+07:00", "2026-11-09T18:30:00+07:00", "coarse", "compound_relative_coarse", False, "พรุ่งนี้ตอนเย็นๆ (9 พ.ย. 16:30-18:30)"),
        ("เมื่อวานตอนบ่ายแก่ๆ", "เมื่อวานตอนบ่ายแก่ๆ", "RANGE", None, None, "2026-11-07T15:00:00+07:00", "2026-11-07T17:00:00+07:00", "coarse", "compound_relative_coarse", False, "เมื่อวานตอนบ่ายแก่ๆ (7 พ.ย. 15:00-17:00)"),
        ("เมื่อวานหัวค่ำ", "เมื่อวานหัวค่ำ", "RANGE", None, None, "2026-11-07T18:30:00+07:00", "2026-11-07T20:30:00+07:00", "coarse", "compound_relative_coarse", False, "เมื่อวานหัวค่ำ (7 พ.ย. 18:30-20:30)"),
        ("คืนนี้ดึกๆ", "คืนนี้ดึกๆ", "RANGE", None, None, "2026-11-08T22:00:00+07:00", "2026-11-09T02:00:00+07:00", "coarse", "compound_relative_coarse", True, "คืนนี้ดึกๆ (8 พ.ย. 22:00 ถึง 9 พ.ย. 02:00)"),
        ("วันศุกร์ตอนหัวค่ำ", "วันศุกร์ตอนหัวค่ำ", "RANGE", None, None, "2026-11-13T18:30:00+07:00", "2026-11-13T20:30:00+07:00", "coarse", "compound_relative_coarse", False, "วันศุกร์ตอนหัวค่ำ (13 พ.ย. 18:30-20:30)"),
    ]

    for raw, span, typ, grain, iso, rstart, rend, prec, ling, mid, notes in cat3_defs:
        items.append({
            "category": "coarse_time_periods",
            "raw_input": raw,
            "text_span": span,
            "type": typ,
            "grain": grain,
            "iso_value": iso,
            "range_start": rstart,
            "range_end": rend,
            "precision": prec,
            "confidence": 0.85 if "แก่" in raw or "เช้ามืด" in raw or "หัวค่ำ" in raw else 0.80,
            "linguistic_type": ling,
            "has_midnight_crossing": mid,
            "notes": notes,
        })

    # -------------------------------------------------------------------------
    # Category 4: short_buddhist_era (30 items)
    # Ref: 2026 CE (2569 BE). Dynamic Pivot Window YY <= 79 -> 2500s BE
    # -------------------------------------------------------------------------
    cat4_defs = [
        # Short year tokens
        ("ปี 69", "ปี 69", "POINT", "year", "2026-01-01T00:00:00+07:00", None, None, "exact", "short_be_year", False, "ปี 69 = BE 2569 = CE 2026"),
        ("ปี'69", "ปี'69", "POINT", "year", "2026-01-01T00:00:00+07:00", None, None, "exact", "short_be_year_apostrophe", False, "ปี'69 = BE 2569 = CE 2026"),
        ("ปี 67", "ปี 67", "POINT", "year", "2024-01-01T00:00:00+07:00", None, None, "exact", "short_be_year", False, "ปี 67 = BE 2567 = CE 2024"),
        ("ปี 68", "ปี 68", "POINT", "year", "2025-01-01T00:00:00+07:00", None, None, "exact", "short_be_year", False, "ปี 68 = BE 2568 = CE 2025"),
        ("ปี 70", "ปี 70", "POINT", "year", "2027-01-01T00:00:00+07:00", None, None, "exact", "short_be_year", False, "ปี 70 = BE 2570 = CE 2027"),
        ("ปี 66", "ปี 66", "POINT", "year", "2023-01-01T00:00:00+07:00", None, None, "exact", "short_be_year", False, "ปี 66 = BE 2566 = CE 2023"),
        ("ปี 65", "ปี 65", "POINT", "year", "2022-01-01T00:00:00+07:00", None, None, "exact", "short_be_year", False, "ปี 65 = BE 2565 = CE 2022"),
        ("ปี 72", "ปี 72", "POINT", "year", "2029-01-01T00:00:00+07:00", None, None, "exact", "short_be_year", False, "ปี 72 = BE 2572 = CE 2029"),
        # Full date with short BE (DD Month ShortBE)
        ("15 มี.ค. 67", "15 มี.ค. 67", "POINT", "day", "2024-03-15T00:00:00+07:00", None, None, "exact", "short_be_full_date", False, "15 มี.ค. 67 = 2024-03-15"),
        ("1 ม.ค. 69", "1 ม.ค. 69", "POINT", "day", "2026-01-01T00:00:00+07:00", None, None, "exact", "short_be_full_date", False, "1 ม.ค. 69 = 2026-01-01"),
        ("31 ธ.ค. 68", "31 ธ.ค. 68", "POINT", "day", "2025-12-31T00:00:00+07:00", None, None, "exact", "short_be_full_date", False, "31 ธ.ค. 68 = 2025-12-31"),
        ("28 ก.พ. 67", "28 ก.พ. 67", "POINT", "day", "2024-02-28T00:00:00+07:00", None, None, "exact", "short_be_full_date", False, "28 ก.พ. 67 = 2024-02-28"),
        ("12 ส.ค. 66", "12 ส.ค. 66", "POINT", "day", "2023-08-12T00:00:00+07:00", None, None, "exact", "short_be_full_date", False, "12 ส.ค. 66 = 2023-08-12"),
        ("5 ธ.ค. 65", "5 ธ.ค. 65", "POINT", "day", "2022-12-05T00:00:00+07:00", None, None, "exact", "short_be_full_date", False, "5 ธ.ค. 65 = 2022-12-05"),
        ("13 เม.ย. 68", "13 เม.ย. 68", "POINT", "day", "2025-04-13T00:00:00+07:00", None, None, "exact", "short_be_full_date", False, "13 เม.ย. 68 = 2025-04-13"),
        ("23 ต.ค. 67", "23 ต.ค. 67", "POINT", "day", "2024-10-23T00:00:00+07:00", None, None, "exact", "short_be_full_date", False, "23 ต.ค. 67 = 2024-10-23"),
        # Slash / Dash date with short BE
        ("15/03/67", "15/03/67", "POINT", "day", "2024-03-15T00:00:00+07:00", None, None, "exact", "short_be_slash_date", False, "15/03/67 = 2024-03-15"),
        ("01/01/69", "01/01/69", "POINT", "day", "2026-01-01T00:00:00+07:00", None, None, "exact", "short_be_slash_date", False, "01/01/69 = 2026-01-01"),
        ("31/12/68", "31/12/68", "POINT", "day", "2025-12-31T00:00:00+07:00", None, None, "exact", "short_be_slash_date", False, "31/12/68 = 2025-12-31"),
        ("15-03-67", "15-03-67", "POINT", "day", "2024-03-15T00:00:00+07:00", None, None, "exact", "short_be_dash_date", False, "15-03-67 = 2024-03-15"),
        ("05/12/66", "05/12/66", "POINT", "day", "2023-12-05T00:00:00+07:00", None, None, "exact", "short_be_slash_date", False, "05/12/66 = 2023-12-05"),
        ("20 ก.ค. 67", "20 ก.ค. 67", "POINT", "day", "2024-07-20T00:00:00+07:00", None, None, "exact", "short_be_full_date", False, "20 ก.ค. 67 = 2024-07-20"),
        ("30 ก.ย. 68", "30 ก.ย. 68", "POINT", "day", "2025-09-30T00:00:00+07:00", None, None, "exact", "short_be_full_date", False, "30 ก.ย. 68 = 2025-09-30"),
        ("10 พ.ย. 69", "10 พ.ย. 69", "POINT", "day", "2026-11-10T00:00:00+07:00", None, None, "exact", "short_be_full_date", False, "10 พ.ย. 69 = 2026-11-10"),
        ("14 ก.พ. 67", "14 ก.พ. 67", "POINT", "day", "2024-02-14T00:00:00+07:00", None, None, "exact", "short_be_full_date", False, "14 ก.พ. 67 = 2024-02-14"),
        # Vague seasonal short BE
        ("ต้นปี 68", "ต้นปี 68", "RANGE", None, None, "2025-01-01T00:00:00+07:00", "2025-04-30T23:59:59+07:00", "coarse", "short_be_seasonal", False, "ต้นปี 68 = Jan-Apr 2025"),
        ("กลางปี 67", "กลางปี 67", "RANGE", None, None, "2024-05-01T00:00:00+07:00", "2024-08-31T23:59:59+07:00", "coarse", "short_be_seasonal", False, "กลางปี 67 = May-Aug 2024"),
        ("ปลายปี 69", "ปลายปี 69", "RANGE", None, None, "2026-09-01T00:00:00+07:00", "2026-12-31T23:59:59+07:00", "coarse", "short_be_seasonal", False, "ปลายปี 69 = Sep-Dec 2026"),
        ("ไตรมาส 1 ปี 68", "ไตรมาส 1 ปี 68", "RANGE", None, None, "2025-01-01T00:00:00+07:00", "2025-03-31T23:59:59+07:00", "exact", "short_be_quarter", False, "ไตรมาส 1 ปี 68 = Q1 2025"),
        ("สิ้นปี 67", "สิ้นปี 67", "POINT", "day", "2024-12-31T00:00:00+07:00", None, None, "exact", "short_be_end_year", False, "สิ้นปี 67 = 31 ธ.ค. 2024"),
    ]

    for raw, span, typ, grain, iso, rstart, rend, prec, ling, mid, notes in cat4_defs:
        items.append({
            "category": "short_buddhist_era",
            "raw_input": raw,
            "text_span": span,
            "type": typ,
            "grain": grain,
            "iso_value": iso,
            "range_start": rstart,
            "range_end": rend,
            "precision": prec,
            "confidence": 1.0 if prec == "exact" else 0.85,
            "linguistic_type": ling,
            "has_midnight_crossing": mid,
            "notes": notes,
        })

    # -------------------------------------------------------------------------
    # Category 5: night_date_rollover (30 items)
    # Target anchor relative to ref: 2026-11-08 (Sunday)
    # Positive: Night marker + 00:00-05:59 -> [TargetDay] + 1 Day
    # Negative controls: >= 18:00 or morning anchor -> no rollover
    # -------------------------------------------------------------------------
    cat5_defs = [
        # Positive rollover cases (20 items)
        ("คืนวันศุกร์ ตีสอง", "คืนวันศุกร์ ตีสอง", "POINT", "minute", "2026-11-14T02:00:00+07:00", None, None, "exact", "rollover_positive", True, "คืนวันศุกร์ (13 พ.ย.) + ตีสอง -> เสาร์ 14 พ.ย. 02:00"),
        ("ดึกวันจันทร์ ตีหนึ่งครึ่ง", "ดึกวันจันทร์ ตีหนึ่งครึ่ง", "POINT", "minute", "2026-11-10T01:30:00+07:00", None, None, "exact", "rollover_positive", True, "ดึกวันจันทร์ (9 พ.ย.) + ตีหนึ่งครึ่ง -> อังคาร 10 พ.ย. 01:30"),
        ("คืนนี้ ตีสาม", "คืนนี้ ตีสาม", "POINT", "hour", "2026-11-09T03:00:00+07:00", None, None, "exact", "rollover_positive", True, "คืนนี้ (8 พ.ย.) + ตีสาม -> จันทร์ 9 พ.ย. 03:00"),
        ("เมื่อคืน ตีสอง", "เมื่อคืน ตีสอง", "POINT", "hour", "2026-11-08T02:00:00+07:00", None, None, "exact", "rollover_positive", True, "เมื่อคืน (7 พ.ย.) + ตีสอง -> อาทิตย์ 8 พ.ย. 02:00"),
        ("คืนวันเสาร์ ตีสี่", "คืนวันเสาร์ ตีสี่", "POINT", "hour", "2026-11-15T04:00:00+07:00", None, None, "exact", "rollover_positive", True, "คืนวันเสาร์ (14 พ.ย.) + ตีสี่ -> อาทิตย์ 15 พ.ย. 04:00"),
        ("คืนวันพุธ เที่ยงคืนครึ่ง", "คืนวันพุธ เที่ยงคืนครึ่ง", "POINT", "minute", "2026-11-12T00:30:00+07:00", None, None, "exact", "rollover_positive", True, "คืนวันพุธ (11 พ.ย.) + เที่ยงคืนครึ่ง -> พฤหัส 12 พ.ย. 00:30"),
        ("คืนวันพฤหัส ตีหนึ่ง", "คืนวันพฤหัส ตีหนึ่ง", "POINT", "hour", "2026-11-13T01:00:00+07:00", None, None, "exact", "rollover_positive", True, "คืนวันพฤหัส (12 พ.ย.) + ตีหนึ่ง -> ศุกร์ 13 พ.ย. 01:00"),
        ("คืนวันอังคาร ตีสองสิบห้า", "คืนวันอังคาร ตีสองสิบห้า", "POINT", "minute", "2026-11-11T02:15:00+07:00", None, None, "exact", "rollover_positive", True, "คืนวันอังคาร (10 พ.ย.) + ตีสองสิบห้า -> พุธ 11 พ.ย. 02:15"),
        ("ดึกคืนวันศุกร์ ตีสาม", "ดึกคืนวันศุกร์ ตีสาม", "POINT", "hour", "2026-11-14T03:00:00+07:00", None, None, "exact", "rollover_positive", True, "ดึกคืนวันศุกร์ (13 พ.ย.) + ตีสาม -> เสาร์ 14 พ.ย. 03:00"),
        ("เมื่อคืน ตีสามครึ่ง", "เมื่อคืน ตีสามครึ่ง", "POINT", "minute", "2026-11-08T03:30:00+07:00", None, None, "exact", "rollover_positive", True, "เมื่อคืน (7 พ.ย.) + ตีสามครึ่ง -> อาทิตย์ 8 พ.ย. 03:30"),
        ("คืนนี้ ตีห้า", "คืนนี้ ตีห้า", "POINT", "hour", "2026-11-09T05:00:00+07:00", None, None, "exact", "rollover_positive", True, "คืนนี้ (8 พ.ย.) + ตีห้า -> จันทร์ 9 พ.ย. 05:00"),
        ("คืนวันอาทิตย์ ตีสี่", "คืนวันอาทิตย์ ตีสี่", "POINT", "hour", "2026-11-09T04:00:00+07:00", None, None, "exact", "rollover_positive", True, "คืนวันอาทิตย์ (8 พ.ย.) + ตีสี่ -> จันทร์ 9 พ.ย. 04:00"),
        ("คืนวันจันทร์ เที่ยงคืน", "คืนวันจันทร์ เที่ยงคืน", "POINT", "minute", "2026-11-10T00:00:00+07:00", None, None, "exact", "rollover_positive", True, "คืนวันจันทร์ (9 พ.ย.) + เที่ยงคืน -> อังคาร 10 พ.ย. 00:00"),
        ("คืนวันเสาร์ ตีสอง", "คืนวันเสาร์ ตีสอง", "POINT", "hour", "2026-11-15T02:00:00+07:00", None, None, "exact", "rollover_positive", True, "คืนวันเสาร์ (14 พ.ย.) + ตีสอง -> อาทิตย์ 15 พ.ย. 02:00"),
        ("ดึกคืนนี้ ตีหนึ่ง", "ดึกคืนนี้ ตีหนึ่ง", "POINT", "hour", "2026-11-09T01:00:00+07:00", None, None, "exact", "rollover_positive", True, "ดึกคืนนี้ (8 พ.ย.) + ตีหนึ่ง -> จันทร์ 9 พ.ย. 01:00"),
        ("คืนวันพฤหัส ตีสี่ครึ่ง", "คืนวันพฤหัส ตีสี่ครึ่ง", "POINT", "minute", "2026-11-13T04:30:00+07:00", None, None, "exact", "rollover_positive", True, "คืนวันพฤหัส (12 พ.ย.) + ตีสี่ครึ่ง -> ศุกร์ 13 พ.ย. 04:30"),
        ("ดึกวันพุธ ตีสอง", "ดึกวันพุธ ตีสอง", "POINT", "hour", "2026-11-12T02:00:00+07:00", None, None, "exact", "rollover_positive", True, "ดึกวันพุธ (11 พ.ย.) + ตีสอง -> พฤหัส 12 พ.ย. 02:00"),
        ("เมื่อคืน ตีหนึ่ง", "เมื่อคืน ตีหนึ่ง", "POINT", "hour", "2026-11-08T01:00:00+07:00", None, None, "exact", "rollover_positive", True, "เมื่อคืน (7 พ.ย.) + ตีหนึ่ง -> อาทิตย์ 8 พ.ย. 01:00"),
        ("คืนนี้ ตีสอง", "คืนนี้ ตีสอง", "POINT", "hour", "2026-11-09T02:00:00+07:00", None, None, "exact", "rollover_positive", True, "คืนนี้ (8 พ.ย.) + ตีสอง -> จันทร์ 9 พ.ย. 02:00"),
        ("คืนวันอังคาร ตีสาม", "คืนวันอังคาร ตีสาม", "POINT", "hour", "2026-11-11T03:00:00+07:00", None, None, "exact", "rollover_positive", True, "คืนวันอังคาร (10 พ.ย.) + ตีสาม -> พุธ 11 พ.ย. 03:00"),
        # Negative control cases (10 items: time >= 18:00 or morning anchor)
        ("คืนวันศุกร์ สี่ทุ่ม", "คืนวันศุกร์ สี่ทุ่ม", "POINT", "hour", "2026-11-13T22:00:00+07:00", None, None, "exact", "rollover_negative_control", False, "คืนวันศุกร์ สี่ทุ่ม (13 พ.ย. 22:00) ไม่ข้ามวัน"),
        ("คืนวันศุกร์ สามทุ่ม", "คืนวันศุกร์ สามทุ่ม", "POINT", "hour", "2026-11-13T21:00:00+07:00", None, None, "exact", "rollover_negative_control", False, "คืนวันศุกร์ สามทุ่ม (13 พ.ย. 21:00) ไม่ข้ามวัน"),
        ("ดึกวันจันทร์ ห้าทุ่ม", "ดึกวันจันทร์ ห้าทุ่ม", "POINT", "hour", "2026-11-09T23:00:00+07:00", None, None, "exact", "rollover_negative_control", False, "ดึกวันจันทร์ ห้าทุ่ม (9 พ.ย. 23:00) ไม่ข้ามวัน"),
        ("คืนนี้ สองทุ่มครึ่ง", "คืนนี้ สองทุ่มครึ่ง", "POINT", "minute", "2026-11-08T20:30:00+07:00", None, None, "exact", "rollover_negative_control", False, "คืนนี้ สองทุ่มครึ่ง (8 พ.ย. 20:30) ไม่ข้ามวัน"),
        ("เมื่อคืน สี่ทุ่ม", "เมื่อคืน สี่ทุ่ม", "POINT", "hour", "2026-11-07T22:00:00+07:00", None, None, "exact", "rollover_negative_control", False, "เมื่อคืน สี่ทุ่ม (7 พ.ย. 22:00) ไม่ข้ามวัน"),
        ("คืนวันเสาร์ หนึ่งทุ่ม", "คืนวันเสาร์ หนึ่งทุ่ม", "POINT", "hour", "2026-11-14T19:00:00+07:00", None, None, "exact", "rollover_negative_control", False, "คืนวันเสาร์ หนึ่งทุ่ม (14 พ.ย. 19:00) ไม่ข้ามวัน"),
        ("คืนวันพุธ ห้าทุ่มครึ่ง", "คืนวันพุธ ห้าทุ่มครึ่ง", "POINT", "minute", "2026-11-11T23:30:00+07:00", None, None, "exact", "rollover_negative_control", False, "คืนวันพุธ ห้าทุ่มครึ่ง (11 พ.ย. 23:30) ไม่ข้ามวัน"),
        ("เช้าวันเสาร์ ตีสอง", "เช้าวันเสาร์ ตีสอง", "POINT", "hour", "2026-11-14T02:00:00+07:00", None, None, "exact", "rollover_negative_control", False, "เช้าวันเสาร์ ตีสอง (14 พ.ย. 02:00) anchor เช้า ไม่ข้ามวัน"),
        ("เช้าวันอาทิตย์ ตีสาม", "เช้าวันอาทิตย์ ตีสาม", "POINT", "hour", "2026-11-08T03:00:00+07:00", None, None, "exact", "rollover_negative_control", False, "เช้าวันอาทิตย์ ตีสาม (8 พ.ย. 03:00) anchor เช้า ไม่ข้ามวัน"),
        ("วันศุกร์ ตีสอง", "วันศุกร์ ตีสอง", "POINT", "hour", "2026-11-13T02:00:00+07:00", None, None, "exact", "rollover_negative_control", False, "วันศุกร์ ตีสอง (13 พ.ย. 02:00) ไม่มี night prefix"),
    ]

    for raw, span, typ, grain, iso, rstart, rend, prec, ling, mid, notes in cat5_defs:
        items.append({
            "category": "night_date_rollover",
            "raw_input": raw,
            "text_span": span,
            "type": typ,
            "grain": grain,
            "iso_value": iso,
            "range_start": rstart,
            "range_end": rend,
            "precision": prec,
            "confidence": 1.0,
            "linguistic_type": ling,
            "has_midnight_crossing": mid,
            "notes": notes,
        })

    return items


def main() -> None:
    raw_defs = build_raw_items()
    assert len(raw_defs) == 150, f"Expected 150 items, got {len(raw_defs)}"

    # Validate category counts
    from collections import Counter
    counts = Counter(d["category"] for d in raw_defs)
    print("Category distribution:", dict(counts))
    for cat, count in counts.items():
        assert count == 30, f"Category {cat} has {count} items, expected 30"

    # Assemble JSONL records
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    for idx, d in enumerate(raw_defs, start=1):
        raw_text = d["raw_input"]
        span = d["text_span"]
        start_char = raw_text.find(span)
        assert start_char != -1, f"Span '{span}' not found in raw_input '{raw_text}'"
        end_char = start_char + len(span)
        assert raw_text[start_char:end_char] == span, f"Span mismatch at index {idx}"

        record = {
            "id": idx,
            "raw_input": raw_text,
            "ground_truth": {
                "text_span": span,
                "start_char": start_char,
                "end_char": end_char,
                "type": d["type"],
                "grain": d["grain"],
                "iso_value": d["iso_value"],
                "range_start": d["range_start"],
                "range_end": d["range_end"],
                "precision": d["precision"],
                "confidence": d["confidence"],
            },
            "category": d["category"],
            "metadata": {
                "reference_time": REF_TIME,
                "timezone": TIMEZONE,
                "linguistic_type": d["linguistic_type"],
                "has_midnight_crossing": d["has_midnight_crossing"],
            },
            "notes": d["notes"],
        }
        records.append(record)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Successfully generated and verified {len(records)} items at {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
