"""Synthetic dataset generator for Thai-Temporal-Normalizer.

Generates synthetic Thai temporal datasets across two primary modes:
1. `release_10k` (Default): 10,000-row release for Hugging Face (v1.1)
   - 7,000 train, 1,000 dev, 1,000 test_id, 1,000 test_ood
   - Stratified proportions: ~70% standard, ~15% bounds/durations/recurrence,
     ~10% negative samples (ground_truth: null), ~5% chat colloquialisms
   - SemanticTimeFilter: diurnal window filtering preventing unrealistic pairings
   - 135 templates across short, medium, and long styles (<= 120 char total SLA)
   - AlgebraicSpanCalculator: 100% exact character offset guarantees
   - CrossSplitDeduplicator: strictly 0 duplicate raw_input texts across all splits
   - Cross-platform POSIX LF (\\n) JSONL and SHA-256 checksums.json
2. `baseline_3k`: Deterministic reproduction of the original 3,000-row Phase 2 benchmark
   in tests/data/ (1,500 train, 500 dev, 500 test_id, 500 test_ood) with CRLF (\\r\\n).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

BANGKOK_TZ = timezone(timedelta(hours=7))
DEFAULT_REF_TIME = datetime(2026, 11, 8, 12, 0, 0, tzinfo=BANGKOK_TZ)  # Sunday
REF_TIME_STR = "2026-11-08T12:00:00+07:00"


# =============================================================================
# Context Domains & Semantic Time Filtering
# =============================================================================

class ContextDomain(str, Enum):
    OFFICE_BUSINESS = "OFFICE_BUSINESS"
    DAYTIME_COMMERCIAL = "DAYTIME_COMMERCIAL"
    DINING_SOCIAL = "DINING_SOCIAL"
    IT_OPS_MAINTENANCE = "IT_OPS_MAINTENANCE"
    FINANCIAL_TRANSACTIONS = "FINANCIAL_TRANSACTIONS"
    PERSONAL_CHAT = "PERSONAL_CHAT"
    OOD_DISJOINT_NOISE = "OOD_DISJOINT_NOISE"


class SemanticTimeFilter:
    """Filters candidate temporal expressions based on template context domain."""

    @staticmethod
    def extract_entity_hour_and_grain(cand: Dict[str, Any]) -> Tuple[Optional[int], str, bool]:
        """Extract effective hour (0-23), grain, and midnight crossing flag from candidate."""
        grain = cand.get("grain", "auto")
        has_midnight = bool(cand.get("has_midnight_crossing", False))

        # Date-level and year-level expressions have no diurnal clock restriction
        if grain in ("day", "year"):
            return None, grain, has_midnight

        iso_val = cand.get("iso_value")
        if iso_val and "T" in iso_val:
            try:
                hour = int(iso_val.split("T")[1].split(":")[0])
                return hour, grain, has_midnight
            except (IndexError, ValueError):
                pass

        range_s = cand.get("range_start")
        if range_s and "T" in range_s:
            try:
                hour = int(range_s.split("T")[1].split(":")[0])
                return hour, grain, has_midnight
            except (IndexError, ValueError):
                pass

        range_e = cand.get("range_end")
        if range_e and "T" in range_e:
            try:
                hour = int(range_e.split("T")[1].split(":")[0])
                return hour, grain, has_midnight
            except (IndexError, ValueError):
                pass

        return None, grain, has_midnight

    @classmethod
    def is_compatible(cls, domain: ContextDomain, cand: Dict[str, Any]) -> bool:
        """Evaluate compatibility between candidate entity and template semantic domain."""
        hour, grain, has_midnight = cls.extract_entity_hour_and_grain(cand)

        # Pure date/year entities (e.g. วันนี้, พรุ่งนี้, ปี 69) are allowed across all domains
        if hour is None:
            return True

        if domain == ContextDomain.OFFICE_BUSINESS:
            if has_midnight:
                return False
            return 8 <= hour <= 18

        elif domain == ContextDomain.DAYTIME_COMMERCIAL:
            if has_midnight:
                return False
            return 8 <= hour <= 19

        elif domain == ContextDomain.DINING_SOCIAL:
            # Lunch window: 11:00 - 14:30 (hours 11-14)
            # Dinner & Nightlife: 17:00 - 23:59 (hours 17-23)
            # Midnight social / drinks (hour 0 or has_midnight)
            if 11 <= hour <= 14:
                return True
            if 17 <= hour <= 23:
                return True
            if hour == 0 or has_midnight:
                return True
            return False

        elif domain == ContextDomain.IT_OPS_MAINTENANCE:
            # Nocturnal maintenance: 21:00 - 06:00
            if has_midnight:
                return True
            return hour >= 21 or hour <= 6

        elif domain in (
            ContextDomain.FINANCIAL_TRANSACTIONS,
            ContextDomain.PERSONAL_CHAT,
            ContextDomain.OOD_DISJOINT_NOISE,
        ):
            return True

        return True


# =============================================================================
# Algebraic Span Offset Calculator
# =============================================================================

class AlgebraicSpanCalculator:
    """Calculates exact character span offsets algebraically without substring search.

    Replaces str.find(text_span) to guarantee raw_input[start_char:end_char] == text_span.
    """

    @staticmethod
    def instantiate(pattern: str, text_span: str) -> Tuple[str, int, int]:
        parts = pattern.split("{entity}")
        if len(parts) != 2:
            raise ValueError(f"Pattern must contain exactly one '{{entity}}', got: {pattern}")
        prefix, suffix = parts[0], parts[1]
        start_char = len(prefix)
        end_char = start_char + len(text_span)
        raw_input = f"{prefix}{text_span}{suffix}"
        assert raw_input[start_char:end_char] == text_span, (
            f"Algebraic span mismatch: '{raw_input[start_char:end_char]}' != '{text_span}'"
        )
        return raw_input, start_char, end_char


# =============================================================================
# Sentence Template Registry (135 Templates + Negative Templates)
# =============================================================================

class SentenceTemplate:
    def __init__(self, template_id: str, pattern: str, domain: ContextDomain, length_tier: str):
        self.template_id = template_id
        self.pattern = pattern
        self.domain = domain
        self.length_tier = length_tier
        parts = pattern.split("{entity}")
        if len(parts) != 2:
            raise ValueError(f"Template pattern must contain exactly one '{{entity}}', got: '{pattern}'")
        self.prefix = parts[0]
        self.suffix = parts[1]


# Full 135-template library validated for length <= 88 chars (total <= 120 chars with entity)
TEMPLATES_RAW: List[Tuple[str, str, ContextDomain, str]] = [
    # Domain 1: OFFICE_BUSINESS (26 templates)
    ("TMPL_BUS_S01", "นัดคุยงาน {entity}", ContextDomain.OFFICE_BUSINESS, "SHORT"),
    ("TMPL_BUS_S02", "มีประชุมแผนงาน {entity}", ContextDomain.OFFICE_BUSINESS, "SHORT"),
    ("TMPL_BUS_S03", "ส่งเอกสารสัญญา {entity}", ContextDomain.OFFICE_BUSINESS, "SHORT"),
    ("TMPL_BUS_S04", "สัมภาษณ์รอบไฟนอล {entity}", ContextDomain.OFFICE_BUSINESS, "SHORT"),
    ("TMPL_BUS_S05", "กำหนดส่งรายงาน {entity}", ContextDomain.OFFICE_BUSINESS, "SHORT"),
    ("TMPL_BUS_S06", "เริ่มอบรมพนักงาน {entity}", ContextDomain.OFFICE_BUSINESS, "SHORT"),
    ("TMPL_BUS_S07", "นัดประชุมบอร์ด {entity}", ContextDomain.OFFICE_BUSINESS, "SHORT"),
    ("TMPL_BUS_S08", "{entity} มีพรีเซนต์งานลูกค้า", ContextDomain.OFFICE_BUSINESS, "SHORT"),
    ("TMPL_BUS_S09", "{entity}", ContextDomain.OFFICE_BUSINESS, "SHORT"),
    ("TMPL_BUS_M01", "ขอเชิญเข้าร่วมประชุมสรุปงบประมาณ {entity} ที่ห้องประชุม 1", ContextDomain.OFFICE_BUSINESS, "MEDIUM"),
    ("TMPL_BUS_M02", "รบกวนส่งสไลด์นำเสนอผลงานให้ภายใน {entity} เพื่อเตรียมจัดพิมพ์", ContextDomain.OFFICE_BUSINESS, "MEDIUM"),
    ("TMPL_BUS_M03", "ฝ่ายบุคคลนัดสัมภาษณ์งานตำแหน่งใหม่ {entity} ทางออนไลน์", ContextDomain.OFFICE_BUSINESS, "MEDIUM"),
    ("TMPL_BUS_M04", "หัวหน้าขอเลื่อนการประชุมทีมไปเป็น {entity} แทน", ContextDomain.OFFICE_BUSINESS, "MEDIUM"),
    ("TMPL_BUS_M05", "กำหนดการสัมมนาพัฒนาบุคลากรจะเริ่มขึ้น {entity} ตรง", ContextDomain.OFFICE_BUSINESS, "MEDIUM"),
    ("TMPL_BUS_M06", "ทีมวิจัยมีกำหนดส่งมอบรายงานฉบับสมบูรณ์ {entity}", ContextDomain.OFFICE_BUSINESS, "MEDIUM"),
    ("TMPL_BUS_M07", "กรุณาส่งแบบฟอร์มประเมินผลการทำงานภายใน {entity} นะครับ", ContextDomain.OFFICE_BUSINESS, "MEDIUM"),
    ("TMPL_BUS_M08", "{entity} จะมีการประชุมวางแผนกลยุทธ์ประจำปี", ContextDomain.OFFICE_BUSINESS, "MEDIUM"),
    ("TMPL_BUS_M09", "ตัวแทนฝ่ายขายจะเข้าพบคู่ค้าเพื่อเจรจาธุรกิจ {entity}", ContextDomain.OFFICE_BUSINESS, "MEDIUM"),
    ("TMPL_BUS_L01", "ตามที่ได้นัดหมายไว้ ขอเชิญทุกท่านเข้าร่วมประชุมทบทวนผลการดำเนินงาน {entity} ณ ห้องประชุมใหญ่", ContextDomain.OFFICE_BUSINESS, "LONG"),
    ("TMPL_BUS_L02", "ฝ่ายบริหารขอแจ้งกำหนดการยื่นข้อเสนอโครงการประจำปีรอบสุดท้ายภายใน {entity} นี้", ContextDomain.OFFICE_BUSINESS, "LONG"),
    ("TMPL_BUS_L03", "พนักงานทุกท่านที่ผ่านการคัดเลือกรบกวนมารายงานตัวและเริ่มปฐมนิเทศ {entity} ที่สำนักงานใหญ่", ContextDomain.OFFICE_BUSINESS, "LONG"),
    ("TMPL_BUS_L04", "เอกสารสัญญาความร่วมมือทางธุรกิจจะได้รับการลงนามอย่างเป็นทางการ {entity} ตามที่ตกลงกัน", ContextDomain.OFFICE_BUSINESS, "LONG"),
    ("TMPL_BUS_L05", "ทีมที่ปรึกษาการเงินจะเข้าชี้แจงแผนการปรับโครงสร้างองค์กร {entity} ให้คณะกรรมการทราบ", ContextDomain.OFFICE_BUSINESS, "LONG"),
    ("TMPL_BUS_L06", "{entity} ขอให้หัวหน้าแผนกทุกท่านเตรียมข้อมูลสรุปยอดขายเข้าประชุมวาระเร่งด่วน", ContextDomain.OFFICE_BUSINESS, "LONG"),
    ("TMPL_BUS_L07", "ทางบริษัทกำหนดจัดงานแถลงข่าวเปิดตัวผลิตภัณฑ์นวัตกรรมใหม่ {entity} ขอเรียนเชิญสื่อมวลชน", ContextDomain.OFFICE_BUSINESS, "LONG"),
    ("TMPL_BUS_L08", "กรุณาจัดเตรียมเอกสารหลักฐานและยื่นแบบแสดงรายการภาษีให้เรียบร้อยภายใน {entity} นะคะ", ContextDomain.OFFICE_BUSINESS, "LONG"),

    # Domain 2: DAYTIME_COMMERCIAL (20 templates)
    ("TMPL_COM_S01", "พร้อมจัดส่งสินค้า {entity}", ContextDomain.DAYTIME_COMMERCIAL, "SHORT"),
    ("TMPL_COM_S02", "หมอนัดตรวจสุขภาพ {entity}", ContextDomain.DAYTIME_COMMERCIAL, "SHORT"),
    ("TMPL_COM_S03", "นัดช่างเข้ามาตรวจเช็ค {entity}", ContextDomain.DAYTIME_COMMERCIAL, "SHORT"),
    ("TMPL_COM_S04", "ตรวจรับบ้าน {entity}", ContextDomain.DAYTIME_COMMERCIAL, "SHORT"),
    ("TMPL_COM_S05", "ส่งมอบรถยนต์ {entity}", ContextDomain.DAYTIME_COMMERCIAL, "SHORT"),
    ("TMPL_COM_S06", "{entity} พัสดุจะจัดส่งถึงบ้าน", ContextDomain.DAYTIME_COMMERCIAL, "SHORT"),
    ("TMPL_COM_M01", "พนักงานขนส่งจะโทรแจ้งล่วงหน้าก่อนนำจ่ายพัสดุช่วง {entity}", ContextDomain.DAYTIME_COMMERCIAL, "MEDIUM"),
    ("TMPL_COM_M02", "ทันตแพทย์นัดขูดหินปูนและตรวจฟัน {entity} กรุณาตรงเวลา", ContextDomain.DAYTIME_COMMERCIAL, "MEDIUM"),
    ("TMPL_COM_M03", "ช่างนิติบุคคลจะเข้าซ่อมท่อน้ำประปา {entity} ที่ห้องพัก", ContextDomain.DAYTIME_COMMERCIAL, "MEDIUM"),
    ("TMPL_COM_M04", "ลูกค้าสามารถนำรถยนต์เข้าตรวจเช็คระยะได้ {entity} ครับ", ContextDomain.DAYTIME_COMMERCIAL, "MEDIUM"),
    ("TMPL_COM_M05", "เจ้าหน้าที่จะเข้าไปตรวจประเมินราคาทรัพย์สิน {entity}", ContextDomain.DAYTIME_COMMERCIAL, "MEDIUM"),
    ("TMPL_COM_M06", "ห้างสรรพสินค้าเปิดให้บริการรอบพิเศษ {entity} เป็นต้นไป", ContextDomain.DAYTIME_COMMERCIAL, "MEDIUM"),
    ("TMPL_COM_M07", "นัดรับสินค้าที่สั่งจองไว้ที่สาขา {entity} ได้เลยนะคะ", ContextDomain.DAYTIME_COMMERCIAL, "MEDIUM"),
    ("TMPL_COM_M08", "{entity} ทางร้านจะเริ่มจัดส่งสินค้าตามลำดับคำสั่งซื้อ", ContextDomain.DAYTIME_COMMERCIAL, "MEDIUM"),
    ("TMPL_COM_L01", "เจ้าหน้าที่ช่างเทคนิคจะเดินทางเข้าไปติดตั้งเครื่องปรับอากาศ {entity} ตามที่ลูกค้านัดหมาย", ContextDomain.DAYTIME_COMMERCIAL, "LONG"),
    ("TMPL_COM_L02", "คลินิกเฉพาะทางขอแจ้งยืนยันคิวนัดตรวจของคุณหมอ {entity} กรุณามาก่อนเวลานัด 15 นาที", ContextDomain.DAYTIME_COMMERCIAL, "LONG"),
    ("TMPL_COM_L03", "บริษัทขนส่งเอกชนจะเข้ามารับพัสดุชิ้นใหญ่ที่บ้านของคุณ {entity} เพื่อนำไปกระจายสินค้า", ContextDomain.DAYTIME_COMMERCIAL, "LONG"),
    ("TMPL_COM_L04", "ฝ่ายบริการลูกค้าพร้อมส่งทีมช่างเข้าตรวจสอบระบบไฟฟ้า {entity} เพื่อความปลอดภัยของผู้อยู่อาศัย", ContextDomain.DAYTIME_COMMERCIAL, "LONG"),
    ("TMPL_COM_L05", "ศูนย์บริการรถยนต์ขอเรียนเชิญลูกค้านำรถเข้าทดลองขับ {entity} พร้อมรับข้อเสนอสุดพิเศษ", ContextDomain.DAYTIME_COMMERCIAL, "LONG"),
    ("TMPL_COM_L06", "{entity} เจ้าหน้าที่จะนำเอกสารสัญญาเช่าซื้อคอนโดมิเนียมไปให้เซ็นยินยอมที่โครงการ", ContextDomain.DAYTIME_COMMERCIAL, "LONG"),

    # Domain 3: DINING_SOCIAL (20 templates)
    ("TMPL_DIN_S01", "นัดทานข้าว {entity}", ContextDomain.DINING_SOCIAL, "SHORT"),
    ("TMPL_DIN_S02", "เจอกันที่ร้าน {entity} นะ", ContextDomain.DINING_SOCIAL, "SHORT"),
    ("TMPL_DIN_S03", "จองโต๊ะกินเลี้ยง {entity}", ContextDomain.DINING_SOCIAL, "SHORT"),
    ("TMPL_DIN_S04", "นัดจิบกาแฟคุยเล่น {entity}", ContextDomain.DINING_SOCIAL, "SHORT"),
    ("TMPL_DIN_S05", "ไปกินชาบูกัน {entity}", ContextDomain.DINING_SOCIAL, "SHORT"),
    ("TMPL_DIN_S06", "นัดฉลองวันเกิด {entity} ครับ", ContextDomain.DINING_SOCIAL, "SHORT"),
    ("TMPL_DIN_S07", "{entity} เจอที่ร้านอาหารญี่ปุ่น", ContextDomain.DINING_SOCIAL, "SHORT"),
    ("TMPL_DIN_M01", "จองโต๊ะร้านอาหารสำหรับ 4 ท่านไว้ {entity} เรียบร้อยแล้ว", ContextDomain.DINING_SOCIAL, "MEDIUM"),
    ("TMPL_DIN_M02", "เพื่อนๆ นัดรวมตัวเลี้ยงส่งพี่ที่ทำงาน {entity} ที่ร้านเดิม", ContextDomain.DINING_SOCIAL, "MEDIUM"),
    ("TMPL_DIN_M03", "สะดวกแวะมารับประทานอาหารกลางวันด้วยกัน {entity} ไหมครับ", ContextDomain.DINING_SOCIAL, "MEDIUM"),
    ("TMPL_DIN_M04", "นัดปาร์ตี้หมูกระทะบนดาดฟ้า {entity} อย่าลืมเตรียมของมานะ", ContextDomain.DINING_SOCIAL, "MEDIUM"),
    ("TMPL_DIN_M05", "คืนนี้เจอกันที่ร้านนั่งชิล {entity} ดนตรีสดเริ่มเล่นพอดี", ContextDomain.DINING_SOCIAL, "MEDIUM"),
    ("TMPL_DIN_M06", "ชวนไปชิมบุฟเฟต์ซีฟู้ดเปิดใหม่ {entity} ลดราคาพิเศษด้วย", ContextDomain.DINING_SOCIAL, "MEDIUM"),
    ("TMPL_DIN_M07", "{entity} นัดดื่มกาแฟและคุยเรื่องแผนเที่ยวพักผ่อน", ContextDomain.DINING_SOCIAL, "MEDIUM"),
    ("TMPL_DIN_L01", "ทางเราได้สำรองที่นั่งสำหรับงานเลี้ยงต้อนรับลูกค้าคนสำคัญ {entity} ที่ห้องอาหารชั้น 25", ContextDomain.DINING_SOCIAL, "LONG"),
    ("TMPL_DIN_L02", "นัดรวมรุ่นเพื่อนร่วมชั้นสมัยมัธยมจัดขึ้น {entity} ที่ร้านริมแม่น้ำเจ้าพระยา อย่าลืมมานะ", ContextDomain.DINING_SOCIAL, "LONG"),
    ("TMPL_DIN_L03", "ขอเชิญร่วมรับประทานอาหารค่ำเพื่อเฉลิมฉลองความสำเร็จของทีม {entity} พร้อมกิจกรรมจับสลาก", ContextDomain.DINING_SOCIAL, "LONG"),
    ("TMPL_DIN_L04", "ร้านอาหารอิตาเลียนขอยืนยันการจองคอร์สดินเนอร์สุดหรู {entity} ขอให้มีความสุขกับมื้อพิเศษ", ContextDomain.DINING_SOCIAL, "LONG"),
    ("TMPL_DIN_L05", "พวกเรามีแผนจะไปสังสรรค์ฉลองปิดโปรเจกต์กัน {entity} ทุกคนสามารถชวนครอบครัวมาได้ด้วย", ContextDomain.DINING_SOCIAL, "LONG"),
    ("TMPL_DIN_L06", "{entity} มีนัดดินเนอร์กับครอบครัวที่ภัตตาคารริมน้ำ บรรยากาศริมแม่น้ำดีมากๆ แน่นอน", ContextDomain.DINING_SOCIAL, "LONG"),

    # Domain 4: IT_OPS_MAINTENANCE (16 templates)
    ("TMPL_OPS_S01", "ปิดปรับปรุงระบบ {entity}", ContextDomain.IT_OPS_MAINTENANCE, "SHORT"),
    ("TMPL_OPS_S02", "ทีมไอที deploy โค้ด {entity}", ContextDomain.IT_OPS_MAINTENANCE, "SHORT"),
    ("TMPL_OPS_S03", "รีสตาร์ตเซิร์ฟเวอร์ {entity}", ContextDomain.IT_OPS_MAINTENANCE, "SHORT"),
    ("TMPL_OPS_S04", "แบ็กอัปฐานข้อมูล {entity}", ContextDomain.IT_OPS_MAINTENANCE, "SHORT"),
    ("TMPL_OPS_S05", "{entity} ปิดซ่อมบำรุงเครือข่าย", ContextDomain.IT_OPS_MAINTENANCE, "SHORT"),
    ("TMPL_OPS_M01", "ระบบคลาวด์จะปิดปรับปรุงชั่วคราว {entity} เพื่ออัปเกรดความปลอดภัย", ContextDomain.IT_OPS_MAINTENANCE, "MEDIUM"),
    ("TMPL_OPS_M02", "วิศวกรระบบจะเริ่มย้ายฐานข้อมูลเข้าศูนย์ข้อมูลใหม่ {entity}", ContextDomain.IT_OPS_MAINTENANCE, "MEDIUM"),
    ("TMPL_OPS_M03", "แจ้งระงับการใช้งานระบบธนาคารออนไลน์ช่วง {entity} ด่วน", ContextDomain.IT_OPS_MAINTENANCE, "MEDIUM"),
    ("TMPL_OPS_M04", "ทีมงานจะเข้าตรวจสอบเหตุไฟร์วอลล์ทำงานผิดปกติ {entity}", ContextDomain.IT_OPS_MAINTENANCE, "MEDIUM"),
    ("TMPL_OPS_M05", "กำหนดการปล่อยแพตช์ความปลอดภัยประจำเดือนช่วง {entity}", ContextDomain.IT_OPS_MAINTENANCE, "MEDIUM"),
    ("TMPL_OPS_M06", "{entity} มีแผนสลับการทำงานไปยังเครื่องแม่ข่ายสำรอง", ContextDomain.IT_OPS_MAINTENANCE, "MEDIUM"),
    ("TMPL_OPS_L01", "ทีมงาน DevOps จะอัปเดตเวอร์ชันและรีบูตเซิร์ฟเวอร์ {entity} ตามรอบซ่อมบำรุงดึก", ContextDomain.IT_OPS_MAINTENANCE, "LONG"),
    ("TMPL_OPS_L02", "เพื่อหลีกเลี่ยงผลกระทบต่อผู้ใช้งาน ทางเรากำหนดเวลาปิดปรับปรุงฐานข้อมูลส่วนกลาง {entity}", ContextDomain.IT_OPS_MAINTENANCE, "LONG"),
    ("TMPL_OPS_L03", "แจ้งตัดถ่ายสายไฟเบอร์ออปติกและเราเตอร์เครือข่าย {entity} ส่งผลให้ระบบขัดข้องชั่วขณะ", ContextDomain.IT_OPS_MAINTENANCE, "LONG"),
    ("TMPL_OPS_L04", "เจ้าหน้าที่ดูแลระบบจะเข้าดำเนินการสำรองข้อมูลกู้คืนภัยพิบัติประจำสัปดาห์ {entity}", ContextDomain.IT_OPS_MAINTENANCE, "LONG"),
    ("TMPL_OPS_L05", "{entity} ฝ่ายไอทีจะเข้าปฏิบัติงานอัปเกรดซอฟต์แวร์ระบบคลาวด์หลักเพื่อเพิ่มประสิทธิภาพ", ContextDomain.IT_OPS_MAINTENANCE, "LONG"),

    # Domain 5: FINANCIAL_TRANSACTIONS (16 templates)
    ("TMPL_FIN_S01", "ชำระค่าบริการ {entity}", ContextDomain.FINANCIAL_TRANSACTIONS, "SHORT"),
    ("TMPL_FIN_S02", "โอนเงินยอดแรก {entity}", ContextDomain.FINANCIAL_TRANSACTIONS, "SHORT"),
    ("TMPL_FIN_S03", "ปิดยอดขายประจำเดือน {entity}", ContextDomain.FINANCIAL_TRANSACTIONS, "SHORT"),
    ("TMPL_FIN_S04", "เตรียมเอกสารยื่นภาษี {entity}", ContextDomain.FINANCIAL_TRANSACTIONS, "SHORT"),
    ("TMPL_FIN_S05", "{entity} เงินเดือนจะโอนเข้าบัญชี", ContextDomain.FINANCIAL_TRANSACTIONS, "SHORT"),
    ("TMPL_FIN_M01", "กรุณาชำระยอดค้างชำระบัตรเครดิตให้เรียบร้อยภายใน {entity}", ContextDomain.FINANCIAL_TRANSACTIONS, "MEDIUM"),
    ("TMPL_FIN_M02", "ฝ่ายบัญชีจะโอนเงินค่าจ้างพนักงานฟรีแลนซ์ช่วง {entity} ครับ", ContextDomain.FINANCIAL_TRANSACTIONS, "MEDIUM"),
    ("TMPL_FIN_M03", "กำหนดการตัดรอบบิลและคำนวณดอกเบี้ยสินเชื่อบ้าน {entity}", ContextDomain.FINANCIAL_TRANSACTIONS, "MEDIUM"),
    ("TMPL_FIN_M04", "จะดำเนินการโอนเงินส่วนที่เหลือเข้าบัญชีให้ {entity} นะคะ", ContextDomain.FINANCIAL_TRANSACTIONS, "MEDIUM"),
    ("TMPL_FIN_M05", "โปรโมชั่นส่วนลดเงินคืนพิเศษจะสิ้นสุดลง {entity} นี้เท่านั้น", ContextDomain.FINANCIAL_TRANSACTIONS, "MEDIUM"),
    ("TMPL_FIN_M06", "{entity} ระบบจะทำการหักค่าธรรมเนียมรายปีอัตโนมัติ", ContextDomain.FINANCIAL_TRANSACTIONS, "MEDIUM"),
    ("TMPL_FIN_L01", "แจ้งเตือนกำหนดชำระเบี้ยประกันชีวิตงวดถัดไป โปรดชำระเงินก่อน {entity} เพื่อรักษาสิทธิ", ContextDomain.FINANCIAL_TRANSACTIONS, "LONG"),
    ("TMPL_FIN_L02", "บริษัทจะทำการจ่ายเงินปันผลระหว่างกาลแก่ผู้ถือหุ้นที่มีรายชื่อปรากฏในสมุดทะเบียน {entity}", ContextDomain.FINANCIAL_TRANSACTIONS, "LONG"),
    ("TMPL_FIN_L03", "กรุณาส่งใบเสร็จรับเงินเพื่อเบิกจ่ายงบประมาณโครงการให้ฝ่ายบัญชีภายใน {entity} เท่านั้น", ContextDomain.FINANCIAL_TRANSACTIONS, "LONG"),
    ("TMPL_FIN_L04", "ระบบจะเปิดให้ทำรายการกู้ยืมและลงทะเบียนรับสิทธิประโยชน์ทางการเงิน {entity} เป็นต้นไป", ContextDomain.FINANCIAL_TRANSACTIONS, "LONG"),
    ("TMPL_FIN_L05", "{entity} ฝ่ายการเงินจะส่งรายงานสรุปกระแสเงินสดและงบดุลประจำไตรมาสให้ผู้สอบบัญชี", ContextDomain.FINANCIAL_TRANSACTIONS, "LONG"),

    # Domain 6: PERSONAL_CHAT (22 templates)
    ("TMPL_CHT_S01", "เจอกัน {entity} นะ", ContextDomain.PERSONAL_CHAT, "SHORT"),
    ("TMPL_CHT_S02", "ว่าไงพวก ว่างเจอกัน {entity} ปะ", ContextDomain.PERSONAL_CHAT, "SHORT"),
    ("TMPL_CHT_S03", "เด๋วรีบทักไปหา {entity} เด้อ", ContextDomain.PERSONAL_CHAT, "SHORT"),
    ("TMPL_CHT_S04", "ออนดิสคอร์ด {entity} นะทุกคน", ContextDomain.PERSONAL_CHAT, "SHORT"),
    ("TMPL_CHT_S05", "ตื่นนอนตอน {entity} ง่วงมาก", ContextDomain.PERSONAL_CHAT, "SHORT"),
    ("TMPL_CHT_S06", "เค้าจะแวะไปหานะ {entity}", ContextDomain.PERSONAL_CHAT, "SHORT"),
    ("TMPL_CHT_S07", "ขอเลื่อนนัดไป {entity} แทน", ContextDomain.PERSONAL_CHAT, "SHORT"),
    ("TMPL_CHT_S08", "{entity} ทักแชตมาบอกด้วยน้า", ContextDomain.PERSONAL_CHAT, "SHORT"),
    ("TMPL_CHT_M01", "เอ่อ... เจอกัน {entity} ละกันนะเพื่อน ไม่ต้องรีบ", ContextDomain.PERSONAL_CHAT, "MEDIUM"),
    ("TMPL_CHT_M02", "แกรรรร เจอกัน {entity} ห้ามสายเด็ดขาด ใครสายเลี้ยงข้าว", ContextDomain.PERSONAL_CHAT, "MEDIUM"),
    ("TMPL_CHT_M03", "เค้าติดงานด่วนอยู่ น่าจะว่างคุยอีกทีช่วง {entity} จ้า", ContextDomain.PERSONAL_CHAT, "MEDIUM"),
    ("TMPL_CHT_M04", "ประมาณ {entity} น่าจะถึงห้อง เดี๋ยวโทรหาอีกรอบนะ", ContextDomain.PERSONAL_CHAT, "MEDIUM"),
    ("TMPL_CHT_M05", "คืนนี้นอนดึกแน่ๆ กะว่าจะเล่นเกมยัน {entity} เลยว่ะ", ContextDomain.PERSONAL_CHAT, "MEDIUM"),
    ("TMPL_CHT_M06", "ต้องตื่นแต่เช้ามาก มีนัดตั้งแต่ {entity} นู่นเลย", ContextDomain.PERSONAL_CHAT, "MEDIUM"),
    ("TMPL_CHT_M07", "ส่งรูปทริปเที่ยวให้ดูหน่อยสิ เดี๋ยวทักไปหาช่วง {entity}", ContextDomain.PERSONAL_CHAT, "MEDIUM"),
    ("TMPL_CHT_M08", "{entity} ค่อยมานั่งคุยรายละเอียดกันอีกทีนะเตง", ContextDomain.PERSONAL_CHAT, "MEDIUM"),
    ("TMPL_CHT_L01", "การจราจรติดขัดหนักมากเลยอะพวกแก น่าจะหลุดจากแถวนี้แล้วไปถึงจุดนัดพบประมาณ {entity} แน่ๆ", ContextDomain.PERSONAL_CHAT, "LONG"),
    ("TMPL_CHT_L02", "ถ้าการบ้านยังทำไม่เสร็จ คืนนี้พวกเรามาเปิดคอลดิสคอร์ดนั่งปั่นงานด้วยกันตอน {entity} ไหม", ContextDomain.PERSONAL_CHAT, "LONG"),
    ("TMPL_CHT_L03", "ช่วงนี้งานยุ่งจนแทบไม่ได้พักเลย ไว้ถ้าเคลียร์งานเสร็จแล้วนัดเจอกันกินขนม {entity} ละกันน้า", ContextDomain.PERSONAL_CHAT, "LONG"),
    ("TMPL_CHT_L04", "อย่าลืมตั้งนาฬิกาปลุกไว้ด้วยนะ ไฟลต์บินเช้ามาก ต้องรีบออกจากบ้าน {entity} ห้ามลืม", ContextDomain.PERSONAL_CHAT, "LONG"),
    ("TMPL_CHT_L05", "เดี๋ยวแวะซื้อของขวัญวันเกิดให้เพื่อนก่อน แล้วค่อยตามไปสมทบกับทุกคนช่วง {entity} นะ", ContextDomain.PERSONAL_CHAT, "LONG"),
    ("TMPL_CHT_L06", "{entity} ว่างไหม อยากชวนออกไปหาของหวานกินแถวสยาม บ่นอยากกินบิงซูมาหลายวันแล้ว", ContextDomain.PERSONAL_CHAT, "LONG"),

    # Domain 7: OOD_DISJOINT_NOISE (15 templates)
    ("TMPL_OOD_S01", "[URGENT] ส่งมอบงานด่วน {entity}", ContextDomain.OOD_DISJOINT_NOISE, "SHORT"),
    ("TMPL_OOD_S02", "Order Confirmed -> {entity}", ContextDomain.OOD_DISJOINT_NOISE, "SHORT"),
    ("TMPL_OOD_S03", "นัดหมายด่วนที่สุด: {entity}", ContextDomain.OOD_DISJOINT_NOISE, "SHORT"),
    ("TMPL_OOD_S04", "พร้อมตรวจรับมอบงาน {entity} !!!", ContextDomain.OOD_DISJOINT_NOISE, "SHORT"),
    ("TMPL_OOD_S05", "{entity} -> ปิดระบบฉุกเฉิน", ContextDomain.OOD_DISJOINT_NOISE, "SHORT"),
    ("TMPL_OOD_M01", "[ALERT] ระบบจะทำการรีสตาร์ตเพื่อความปลอดภัย: {entity}", ContextDomain.OOD_DISJOINT_NOISE, "MEDIUM"),
    ("TMPL_OOD_M02", "ขอแจ้งนัดหมายล่วงหน้านะครับ... {entity} กรุณายืนยันกลับ", ContextDomain.OOD_DISJOINT_NOISE, "MEDIUM"),
    ("TMPL_OOD_M03", "โปรโมชั่นสุดคุ้ม! สิ้นสุด {entity} เท่านั้น -> ห้ามพลาด", ContextDomain.OOD_DISJOINT_NOISE, "MEDIUM"),
    ("TMPL_OOD_M04", ">>> ประกาศเปิดให้ลงทะเบียนรับสิทธิ์พิเศษ {entity} เป็นต้นไป", ContextDomain.OOD_DISJOINT_NOISE, "MEDIUM"),
    ("TMPL_OOD_M05", "{entity} :: กำหนดการถ่ายทอดสดสัมมนาออนไลน์รอบพิเศษ", ContextDomain.OOD_DISJOINT_NOISE, "MEDIUM"),
    ("TMPL_OOD_L01", "[NOTICE]: ระบบส่วนกลางจะหยุดให้บริการชั่วคราวเพื่อปรับปรุง {entity} ขออภัยในความไม่สะดวก", ContextDomain.OOD_DISJOINT_NOISE, "LONG"),
    ("TMPL_OOD_L02", "*** แจ้งเตือน: ทางร้านขอสงวนสิทธิ์ยกเลิกออเดอร์ที่ไม่ชำระเงินก่อน {entity} ทุกกรณี ***", ContextDomain.OOD_DISJOINT_NOISE, "LONG"),
    ("TMPL_OOD_L03", "[URGENT] กรุณายืนยันการเข้าร่วมประชุมสามัญประจำปีภายใน {entity} เพื่อจัดเตรียมอาหารรับรอง", ContextDomain.OOD_DISJOINT_NOISE, "LONG"),
    ("TMPL_OOD_L04", "ข้อความอัตโนมัติ -> พนักงานจะนำส่งพัสดุตามที่อยู่ที่ระบุไว้ช่วง {entity} ติดต่อเบอร์คนขับ", ContextDomain.OOD_DISJOINT_NOISE, "LONG"),
    ("TMPL_OOD_L05", "{entity} => สิ้นสุดระยะเวลาการยื่นข้อเสนอโครงการประกวดราคาอิเล็กทรอนิกส์ของภาครัฐ", ContextDomain.OOD_DISJOINT_NOISE, "LONG"),
]


class TemplateRegistry:
    """Registry managing the 135 templates and negative sample templates."""

    def __init__(self):
        self.templates = [
            SentenceTemplate(tid, pat, dom, tier)
            for tid, pat, dom, tier in TEMPLATES_RAW
        ]
        assert len(self.templates) == 135, f"Expected 135 templates, got {len(self.templates)}"

        self.by_domain: Dict[ContextDomain, List[SentenceTemplate]] = {}
        for tmpl in self.templates:
            self.by_domain.setdefault(tmpl.domain, []).append(tmpl)

        self.in_distribution = [
            t for t in self.templates if t.domain != ContextDomain.OOD_DISJOINT_NOISE
        ]
        self.ood_templates = [
            t for t in self.templates if t.domain == ContextDomain.OOD_DISJOINT_NOISE
        ]

    def get_templates_by_domain(self, domain: ContextDomain) -> List[SentenceTemplate]:
        return self.by_domain.get(domain, [])

    def get_in_distribution_templates(self) -> List[SentenceTemplate]:
        return self.in_distribution

    def get_ood_templates(self) -> List[SentenceTemplate]:
        return self.ood_templates


# Negative carrier templates (non-temporal / numeric distractors)
NEGATIVE_TEMPLATES = {
    "currency": [
        "โอนเงินจำนวน {amount} {unit} เข้าบัญชีเรียบร้อยแล้วครับ",
        "ยอดชำระทั้งหมด {amount} {unit} รบกวนตรวจสอบสลิปการโอนด้วยครับ",
        "สินค้าราคาพิเศษเพียง {amount} {unit} รวมค่าจัดส่งแล้ว",
        "มัดจำค่าบริการล่วงหน้า {amount} {unit} ก่อนเริ่มงานงวดแรก",
        "ระบบได้ทำการคืนเงินยอด {amount} {unit} เข้าบัญชีของท่านแล้ว",
        "มีค่าธรรมเนียมการทำรายการ {amount} {unit}",
        "แจ้งยอดคงเหลือในบัญชีเงินฝาก {amount} {unit}",
        "ขอใบเสร็จรับเงินสำหรับยอดชำระ {amount} {unit} ด้วยครับ",
        "ส่วนลดพิเศษสำหรับสมาชิก มูลค่า {amount} {unit}",
        "ชำระเบี้ยประกันรายเดือนยอดรวม {amount} {unit} สำเร็จ",
        "โอนเงินค่ามัดจำสินค้าจำนวน {amount} {unit} เรียบร้อยแล้วครับ",
        "เสื้อเชิ้ตตัวนี้ราคาพิเศษเพียง {amount} {unit} จัดส่งฟรีทั่วไทย",
        "ค่าอาหารและเครื่องดื่มมื้อนี้รวมเป็นเงินทั้งสิ้น {amount} {unit}",
    ],
    "tracking_code": [
        "พัสดุหมายเลข {code} กำลังอยู่ระหว่างการนำจ่ายโดยเจ้าหน้าที่",
        "แจ้งเลขพัสดุสำหรับติดตามสินค้า: {code}",
        "สามารถตรวจสอบสถานะการจัดส่งได้ที่รหัสติดตาม {code}",
        "พัสดุหมายเลข {code} จัดส่งถึงศูนย์คัดแยกสินค้าเรียบร้อยแล้ว",
        "เจ้าหน้าที่ขนส่งกำลังนำจ่ายพัสดุรหัส {code} ให้ท่าน",
        "จัดส่งสินค้าสำเร็จแล้ว เลขแทร็กกิ้ง {code} ขอบคุณที่ใช้บริการ",
        "กรุณาตรวจสอบความถูกต้องของหมายเลขพัสดุ {code}",
        "พัสดุด่วนพิเศษ EMS รหัส {code} ส่งถึงผู้รับแล้ว",
        "Tracking Number: {code} ตรวจสอบสถานะการขนส่งผ่านแอปพลิเคชัน",
        "คำสั่งซื้อหมายเลข #{code} ได้รับการอนุมัติเรียบร้อย",
    ],
    "phone_number": [
        "หากมีข้อสงสัยกรุณาติดต่อเจ้าหน้าที่ที่เบอร์ {phone}",
        "ฝ่ายบริการลูกค้าสัมพันธ์ยินดีให้บริการที่หมายเลข {phone}",
        "สามารถติดต่อผู้ประสานงานได้ที่เบอร์โทร {phone} ครับ",
        "สอบถามรายละเอียดข้อมูลเพิ่มเติม โทร {phone} ในวันทำการ",
        "เบอร์โทรศัพท์สำหรับติดต่อกรณีฉุกเฉินคือ {phone}",
        "กรุณายืนยันหมายเลขโทรศัพท์ {phone} เพื่อรับสิทธิพิเศษ",
        "ติดต่อสอบถามฝ่ายขายโดยตรงที่หมายเลข {phone}",
        "ศูนย์บริการข้อมูลลูกค้า Hotline {phone} ยินดีให้บริการและช่วยเหลือทุกขั้นตอน",
        "สนใจสมัครสมาชิกหรือติดต่อสอบถามรายละเอียดที่เบอร์ {phone}",
        "ติดต่อแจ้งเหตุฉุกเฉินหรือขอความช่วยเหลือได้ที่สายด่วน {phone}",
    ],
    "general_text": [
        "การประชุมจัดขึ้นที่ห้องประชุม {room} ชั้น {floor} อาคารบริหาร",
        "ผลการประเมินความพึงพอใจของลูกค้าประจำไตรมาสอยู่ที่ {pct}%",
        "สินค้าในคลังมีจำนวนคงเหลือทั้งหมด {qty} ชิ้น พร้อมจัดส่ง",
        "สั่งซื้อชานมไข่มุกจำนวน {qty} แก้ว หวาน 50% เพิ่มไข่มุก",
        "ขอเชิญเข้าร่วมประชุมที่ห้องสัมมนา {room} อาคารสำนักงานใหญ่",
        "โปรโมชั่นลดราคาสินค้าเครื่องใช้ไฟฟ้าทุกรุ่นสูงสุด {pct}%",
        "มีผู้ลงทะเบียนเข้าร่วมกิจกรรมอบรมเชิงปฏิบัติการแล้ว {qty} คน",
        "สวัสดีครับ สบายดีไหมครับ ไม่ได้พบกันนานเลย หวังว่าสบายดีนะ",
        "ขอบคุณสำหรับความช่วยเหลือและคำแนะนำที่มีประโยชน์มากๆ นะครับ",
        "สภาพอากาศมีฝนตกหนักบนถนนหลายสาย โปรดขับขี่ยานพาหนะด้วยความระมัดระวัง",
        "ร้านอาหารร้านนี้รสชาติกลมกล่อมและบริการประทับใจมาก แนะนำให้มาลอง",
        "ยินดีด้วยกับความสำเร็จในก้าวสำคัญของชีวิต ขอให้มีความสุขมากๆ",
        "ได้รับไฟล์เอกสารสัญญาครบถ้วนเรียบร้อยแล้วครับ ขอบคุณมากครับ",
        "รบกวนช่วยแชร์ตำแหน่งที่ตั้งของร้านอาหารให้ทางแชตหน่อยนะครับ",
        "ยินดีที่ได้ร่วมงานกันครับ หวังว่าโอกาสหน้าจะได้ร่วมงานกันอีก",
        "ทางบริษัทขอขอบพระคุณลูกค้าทุกท่านที่ให้ความไว้วางใจเสมอมา",
        "ขออภัยในความไม่สะดวก ทางทีมงานกำลังเร่งแก้ไขปัญหาอย่างเต็มที่ครับ",
        "รหัสยืนยัน OTP สำหรับเข้าสู่ระบบคือ {otp} (ห้ามเปิดเผยแก่ผู้อื่น)",
    ],
}


# =============================================================================
# Global Cross-Split Deduplicator
# =============================================================================

class CrossSplitDeduplicator:
    """Global registry guaranteeing strictly zero duplicate raw_input texts within and across splits."""

    def __init__(self) -> None:
        self._seen: Set[str] = set()
        self._split_counts: Dict[str, int] = {}
        self._collision_counts: Dict[str, int] = {}

    def is_duplicate(self, text: str) -> bool:
        return text.strip() in self._seen

    def register(self, text: str, split: str) -> bool:
        normalized = text.strip()
        if normalized in self._seen:
            self._collision_counts[split] = self._collision_counts.get(split, 0) + 1
            return False
        self._seen.add(normalized)
        self._split_counts[split] = self._split_counts.get(split, 0) + 1
        return True

    def total_unique(self) -> int:
        return len(self._seen)

    def stats(self) -> Dict[str, Any]:
        return {
            "total_unique": len(self._seen),
            "split_counts": dict(self._split_counts),
            "collisions_rejected": dict(self._collision_counts),
            "total_collisions": sum(self._collision_counts.values()),
        }


# =============================================================================
# Candidate Entity Definitions & Expanded Pools
# =============================================================================

RELATIVE_DAYS = [
    ("วันนี้", 0, "deictic_today"),
    ("พรุ่งนี้", 1, "deictic_tomorrow"),
    ("มะรืนนี้", 2, "deictic_day_after_tomorrow"),
    ("เมื่อวาน", -1, "deictic_yesterday"),
    ("เมื่อวานนี้", -1, "deictic_yesterday_full"),
    ("วานซืน", -2, "deictic_day_before_yesterday"),
    ("เมื่อวานซืน", -2, "deictic_day_before_yesterday_full"),
]

WEEKDAY_OFFSETS = [
    ("จันทร์หน้า", 8, "relative_weekday_next"),
    ("วันจันทร์หน้า", 8, "relative_weekday_next"),
    ("จันทร์ที่จะถึงนี้", 1, "relative_weekday_upcoming"),
    ("อังคารหน้า", 9, "relative_weekday_next"),
    ("วันอังคารหน้า", 9, "relative_weekday_next"),
    ("พุธหน้า", 3, "relative_weekday_next"),
    ("วันพุธหน้า", 3, "relative_weekday_next"),
    ("พฤหัสหน้า", 4, "relative_weekday_next"),
    ("วันพฤหัสหน้า", 4, "relative_weekday_next"),
    ("ศุกร์หน้า", 5, "relative_weekday_next"),
    ("วันศุกร์หน้า", 5, "relative_weekday_next"),
    ("เสาร์หน้า", 6, "relative_weekday_next"),
    ("วันเสาร์หน้า", 6, "relative_weekday_next"),
    ("อาทิตย์หน้า", 7, "relative_weekday_next"),
    ("วันอาทิตย์หน้า", 7, "relative_weekday_next"),
    ("จันทร์ที่แล้ว", -6, "relative_weekday_last"),
    ("อังคารที่แล้ว", -5, "relative_weekday_last"),
    ("พุธที่แล้ว", -4, "relative_weekday_last"),
    ("พฤหัสที่แล้ว", -3, "relative_weekday_last"),
    ("ศุกร์ที่แล้ว", -2, "relative_weekday_last"),
    ("เสาร์ที่แล้ว", -1, "relative_weekday_last"),
    ("เสาร์ที่ผ่านมา", -1, "relative_weekday_past"),
    ("ศุกร์นี้", 5, "relative_weekday_this"),
    ("เสาร์นี้", 6, "relative_weekday_this"),
    ("พุธนี้", 3, "relative_weekday_this"),
    ("พฤหัสนี้", 4, "relative_weekday_this"),
]

NUMERIC_DAY_OFFSETS = [
    ("อีก 1 วัน", 1),
    ("อีก 2 วัน", 2),
    ("อีก 3 วัน", 3),
    ("อีก 4 วัน", 4),
    ("อีก 5 วัน", 5),
    ("อีก 7 วัน", 7),
    ("อีก 10 วัน", 10),
    ("1 วันก่อน", -1),
    ("2 วันก่อน", -2),
    ("3 วันที่แล้ว", -3),
    ("4 วันก่อน", -4),
    ("5 วันที่แล้ว", -5),
    ("อีกสองวัน", 2),
    ("อีกสามวัน", 3),
    ("อีก 1 สัปดาห์", 7),
    ("อีก 2 สัปดาห์", 14),
]

CLOCK_MORNING = [
    ("โมงเช้า", 7, 0, "clock_morning_1st"),
    ("เจ็ดโมงเช้า", 7, 0, "clock_morning"),
    ("แปดโมงเช้า", 8, 0, "clock_morning"),
    ("8 โมงเช้า", 8, 0, "clock_morning_digit"),
    ("แปดโมงครึ่ง", 8, 30, "clock_morning_half"),
    ("เก้าโมง", 9, 0, "clock_morning"),
    ("เก้าโมงเช้า", 9, 0, "clock_morning"),
    ("9 โมง", 9, 0, "clock_morning_digit"),
    ("เก้าโมงครึ่ง", 9, 30, "clock_morning_half"),
    ("สิบโมงเช้า", 10, 0, "clock_morning"),
    ("สิบโมง", 10, 0, "clock_morning"),
    ("10 โมง", 10, 0, "clock_morning_digit"),
    ("สิบโมงสิบห้า", 10, 15, "clock_morning_fraction"),
    ("สิบโมงครึ่ง", 10, 30, "clock_morning_half"),
    ("สิบโมงสี่สิบห้า", 10, 45, "clock_morning_fraction"),
    ("สิบเอ็ดโมง", 11, 0, "clock_morning"),
    ("11 โมง", 11, 0, "clock_morning_digit"),
    ("สิบเอ็ดโมงครึ่ง", 11, 30, "clock_morning_half"),
]

CLOCK_NOON = [
    ("เที่ยง", 12, 0, "clock_noon"),
    ("เที่ยงตรง", 12, 0, "clock_noon_exact"),
    ("เที่ยงวัน", 12, 0, "clock_noon"),
    ("เที่ยงครึ่ง", 12, 30, "clock_noon_half"),
    ("เที่ยง 15", 12, 15, "clock_noon_digit"),
    ("เที่ยงสี่สิบห้า", 12, 45, "clock_noon_fraction"),
]

CLOCK_AFTERNOON = [
    ("บ่ายโมง", 13, 0, "clock_afternoon"),
    ("บ่ายโมงตรง", 13, 0, "clock_afternoon"),
    ("บ่าย 1", 13, 0, "clock_afternoon_digit"),
    ("บ่ายโมงครึ่ง", 13, 30, "clock_afternoon_half"),
    ("บ่ายสอง", 14, 0, "clock_afternoon"),
    ("บ่ายสองโมง", 14, 0, "clock_afternoon"),
    ("บ่าย 2", 14, 0, "clock_afternoon_digit"),
    ("บ่ายสองครึ่ง", 14, 30, "clock_afternoon_half"),
    ("บ่ายสองสิบห้า", 14, 15, "clock_afternoon_fraction"),
    ("บ่ายสองสี่สิบห้า", 14, 45, "clock_afternoon_fraction"),
    ("บ่ายสาม", 15, 0, "clock_afternoon"),
    ("บ่ายสามโมง", 15, 0, "clock_afternoon"),
    ("บ่าย 3", 15, 0, "clock_afternoon_digit"),
    ("บ่ายสามครึ่ง", 15, 30, "clock_afternoon_half"),
]

CLOCK_EVENING = [
    ("สี่โมงเย็น", 16, 0, "clock_evening"),
    ("4 โมงเย็น", 16, 0, "clock_evening_digit"),
    ("สี่โมงครึ่ง", 16, 30, "clock_evening_half"),
    ("ห้าโมงเย็น", 17, 0, "clock_evening"),
    ("5 โมงเย็น", 17, 0, "clock_evening_digit"),
    ("ห้าโมงครึ่ง", 17, 30, "clock_evening_half"),
    ("หกโมงเย็น", 18, 0, "clock_evening"),
    ("6 โมงเย็น", 18, 0, "clock_evening_digit"),
    ("หกโมงครึ่ง", 18, 30, "clock_evening_half"),
]

CLOCK_NIGHT = [
    ("หนึ่งทุ่ม", 19, 0, "clock_night"),
    ("1 ทุ่ม", 19, 0, "clock_night_digit"),
    ("ทุ่มนึง", 19, 0, "clock_night_slang"),
    ("ทุ่มตรง", 19, 0, "clock_night"),
    ("ทุ่มครึ่ง", 19, 30, "clock_night_half"),
    ("หนึ่งทุ่มครึ่ง", 19, 30, "clock_night_half"),
    ("สองทุ่ม", 20, 0, "clock_night"),
    ("2 ทุ่ม", 20, 0, "clock_night_digit"),
    ("สองทุ่มครึ่ง", 20, 30, "clock_night_half"),
    ("สองทุ่ม 15", 20, 15, "clock_night_fraction"),
    ("สามทุ่ม", 21, 0, "clock_night"),
    ("3 ทุ่ม", 21, 0, "clock_night_digit"),
    ("สามทุ่มครึ่ง", 21, 30, "clock_night_half"),
    ("สามทุ่มสิบห้า", 21, 15, "clock_night_fraction"),
    ("สี่ทุ่ม", 22, 0, "clock_night"),
    ("4 ทุ่ม", 22, 0, "clock_night_digit"),
    ("สี่ทุ่มครึ่ง", 22, 30, "clock_night_half"),
    ("ห้าทุ่ม", 23, 0, "clock_night"),
    ("5 ทุ่ม", 23, 0, "clock_night_digit"),
    ("ห้าทุ่มครึ่ง", 23, 30, "clock_night_half"),
    ("ห้าทุ่มสี่สิบห้า", 23, 45, "clock_night_fraction"),
]

CLOCK_MIDNIGHT = [
    ("เที่ยงคืน", 0, 0, "clock_midnight"),
    ("เที่ยงคืนตรง", 0, 0, "clock_midnight_exact"),
    ("เที่ยงคืนครึ่ง", 0, 30, "clock_midnight_half"),
]

CLOCK_EARLY_MORNING = [
    ("ตีหนึ่ง", 1, 0, "clock_wee_hours"),
    ("ตี 1", 1, 0, "clock_wee_hours_digit"),
    ("ตีหนึ่งครึ่ง", 1, 30, "clock_wee_hours_half"),
    ("ตีสอง", 2, 0, "clock_wee_hours"),
    ("ตี 2", 2, 0, "clock_wee_hours_digit"),
    ("ตีสองครึ่ง", 2, 30, "clock_wee_hours_half"),
    ("ตีสองสิบห้า", 2, 15, "clock_wee_hours_fraction"),
    ("ตีสาม", 3, 0, "clock_wee_hours"),
    ("ตี 3", 3, 0, "clock_wee_hours_digit"),
    ("ตีสามครึ่ง", 3, 30, "clock_wee_hours_half"),
    ("ตีสี่", 4, 0, "clock_wee_hours"),
    ("ตี 4", 4, 0, "clock_wee_hours_digit"),
    ("ตีสี่ครึ่ง", 4, 30, "clock_wee_hours_half"),
    ("ตีห้า", 5, 0, "clock_wee_hours"),
    ("ตี 5", 5, 0, "clock_wee_hours_digit"),
    ("ตีห้าครึ่ง", 5, 30, "clock_wee_hours_half"),
]

COARSE_PERIODS = [
    ("เช้ามืด", "04:00:00", "06:00:00", 0, False, "coarse_early_morning"),
    ("สายๆ", "09:00:00", "11:30:00", 0, False, "coarse_late_morning"),
    ("ตอนสายๆ", "09:00:00", "11:30:00", 0, False, "coarse_late_morning_prep"),
    ("ช่วงสายๆ", "09:00:00", "11:30:00", 0, False, "coarse_late_morning_prep"),
    ("บ่ายๆ", "13:00:00", "15:30:00", 0, False, "coarse_afternoon"),
    ("ตอนบ่ายๆ", "13:00:00", "15:30:00", 0, False, "coarse_afternoon_prep"),
    ("ช่วงบ่ายๆ", "13:00:00", "15:30:00", 0, False, "coarse_afternoon_prep"),
    ("บ่ายแก่ๆ", "15:00:00", "17:00:00", 0, False, "coarse_late_afternoon"),
    ("ตอนบ่ายแก่ๆ", "15:00:00", "17:00:00", 0, False, "coarse_late_afternoon_prep"),
    ("เย็นๆ", "16:30:00", "18:30:00", 0, False, "coarse_evening"),
    ("ตอนเย็นๆ", "16:30:00", "18:30:00", 0, False, "coarse_evening_prep"),
    ("ช่วงเย็นๆ", "16:30:00", "18:30:00", 0, False, "coarse_evening_prep"),
    ("หัวค่ำ", "18:30:00", "20:30:00", 0, False, "coarse_dusk"),
    ("ช่วงหัวค่ำ", "18:30:00", "20:30:00", 0, False, "coarse_dusk_prep"),
    ("ค่ำๆ", "19:00:00", "21:30:00", 0, False, "coarse_night"),
    ("ตอนค่ำๆ", "19:00:00", "21:30:00", 0, False, "coarse_night_prep"),
    ("ดึกๆ", "22:00:00", "02:00:00", 1, True, "coarse_late_night_midnight_cross"),
    ("ตอนดึกๆ", "22:00:00", "02:00:00", 1, True, "coarse_late_night_midnight_cross"),
    ("ช่วงดึกๆ", "22:00:00", "02:00:00", 1, True, "coarse_late_night_midnight_cross"),
]

BUDDHIST_ERA_YEARS = [
    ("ปี 67", 2024, "be_short_year"),
    ("ปี 68", 2025, "be_short_year"),
    ("ปี 69", 2026, "be_short_year"),
    ("ปี 70", 2027, "be_short_year"),
    ("ปี 71", 2028, "be_short_year"),
    ("ปี 72", 2029, "be_short_year"),
    ("ปี 75", 2032, "be_short_year"),
    ("ปี 78", 2035, "be_short_year"),
    ("ปี 79", 2036, "be_short_year_pivot_edge"),
    ("ปี 80", 1937, "be_short_year_pivot_past"),
    ("พ.ศ. 2567", 2024, "be_full_year"),
    ("พ.ศ. 2568", 2025, "be_full_year"),
    ("พ.ศ. 2569", 2026, "be_full_year"),
    ("พ.ศ. 2570", 2027, "be_full_year"),
    ("ต้นปี 68", 2025, "be_short_year_modifier_early"),
    ("กลางปี 69", 2026, "be_short_year_modifier_mid"),
    ("ปลายปี 67", 2024, "be_short_year_modifier_late"),
]

BUDDHIST_ERA_DATES = [
    ("1 ม.ค. 67", "2024-01-01T00:00:00+07:00", "be_date_abbr"),
    ("15 มี.ค. 67", "2024-03-15T00:00:00+07:00", "be_date_abbr"),
    ("14 ก.พ. 68", "2025-02-14T00:00:00+07:00", "be_date_abbr"),
    ("13 เม.ย. 69", "2026-04-13T00:00:00+07:00", "be_date_abbr"),
    ("28 ก.ค. 68", "2025-07-28T00:00:00+07:00", "be_date_abbr"),
    ("12 ส.ค. 69", "2026-08-12T00:00:00+07:00", "be_date_abbr"),
    ("5 ธ.ค. 67", "2024-12-05T00:00:00+07:00", "be_date_abbr"),
    ("31 ธ.ค. 68", "2025-12-31T00:00:00+07:00", "be_date_abbr"),
    ("1 มกราคม 2568", "2025-01-01T00:00:00+07:00", "be_date_full"),
    ("28 ก.พ. 67", "2024-02-28T00:00:00+07:00", "be_date_leap_valid"),
    ("29 ก.พ. 67", "2024-02-29T00:00:00+07:00", "be_date_leap_valid"),
]

NIGHT_ROLLOVERS = [
    ("คืนวันศุกร์ ตีสอง", "2026-11-14T02:00:00+07:00", "night_rollover_friday_sat_morning"),
    ("คืนวันศุกร์ ตีหนึ่ง", "2026-11-14T01:00:00+07:00", "night_rollover_friday_sat_morning"),
    ("คืนวันเสาร์ ตีหนึ่ง", "2026-11-15T01:00:00+07:00", "night_rollover_sat_sun_morning"),
    ("คืนวันเสาร์ ตีสาม", "2026-11-15T03:00:00+07:00", "night_rollover_sat_sun_morning"),
    ("คืนวันอาทิตย์ ตีสอง", "2026-11-09T02:00:00+07:00", "night_rollover_sun_mon_morning"),
    ("คืนนี้ ตีสาม", "2026-11-09T03:00:00+07:00", "night_rollover_tonight"),
    ("คืนนี้ ตีสอง", "2026-11-09T02:00:00+07:00", "night_rollover_tonight"),
    ("คืนนี้ ตีสี่ครึ่ง", "2026-11-09T04:30:00+07:00", "night_rollover_tonight_half"),
    ("เมื่อคืน ตีสอง", "2026-11-08T02:00:00+07:00", "night_rollover_last_night"),
    ("ดึกวันจันทร์ ตีหนึ่งครึ่ง", "2026-11-10T01:30:00+07:00", "night_rollover_mon_tue_morning"),
    ("ดึกวันอังคาร ตีสอง", "2026-11-11T02:00:00+07:00", "night_rollover_tue_wed_morning"),
    ("ดึกวันพุธ ตีสาม", "2026-11-12T03:00:00+07:00", "night_rollover_wed_thu_morning"),
    ("ดึกวันพฤหัส ตีสี่", "2026-11-13T04:00:00+07:00", "night_rollover_thu_fri_morning"),
    ("คืนวันพฤหัส ตีสี่", "2026-11-13T04:00:00+07:00", "night_rollover_thu_fri_morning"),
]

# Expanded Bounds, Durations & Recurrence
BOUNDS_DEFINITIONS: List[Tuple[str, str, str, str, Optional[str], Optional[str], str]] = [
    ("ก่อนเที่ยง", "RANGE", "hour", "coarse", None, "2026-11-08T12:00:00+07:00", "bound_upper_noon"),
    ("ก่อนเที่ยงวัน", "RANGE", "hour", "coarse", None, "2026-11-08T12:00:00+07:00", "bound_upper_noon"),
    ("ก่อนบ่ายโมง", "RANGE", "hour", "exact", None, "2026-11-08T13:00:00+07:00", "bound_upper_afternoon"),
    ("ก่อนบ่ายสอง", "RANGE", "hour", "exact", None, "2026-11-08T14:00:00+07:00", "bound_upper_afternoon"),
    ("ก่อนบ่ายสาม", "RANGE", "hour", "exact", None, "2026-11-08T15:00:00+07:00", "bound_upper_afternoon"),
    ("ก่อน 4 โมงเย็น", "RANGE", "hour", "exact", None, "2026-11-08T16:00:00+07:00", "bound_upper_evening"),
    ("ก่อน 5 โมงเย็น", "RANGE", "hour", "exact", None, "2026-11-08T17:00:00+07:00", "bound_upper_evening"),
    ("ก่อนหกโมงเย็น", "RANGE", "hour", "exact", None, "2026-11-08T18:00:00+07:00", "bound_upper_evening"),
    ("ก่อนหนึ่งทุ่ม", "RANGE", "hour", "exact", None, "2026-11-08T19:00:00+07:00", "bound_upper_night"),
    ("ก่อนสองทุ่ม", "RANGE", "hour", "exact", None, "2026-11-08T20:00:00+07:00", "bound_upper_night"),
    ("ก่อนสามทุ่ม", "RANGE", "hour", "exact", None, "2026-11-08T21:00:00+07:00", "bound_upper_night"),
    ("ก่อนห้าทุ่ม", "RANGE", "hour", "exact", None, "2026-11-08T23:00:00+07:00", "bound_upper_night"),
    ("ก่อนเที่ยงคืน", "RANGE", "hour", "coarse", None, "2026-11-08T00:00:00+07:00", "bound_upper_midnight"),
    ("ก่อน 8 โมงเช้า", "RANGE", "hour", "exact", None, "2026-11-08T08:00:00+07:00", "bound_upper_morning"),
    ("ก่อนเก้าโมงเช้า", "RANGE", "hour", "exact", None, "2026-11-08T09:00:00+07:00", "bound_upper_morning"),
    ("ก่อน 10 โมง", "RANGE", "hour", "exact", None, "2026-11-08T10:00:00+07:00", "bound_upper_morning"),
    ("ก่อนสิ้นเดือน", "RANGE", "day", "coarse", None, "2026-11-30T23:59:59+07:00", "bound_upper_month_end"),
    ("ไม่เกินเที่ยง", "RANGE", "hour", "exact", None, "2026-11-08T12:00:00+07:00", "bound_limit_noon"),
    ("ไม่เกินบ่ายสอง", "RANGE", "hour", "exact", None, "2026-11-08T14:00:00+07:00", "bound_limit_afternoon"),
    ("ไม่เกินบ่ายสาม", "RANGE", "hour", "exact", None, "2026-11-08T15:00:00+07:00", "bound_limit_afternoon"),
    ("ไม่เกินห้าโมงเย็น", "RANGE", "hour", "exact", None, "2026-11-08T17:00:00+07:00", "bound_limit_evening"),
    ("ไม่เกิน 6 โมงเย็น", "RANGE", "hour", "exact", None, "2026-11-08T18:00:00+07:00", "bound_limit_evening"),
    ("ไม่เกิน 2 ทุ่ม", "RANGE", "hour", "exact", None, "2026-11-08T20:00:00+07:00", "bound_limit_night"),
    ("ไม่เกินสี่ทุ่ม", "RANGE", "hour", "exact", None, "2026-11-08T22:00:00+07:00", "bound_limit_night"),
    ("ไม่เกินห้าทุ่ม", "RANGE", "hour", "exact", None, "2026-11-08T23:00:00+07:00", "bound_limit_night"),
    ("ไม่เกินเที่ยงคืน", "RANGE", "hour", "exact", None, "2026-11-08T00:00:00+07:00", "bound_limit_midnight"),
    ("หลังเที่ยง", "RANGE", "hour", "coarse", "2026-11-08T12:00:00+07:00", None, "bound_lower_noon"),
    ("หลังบ่ายโมง", "RANGE", "hour", "exact", "2026-11-08T13:00:00+07:00", None, "bound_lower_afternoon"),
    ("หลังบ่ายสอง", "RANGE", "hour", "exact", "2026-11-08T14:00:00+07:00", None, "bound_lower_afternoon"),
    ("หลังบ่ายสาม", "RANGE", "hour", "exact", "2026-11-08T15:00:00+07:00", None, "bound_lower_afternoon"),
    ("หลังสี่โมงเย็น", "RANGE", "hour", "exact", "2026-11-08T16:00:00+07:00", None, "bound_lower_evening"),
    ("หลังหกโมงเย็น", "RANGE", "hour", "exact", "2026-11-08T18:00:00+07:00", None, "bound_lower_evening"),
    ("หลังสองทุ่ม", "RANGE", "hour", "exact", "2026-11-08T20:00:00+07:00", None, "bound_lower_night"),
    ("หลังเที่ยงคืน", "RANGE", "hour", "coarse", "2026-11-08T00:00:00+07:00", None, "bound_lower_midnight"),
    ("ตั้งแต่ 9 โมง ถึง เที่ยง", "RANGE", "hour", "exact", "2026-11-08T09:00:00+07:00", "2026-11-08T12:00:00+07:00", "bound_range_closed"),
    ("ตั้งแต่ 8 โมง ถึง 5 โมงเย็น", "RANGE", "hour", "exact", "2026-11-08T08:00:00+07:00", "2026-11-08T17:00:00+07:00", "bound_range_closed"),
    ("บ่ายโมงถึงบ่ายสาม", "RANGE", "hour", "exact", "2026-11-08T13:00:00+07:00", "2026-11-08T15:00:00+07:00", "bound_range_closed"),
    ("บ่ายสองถึงสี่โมงเย็น", "RANGE", "hour", "exact", "2026-11-08T14:00:00+07:00", "2026-11-08T16:00:00+07:00", "bound_range_closed"),
    ("10:00 - 12:00 น.", "RANGE", "minute", "exact", "2026-11-08T10:00:00+07:00", "2026-11-08T12:00:00+07:00", "bound_range_closed"),
    ("13:30 - 15:00 น.", "RANGE", "minute", "exact", "2026-11-08T13:30:00+07:00", "2026-11-08T15:00:00+07:00", "bound_range_closed"),
]

DURATIONS_DEFINITIONS: List[Tuple[str, str, str, str, str, str]] = [
    ("อีก 5 นาที", "DURATION", "minute", "PT5M", "exact", "duration_minute"),
    ("อีก 10 นาที", "DURATION", "minute", "PT10M", "exact", "duration_minute"),
    ("อีก 15 นาที", "DURATION", "minute", "PT15M", "exact", "duration_minute"),
    ("อีก 20 นาที", "DURATION", "minute", "PT20M", "exact", "duration_minute"),
    ("อีก 30 นาที", "DURATION", "minute", "PT30M", "exact", "duration_minute"),
    ("อีก 45 นาที", "DURATION", "minute", "PT45M", "exact", "duration_minute"),
    ("ในอีก 15 นาที", "DURATION", "minute", "PT15M", "exact", "duration_minute"),
    ("ในอีก 30 นาที", "DURATION", "minute", "PT30M", "exact", "duration_minute"),
    ("อีกครึ่งชั่วโมง", "DURATION", "minute", "PT30M", "exact", "duration_minute"),
    ("อีก 1 ชั่วโมง", "DURATION", "hour", "PT1H", "exact", "duration_hour"),
    ("อีก 2 ชั่วโมง", "DURATION", "hour", "PT2H", "exact", "duration_hour"),
    ("อีก 3 ชั่วโมง", "DURATION", "hour", "PT3H", "exact", "duration_hour"),
    ("ในอีก 2 ชั่วโมง", "DURATION", "hour", "PT2H", "exact", "duration_hour"),
    ("เป็นเวลา 15 นาที", "DURATION", "minute", "PT15M", "exact", "duration_span"),
    ("เป็นเวลา 30 นาที", "DURATION", "minute", "PT30M", "exact", "duration_span"),
    ("เป็นเวลา 1 ชั่วโมง", "DURATION", "hour", "PT1H", "exact", "duration_span"),
    ("เป็นเวลา 2 ชั่วโมง", "DURATION", "hour", "PT2H", "exact", "duration_span"),
    ("เป็นเวลา 3 วัน", "DURATION", "day", "P3D", "exact", "duration_span_day"),
    ("เป็นเวลา 5 วัน", "DURATION", "day", "P5D", "exact", "duration_span_day"),
    ("ภายใน 3 วัน", "DURATION", "day", "P3D", "exact", "duration_span_day"),
    ("ภายใน 5 วัน", "DURATION", "day", "P5D", "exact", "duration_span_day"),
    ("ภายใน 7 วัน", "DURATION", "day", "P7D", "exact", "duration_span_day"),
    ("ระยะเวลา 1 สัปดาห์", "DURATION", "day", "P1W", "exact", "duration_span_week"),
    ("ระยะเวลา 2 สัปดาห์", "DURATION", "day", "P2W", "exact", "duration_span_week"),
]

RECURRENCE_DEFINITIONS: List[Tuple[str, str, str, str, str, str]] = [
    ("ทุกวัน", "RECURRENCE", "day", "FREQ=DAILY", "exact", "recurrence_daily"),
    ("ทุกๆ วัน", "RECURRENCE", "day", "FREQ=DAILY", "exact", "recurrence_daily"),
    ("ทุกวันจันทร์", "RECURRENCE", "day", "FREQ=WEEKLY;BYDAY=MO", "exact", "recurrence_weekly"),
    ("ทุกวันอังคาร", "RECURRENCE", "day", "FREQ=WEEKLY;BYDAY=TU", "exact", "recurrence_weekly"),
    ("ทุกวันพุธ", "RECURRENCE", "day", "FREQ=WEEKLY;BYDAY=WE", "exact", "recurrence_weekly"),
    ("ทุกวันพฤหัสบดี", "RECURRENCE", "day", "FREQ=WEEKLY;BYDAY=TH", "exact", "recurrence_weekly"),
    ("ทุกวันศุกร์", "RECURRENCE", "day", "FREQ=WEEKLY;BYDAY=FR", "exact", "recurrence_weekly"),
    ("ทุกวันเสาร์", "RECURRENCE", "day", "FREQ=WEEKLY;BYDAY=SA", "exact", "recurrence_weekly"),
    ("ทุกวันอาทิตย์", "RECURRENCE", "day", "FREQ=WEEKLY;BYDAY=SU", "exact", "recurrence_weekly"),
    ("ทุกจันทร์", "RECURRENCE", "day", "FREQ=WEEKLY;BYDAY=MO", "exact", "recurrence_weekly"),
    ("ทุกศุกร์", "RECURRENCE", "day", "FREQ=WEEKLY;BYDAY=FR", "exact", "recurrence_weekly"),
    ("ทุกวันเสาร์-อาทิตย์", "RECURRENCE", "day", "FREQ=WEEKLY;BYDAY=SA,SU", "exact", "recurrence_weekend"),
    ("ทุกสุดสัปดาห์", "RECURRENCE", "day", "FREQ=WEEKLY;BYDAY=SA,SU", "exact", "recurrence_weekend"),
    ("ทุกวันธรรมดา", "RECURRENCE", "day", "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR", "exact", "recurrence_weekday"),
    ("ทุกเช้า", "RECURRENCE", "hour", "FREQ=DAILY;BYHOUR=08", "coarse", "recurrence_time_daily"),
    ("ทุกแปดโมงเช้า", "RECURRENCE", "hour", "FREQ=DAILY;BYHOUR=08;BYMINUTE=00", "exact", "recurrence_time_daily"),
    ("ทุกเก้าโมง", "RECURRENCE", "hour", "FREQ=DAILY;BYHOUR=09;BYMINUTE=00", "exact", "recurrence_time_daily"),
    ("ทุกเที่ยง", "RECURRENCE", "hour", "FREQ=DAILY;BYHOUR=12;BYMINUTE=00", "exact", "recurrence_time_daily"),
    ("ทุกบ่ายโมง", "RECURRENCE", "hour", "FREQ=DAILY;BYHOUR=13;BYMINUTE=00", "exact", "recurrence_time_daily"),
    ("ทุกบ่ายสอง", "RECURRENCE", "hour", "FREQ=DAILY;BYHOUR=14;BYMINUTE=00", "exact", "recurrence_time_daily"),
    ("ทุกบ่ายสาม", "RECURRENCE", "hour", "FREQ=DAILY;BYHOUR=15;BYMINUTE=00", "exact", "recurrence_time_daily"),
    ("ทุกเย็น", "RECURRENCE", "hour", "FREQ=DAILY;BYHOUR=17", "coarse", "recurrence_time_daily"),
    ("ทุกสิ้นเดือน", "RECURRENCE", "day", "FREQ=MONTHLY;BYMONTHDAY=-1", "exact", "recurrence_monthly"),
    ("ทุกต้นเดือน", "RECURRENCE", "day", "FREQ=MONTHLY;BYMONTHDAY=1", "exact", "recurrence_monthly"),
    ("ทุกวันที่ 1 ของเดือน", "RECURRENCE", "day", "FREQ=MONTHLY;BYMONTHDAY=1", "exact", "recurrence_monthly"),
    ("ทุกวันที่ 15 ของเดือน", "RECURRENCE", "day", "FREQ=MONTHLY;BYMONTHDAY=15", "exact", "recurrence_monthly"),
    ("ทุกสัปดาห์", "RECURRENCE", "day", "FREQ=WEEKLY", "exact", "recurrence_weekly"),
    ("ทุกเดือน", "RECURRENCE", "day", "FREQ=MONTHLY", "exact", "recurrence_monthly"),
]

# Chat Colloquialisms & Unspaced Times
CHAT_ABBREVIATIONS: List[Tuple[str, str, str, str, str]] = [
    ("พน", "POINT", "day", "2026-11-09T12:00:00+07:00", "chat_abbrev_tomorrow"),
    ("พน.", "POINT", "day", "2026-11-09T12:00:00+07:00", "chat_abbrev_tomorrow"),
    ("มรน", "POINT", "day", "2026-11-10T12:00:00+07:00", "chat_abbrev_overmorrow"),
    ("มรน.", "POINT", "day", "2026-11-10T12:00:00+07:00", "chat_abbrev_overmorrow"),
    ("มะรืน", "POINT", "day", "2026-11-10T12:00:00+07:00", "chat_colloquial_day"),
    ("วานนี้", "POINT", "day", "2026-11-07T12:00:00+07:00", "chat_colloquial_yesterday"),
    ("ทุ่มนึง", "POINT", "hour", "2026-11-08T19:00:00+07:00", "chat_slang_clock"),
    ("ทุ่มตรง", "POINT", "hour", "2026-11-08T19:00:00+07:00", "chat_slang_clock"),
    ("เที่ยงเป๊ะ", "POINT", "hour", "2026-11-08T12:00:00+07:00", "chat_slang_clock"),
    ("เที่ยงคืนเป๊ะ", "POINT", "hour", "2026-11-08T00:00:00+07:00", "chat_slang_clock"),
]

UNSPACED_CLOCK_RAW: List[Tuple[str, int, int, str]] = [
    ("7โมง", 7, 0, "unspaced_clock_morning"),
    ("7โมงเช้า", 7, 0, "unspaced_clock_morning"),
    ("8โมง", 8, 0, "unspaced_clock_morning"),
    ("8โมงเช้า", 8, 0, "unspaced_clock_morning"),
    ("8โมงครึ่ง", 8, 30, "unspaced_clock_morning"),
    ("9โมง", 9, 0, "unspaced_clock_morning"),
    ("9โมงเช้า", 9, 0, "unspaced_clock_morning"),
    ("9โมงครึ่ง", 9, 30, "unspaced_clock_morning"),
    ("10โมง", 10, 0, "unspaced_clock_morning"),
    ("10โมงเช้า", 10, 0, "unspaced_clock_morning"),
    ("10โมงครึ่ง", 10, 30, "unspaced_clock_morning"),
    ("11โมง", 11, 0, "unspaced_clock_morning"),
    ("11โมงครึ่ง", 11, 30, "unspaced_clock_morning"),
    ("บ่าย1", 13, 0, "unspaced_clock_afternoon"),
    ("บ่าย1โมง", 13, 0, "unspaced_clock_afternoon"),
    ("บ่าย1ครึ่ง", 13, 30, "unspaced_clock_afternoon"),
    ("บ่าย2", 14, 0, "unspaced_clock_afternoon"),
    ("บ่าย2โมง", 14, 0, "unspaced_clock_afternoon"),
    ("บ่าย2ครึ่ง", 14, 30, "unspaced_clock_afternoon"),
    ("บ่าย3", 15, 0, "unspaced_clock_afternoon"),
    ("บ่าย3โมง", 15, 0, "unspaced_clock_afternoon"),
    ("บ่าย3ครึ่ง", 15, 30, "unspaced_clock_afternoon"),
    ("บ่าย4", 16, 0, "unspaced_clock_afternoon"),
    ("บ่าย4โมง", 16, 0, "unspaced_clock_afternoon"),
    ("4โมงเย็น", 16, 0, "unspaced_clock_evening"),
    ("4โมงครึ่ง", 16, 30, "unspaced_clock_evening"),
    ("5โมงเย็น", 17, 0, "unspaced_clock_evening"),
    ("5โมงครึ่ง", 17, 30, "unspaced_clock_evening"),
    ("6โมงเย็น", 18, 0, "unspaced_clock_evening"),
    ("6โมงครึ่ง", 18, 30, "unspaced_clock_evening"),
    ("1ทุ่ม", 19, 0, "unspaced_clock_night"),
    ("1ทุ่มตรง", 19, 0, "unspaced_clock_night"),
    ("1ทุ่มครึ่ง", 19, 30, "unspaced_clock_night"),
    ("2ทุ่ม", 20, 0, "unspaced_clock_night"),
    ("2ทุ่มตรง", 20, 0, "unspaced_clock_night"),
    ("2ทุ่มครึ่ง", 20, 30, "unspaced_clock_night"),
    ("3ทุ่ม", 21, 0, "unspaced_clock_night"),
    ("3ทุ่มตรง", 21, 0, "unspaced_clock_night"),
    ("3ทุ่มครึ่ง", 21, 30, "unspaced_clock_night"),
    ("4ทุ่ม", 22, 0, "unspaced_clock_night"),
    ("4ทุ่มตรง", 22, 0, "unspaced_clock_night"),
    ("4ทุ่มครึ่ง", 22, 30, "unspaced_clock_night"),
    ("5ทุ่ม", 23, 0, "unspaced_clock_night"),
    ("5ทุ่มตรง", 23, 0, "unspaced_clock_night"),
    ("5ทุ่มครึ่ง", 23, 30, "unspaced_clock_night"),
    ("ตี1", 1, 0, "unspaced_clock_early"),
    ("ตี2", 2, 0, "unspaced_clock_early"),
    ("ตี3", 3, 0, "unspaced_clock_early"),
    ("ตี4", 4, 0, "unspaced_clock_early"),
    ("ตี5", 5, 0, "unspaced_clock_early"),
]

CHAT_COMPOUNDS: List[Tuple[str, str, str]] = [
    ("พน 10โมง", "2026-11-09T10:00:00+07:00", "chat_compound_rel_clock"),
    ("พน บ่าย2", "2026-11-09T14:00:00+07:00", "chat_compound_rel_clock"),
    ("พน 2ทุ่ม", "2026-11-09T20:00:00+07:00", "chat_compound_rel_clock"),
    ("มรน บ่าย3", "2026-11-10T15:00:00+07:00", "chat_compound_rel_clock"),
    ("วันนี้ 4ทุ่ม", "2026-11-08T22:00:00+07:00", "chat_compound_rel_clock"),
    ("คืนนี้ 5ทุ่ม", "2026-11-08T23:00:00+07:00", "chat_compound_rel_clock"),
    ("พรุ่งนี้ บ่าย3", "2026-11-09T15:00:00+07:00", "chat_compound_rel_clock"),
    ("เมื่อวาน 2ทุ่ม", "2026-11-07T20:00:00+07:00", "chat_compound_rel_clock"),
    ("ศุกร์นี้ 1ทุ่ม", "2026-11-13T19:00:00+07:00", "chat_compound_rel_clock"),
    ("เสาร์หน้า 3ทุ่ม", "2026-11-14T21:00:00+07:00", "chat_compound_rel_clock"),
]

THAI_MONTHS_SHORT = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]
THAI_MONTHS_FULL = [
    "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"
]
BE_YEAR_MAP = {
    "67": 2024, "68": 2025, "69": 2026, "70": 2027,
    "2567": 2024, "2568": 2025, "2569": 2026, "2570": 2027
}


# =============================================================================
# Entity Generators
# =============================================================================

def generate_standard_entity(rng: random.Random) -> Dict[str, Any]:
    """Sample a standard temporal candidate entity (~70% of dataset)."""
    sub_category = rng.choices(
        ["relative_day_week", "thai_6hour_clock", "standard_24h", "coarse_period", "buddhist_era", "night_rollover", "compound"],
        weights=[20, 25, 15, 10, 15, 5, 10],
        k=1,
    )[0]

    if sub_category == "relative_day_week":
        mode = rng.choice(["deictic", "weekday", "numeric"])
        if mode == "deictic":
            span, day_offset, ling = rng.choice(RELATIVE_DAYS)
        elif mode == "weekday":
            span, day_offset, ling = rng.choice(WEEKDAY_OFFSETS)
        else:
            span, day_offset = rng.choice(NUMERIC_DAY_OFFSETS)
            ling = "numeric_offset"

        target_dt = DEFAULT_REF_TIME + timedelta(days=day_offset)
        iso_val = target_dt.replace(hour=12, minute=0, second=0).isoformat()
        return {
            "text_span": span,
            "type": "POINT",
            "grain": "day",
            "iso_value": iso_val,
            "range_start": None,
            "range_end": None,
            "precision": "exact",
            "confidence": 1.0,
            "category": "relative_day_week",
            "linguistic_type": ling,
            "has_midnight_crossing": False,
            "notes": f"Relative day/week offset {day_offset} days",
        }

    elif sub_category == "thai_6hour_clock":
        clock_sub = rng.choice([
            CLOCK_MORNING, CLOCK_NOON, CLOCK_AFTERNOON,
            CLOCK_EVENING, CLOCK_NIGHT, CLOCK_MIDNIGHT, CLOCK_EARLY_MORNING
        ])
        span, h, m, ling = rng.choice(clock_sub)
        target_dt = DEFAULT_REF_TIME.replace(hour=h, minute=m, second=0)
        grain = "minute" if m != 0 else "hour"
        return {
            "text_span": span,
            "type": "POINT",
            "grain": grain,
            "iso_value": target_dt.isoformat(),
            "range_start": None,
            "range_end": None,
            "precision": "exact",
            "confidence": 1.0,
            "category": "thai_6hour_clock",
            "linguistic_type": ling,
            "has_midnight_crossing": False,
            "notes": f"Thai clock time {h:02d}:{m:02d}",
        }

    elif sub_category == "standard_24h":
        h = rng.randint(0, 23)
        m = rng.choice([0, 15, 30, 45, 10, 20, 40, 50])
        fmt = rng.choice(["colon", "dot"])
        sep = ":" if fmt == "colon" else "."
        span = f"{h:02d}{sep}{m:02d} น."
        target_dt = DEFAULT_REF_TIME.replace(hour=h, minute=m, second=0)
        return {
            "text_span": span,
            "type": "POINT",
            "grain": "minute" if m != 0 else "hour",
            "iso_value": target_dt.isoformat(),
            "range_start": None,
            "range_end": None,
            "precision": "exact",
            "confidence": 1.0,
            "category": "standard_24h_time",
            "linguistic_type": "standard_24h_digital",
            "has_midnight_crossing": False,
            "notes": f"Standard 24-hour clock {h:02d}:{m:02d}",
        }

    elif sub_category == "coarse_period":
        span, s_time, e_time, day_shift, is_mid, ling = rng.choice(COARSE_PERIODS)
        s_h, s_m, s_s = map(int, s_time.split(":"))
        e_h, e_m, e_s = map(int, e_time.split(":"))
        dt_s = DEFAULT_REF_TIME.replace(hour=s_h, minute=s_m, second=s_s)
        dt_e = (DEFAULT_REF_TIME + timedelta(days=day_shift)).replace(hour=e_h, minute=e_m, second=e_s)
        return {
            "text_span": span,
            "type": "RANGE",
            "grain": "auto",
            "iso_value": None,
            "range_start": dt_s.isoformat(),
            "range_end": dt_e.isoformat(),
            "precision": "coarse",
            "confidence": 0.80 if not is_mid else 0.75,
            "category": "coarse_time_periods",
            "linguistic_type": ling,
            "has_midnight_crossing": is_mid,
            "notes": f"Coarse range {s_time} to {e_time}",
        }

    elif sub_category == "buddhist_era":
        pick = rng.choice(["fixed_year", "fixed_date", "dynamic_date"])
        if pick == "fixed_year":
            span, ce_yr, ling = rng.choice(BUDDHIST_ERA_YEARS)
            if span.startswith("ต้นปี"):
                return {
                    "text_span": span,
                    "type": "RANGE",
                    "grain": "auto",
                    "iso_value": None,
                    "range_start": f"{ce_yr:04d}-01-01T00:00:00+07:00",
                    "range_end": f"{ce_yr:04d}-04-30T23:59:59+07:00",
                    "precision": "coarse",
                    "confidence": 0.85,
                    "category": "short_buddhist_era",
                    "linguistic_type": ling,
                    "has_midnight_crossing": False,
                    "notes": f"Buddhist Era early-year range {span}",
                }
            elif span.startswith("กลางปี"):
                return {
                    "text_span": span,
                    "type": "RANGE",
                    "grain": "auto",
                    "iso_value": None,
                    "range_start": f"{ce_yr:04d}-05-01T00:00:00+07:00",
                    "range_end": f"{ce_yr:04d}-08-31T23:59:59+07:00",
                    "precision": "coarse",
                    "confidence": 0.85,
                    "category": "short_buddhist_era",
                    "linguistic_type": ling,
                    "has_midnight_crossing": False,
                    "notes": f"Buddhist Era mid-year range {span}",
                }
            elif span.startswith("ปลายปี"):
                return {
                    "text_span": span,
                    "type": "RANGE",
                    "grain": "auto",
                    "iso_value": None,
                    "range_start": f"{ce_yr:04d}-09-01T00:00:00+07:00",
                    "range_end": f"{ce_yr:04d}-12-31T23:59:59+07:00",
                    "precision": "coarse",
                    "confidence": 0.85,
                    "category": "short_buddhist_era",
                    "linguistic_type": ling,
                    "has_midnight_crossing": False,
                    "notes": f"Buddhist Era late-year range {span}",
                }
            else:
                return {
                    "text_span": span,
                    "type": "POINT",
                    "grain": "year",
                    "iso_value": f"{ce_yr:04d}-01-01T00:00:00+07:00",
                    "range_start": None,
                    "range_end": None,
                    "precision": "exact",
                    "confidence": 1.0,
                    "category": "short_buddhist_era",
                    "linguistic_type": ling,
                    "has_midnight_crossing": False,
                    "notes": f"Buddhist Era expression {span}",
                }
        elif pick == "fixed_date":
            span, iso_val, ling = rng.choice(BUDDHIST_ERA_DATES)
            return {
                "text_span": span,
                "type": "POINT",
                "grain": "day",
                "iso_value": iso_val,
                "range_start": None,
                "range_end": None,
                "precision": "exact",
                "confidence": 1.0,
                "category": "short_buddhist_era",
                "linguistic_type": ling,
                "has_midnight_crossing": False,
                "notes": f"Buddhist Era expression {span}",
            }
        else:
            day = rng.randint(1, 28)
            month_idx = rng.randint(1, 12)
            use_short_m = rng.choice([True, False])
            m_str = THAI_MONTHS_SHORT[month_idx - 1] if use_short_m else THAI_MONTHS_FULL[month_idx - 1]
            y_key, ce_yr = rng.choice(list(BE_YEAR_MAP.items()))
            prefix_fmt = rng.choice(["วันที่ {d} {m} {y}", "{d} {m} {y}"])
            span = prefix_fmt.format(d=day, m=m_str, y=y_key)
            target_dt = datetime(ce_yr, month_idx, day, 12, 0, 0, tzinfo=BANGKOK_TZ)
            return {
                "text_span": span,
                "type": "POINT",
                "grain": "day",
                "iso_value": target_dt.isoformat(),
                "range_start": None,
                "range_end": None,
                "precision": "exact",
                "confidence": 1.0,
                "category": "short_buddhist_era",
                "linguistic_type": "be_date_dynamic",
                "has_midnight_crossing": False,
                "notes": f"Dynamic Buddhist Era date {span}",
            }

    elif sub_category == "night_rollover":
        span, iso_val, ling = rng.choice(NIGHT_ROLLOVERS)
        return {
            "text_span": span,
            "type": "POINT",
            "grain": "minute",
            "iso_value": iso_val,
            "range_start": None,
            "range_end": None,
            "precision": "exact",
            "confidence": 0.95,
            "category": "night_date_rollover",
            "linguistic_type": ling,
            "has_midnight_crossing": True,
            "notes": f"Night-time day rollover {span}",
        }

    else:  # compound
        day_span, day_offset, _ = rng.choice(RELATIVE_DAYS[:4])
        clock_sub = rng.choice([CLOCK_MORNING, CLOCK_NOON, CLOCK_AFTERNOON, CLOCK_EVENING, CLOCK_NIGHT])
        c_span, h, m, _ = rng.choice(clock_sub)
        span = f"{day_span} {c_span}"
        target_dt = (DEFAULT_REF_TIME + timedelta(days=day_offset)).replace(hour=h, minute=m, second=0)
        return {
            "text_span": span,
            "type": "POINT",
            "grain": "minute" if m != 0 else "hour",
            "iso_value": target_dt.isoformat(),
            "range_start": None,
            "range_end": None,
            "precision": "exact",
            "confidence": 0.95,
            "category": "compound_datetime",
            "linguistic_type": "compound_relative_clock",
            "has_midnight_crossing": False,
            "notes": f"Compound relative day + clock {span}",
        }


def generate_bound_duration_recurrence_entity(rng: random.Random) -> Dict[str, Any]:
    """Sample an entity from Bounds, Durations, or Recurrence (~15% of dataset)."""
    sub_type = rng.choices(["bound", "duration", "recurrence"], weights=[45, 30, 25], k=1)[0]

    if sub_type == "bound":
        span, typ, gr, prec, rs, re, ling = rng.choice(BOUNDS_DEFINITIONS)
        return {
            "text_span": span,
            "type": typ,
            "grain": gr,
            "iso_value": None,
            "range_start": rs,
            "range_end": re,
            "precision": prec,
            "confidence": 0.95,
            "category": "bound_duration_recurrence",
            "linguistic_type": ling,
            "has_midnight_crossing": False,
            "notes": f"Temporal bound expression {span}",
        }

    elif sub_type == "duration":
        span, typ, gr, iso, prec, ling = rng.choice(DURATIONS_DEFINITIONS)
        return {
            "text_span": span,
            "type": typ,
            "grain": gr,
            "iso_value": iso,
            "range_start": None,
            "range_end": None,
            "precision": prec,
            "confidence": 0.95,
            "category": "bound_duration_recurrence",
            "linguistic_type": ling,
            "has_midnight_crossing": False,
            "notes": f"Relative duration expression {span}",
        }

    else:  # recurrence
        span, typ, gr, iso, prec, ling = rng.choice(RECURRENCE_DEFINITIONS)
        return {
            "text_span": span,
            "type": typ,
            "grain": gr,
            "iso_value": iso,
            "range_start": None,
            "range_end": None,
            "precision": prec,
            "confidence": 0.95,
            "category": "bound_duration_recurrence",
            "linguistic_type": ling,
            "has_midnight_crossing": False,
            "notes": f"Recurrence pattern {span}",
        }


def generate_chat_colloquial_entity(rng: random.Random) -> Dict[str, Any]:
    """Sample an entity from Chat colloquialisms & unspaced times (~5% of dataset)."""
    sub_type = rng.choices(["abbreviation", "unspaced_clock", "compound"], weights=[30, 50, 20], k=1)[0]

    if sub_type == "abbreviation":
        span, typ, gr, iso, ling = rng.choice(CHAT_ABBREVIATIONS)
        return {
            "text_span": span,
            "type": typ,
            "grain": gr,
            "iso_value": iso,
            "range_start": None,
            "range_end": None,
            "precision": "exact",
            "confidence": 0.90,
            "category": "chat_colloquial",
            "linguistic_type": ling,
            "has_midnight_crossing": False,
            "notes": f"Chat colloquial abbreviation {span}",
        }

    elif sub_type == "unspaced_clock":
        span, h, m, ling = rng.choice(UNSPACED_CLOCK_RAW)
        target_dt = DEFAULT_REF_TIME.replace(hour=h, minute=m, second=0)
        return {
            "text_span": span,
            "type": "POINT",
            "grain": "minute" if m != 0 else "hour",
            "iso_value": target_dt.isoformat(),
            "range_start": None,
            "range_end": None,
            "precision": "exact",
            "confidence": 0.90,
            "category": "chat_colloquial",
            "linguistic_type": ling,
            "has_midnight_crossing": False,
            "notes": f"Unspaced Thai clock expression {span}",
        }

    else:  # compound
        span, iso, ling = rng.choice(CHAT_COMPOUNDS)
        return {
            "text_span": span,
            "type": "POINT",
            "grain": "minute",
            "iso_value": iso,
            "range_start": None,
            "range_end": None,
            "precision": "exact",
            "confidence": 0.90,
            "category": "chat_colloquial",
            "linguistic_type": ling,
            "has_midnight_crossing": False,
            "notes": f"Chat compound pairing {span}",
        }


def generate_negative_sample(rng: random.Random, split_name: str, item_id: Any) -> Dict[str, Any]:
    """Generate a non-temporal distractor sentence with ground_truth: null (~10% of dataset)."""
    neg_type = rng.choices(
        ["currency", "tracking_code", "phone_number", "general_text"],
        weights=[25, 25, 25, 25],
        k=1,
    )[0]

    if neg_type == "currency":
        tmpl = rng.choice(NEGATIVE_TEMPLATES["currency"])
        amt_val = rng.choice([
            50, 99, 100, 150, 200, 250, 300, 350, 450, 500, 750, 990, 1000,
            1250, 1500, 1800, 2000, 2500, 3200, 4500, 5000, 8900, 9990,
            12500, 15000, 25000, 35000, 50000, 100000, rng.randint(20, 50000)
        ])
        fmt = rng.choice(["plain", "comma", "decimal"])
        if fmt == "comma":
            amt_str = f"{amt_val:,}"
        elif fmt == "decimal":
            amt_str = f"{amt_val:,.2f}"
        else:
            amt_str = str(amt_val)

        unit = rng.choice(["บาท", "บาทถ้วน", "บ.", "THB"])
        raw_input = tmpl.format(amount=amt_str, unit=unit)
        domain = "financial_banking"

    elif neg_type == "tracking_code":
        tmpl = rng.choice(NEGATIVE_TEMPLATES["tracking_code"])
        prefix = rng.choice(["TH", "KEX", "FLASH", "SPX", "POST", "EMS", "JNT", "SHIP"])
        code = f"{prefix}{rng.randint(100000000, 9999999999)}"
        raw_input = tmpl.format(code=code)
        domain = "logistics_delivery"

    elif neg_type == "phone_number":
        tmpl = rng.choice(NEGATIVE_TEMPLATES["phone_number"])
        p_type = rng.choice(["mobile", "landline"])
        if p_type == "mobile":
            p_prefix = rng.choice(["081", "082", "084", "085", "086", "089", "090", "092", "095", "098", "061", "062"])
            phone = f"{p_prefix}-{rng.randint(100, 999):03d}-{rng.randint(1000, 9999):04d}"
        else:
            phone = f"02-{rng.randint(100, 999):03d}-{rng.randint(1000, 9999):04d}"
        raw_input = tmpl.format(phone=phone)
        domain = "customer_service"

    else:  # general_text
        tmpl = rng.choice(NEGATIVE_TEMPLATES["general_text"])
        otp = f"{rng.randint(100000, 999999)}"
        room = f"{rng.randint(101, 809)}"
        floor = f"{rng.randint(1, 35)}"
        pct = f"{rng.randint(70, 99)}"
        qty = f"{rng.randint(2, 50)}"
        raw_input = tmpl.format(otp=otp, room=room, floor=floor, pct=pct, qty=qty)
        domain = "general_daily"

    return {
        "id": item_id,
        "raw_input": raw_input,
        "ground_truth": None,
        "category": "negative_sample",
        "metadata": {
            "split": split_name,
            "category": "negative_sample",
            "negative_type": neg_type,
            "semantic_domain": domain,
            "reference_time": REF_TIME_STR,
            "timezone": "Asia/Bangkok",
            "linguistic_type": f"negative_{neg_type}",
            "has_midnight_crossing": False,
        },
        "notes": f"Negative sample: non-temporal {neg_type} expression",
    }


# =============================================================================
# Split Generation Pipeline (Release 10k)
# =============================================================================

def generate_positive_item(
    item_id: Any,
    split_name: str,
    target_category: str,
    template: SentenceTemplate,
    cand: Dict[str, Any],
    start_char: int,
    end_char: int,
    raw_input: str,
) -> Dict[str, Any]:
    """Construct a validated positive record conforming to the project schema."""
    # Compute primary normalized value
    if cand["type"] == "POINT":
        norm_val = cand.get("iso_value")
    elif cand["type"] == "RANGE":
        rs = cand.get("range_start")
        re = cand.get("range_end")
        if rs and re:
            norm_val = f"{rs}/{re}"
        elif rs:
            norm_val = f"{rs}/.."
        elif re:
            norm_val = f"../{re}"
        else:
            norm_val = None
    elif cand["type"] in ("DURATION", "RECURRENCE"):
        norm_val = cand.get("iso_value")
    else:
        norm_val = cand.get("iso_value")

    return {
        "id": item_id,
        "raw_input": raw_input,
        "ground_truth": {
            "text_span": cand["text_span"],
            "start_char": start_char,
            "end_char": end_char,
            "type": cand["type"],
            "normalized_value": norm_val,
            "grain": cand["grain"],
            "iso_value": cand.get("iso_value"),
            "range_start": cand.get("range_start"),
            "range_end": cand.get("range_end"),
            "precision": cand["precision"],
            "confidence": cand["confidence"],
        },
        "category": cand["category"],
        "metadata": {
            "split": split_name,
            "category": target_category,
            "semantic_domain": template.domain.value.lower(),
            "reference_time": REF_TIME_STR,
            "timezone": "Asia/Bangkok",
            "linguistic_type": cand["linguistic_type"],
            "has_midnight_crossing": cand["has_midnight_crossing"],
        },
        "notes": f"{cand['notes']} [tmpl: {template.template_id}]",
    }


def generate_split_v1_1(
    split_name: str,
    count: int,
    registry: TemplateRegistry,
    deduplicator: CrossSplitDeduplicator,
    rng: random.Random,
    max_retries_per_item: int = 1000,
) -> List[Dict[str, Any]]:
    """Generate a single dataset split adhering to the 70/15/10/5 stratification with global deduplication."""
    n_standard = int(round(count * 0.70))
    n_bounds = int(round(count * 0.15))
    n_negatives = int(round(count * 0.10))
    n_chat = count - (n_standard + n_bounds + n_negatives)  # remaining ~5%

    cat_list = (
        ["standard"] * n_standard
        + ["bound_duration_recurrence"] * n_bounds
        + ["negative_sample"] * n_negatives
        + ["chat_colloquial"] * n_chat
    )
    rng.shuffle(cat_list)
    assert len(cat_list) == count

    records: List[Dict[str, Any]] = []

    for idx, target_cat in enumerate(cat_list, start=1):
        item_id = f"{split_name}_{idx:04d}"
        retries = 0
        accepted = False

        while not accepted:
            if retries >= max_retries_per_item:
                raise RuntimeError(
                    f"Deduplication retry budget exhausted ({max_retries_per_item}) for split='{split_name}', cat='{target_cat}'"
                )

            if target_cat == "negative_sample":
                candidate_rec = generate_negative_sample(rng, split_name, item_id)
                raw_text = candidate_rec["raw_input"].strip()
                if deduplicator.register(raw_text, split_name):
                    candidate_rec["raw_input"] = raw_text
                    records.append(candidate_rec)
                    accepted = True
                else:
                    retries += 1
                continue

            # Positive sample generation
            if split_name == "test_ood":
                # For OOD test split, prioritize OOD_DISJOINT_NOISE templates
                if retries < 15:
                    template = rng.choice(registry.get_ood_templates())
                else:
                    template = rng.choice(registry.templates)
            else:
                # In-distribution splits use the 120 in-distribution templates
                if target_cat == "chat_colloquial":
                    template = rng.choice(
                        registry.get_templates_by_domain(ContextDomain.PERSONAL_CHAT)
                        or registry.get_in_distribution_templates()
                    )
                else:
                    template = rng.choice(registry.get_in_distribution_templates())

            # Sample candidate entity
            if target_cat == "standard":
                cand = generate_standard_entity(rng)
            elif target_cat == "bound_duration_recurrence":
                cand = generate_bound_duration_recurrence_entity(rng)
            else:  # chat_colloquial
                cand = generate_chat_colloquial_entity(rng)

            # Check diurnal semantic compatibility
            if not SemanticTimeFilter.is_compatible(template.domain, cand):
                retries += 1
                continue

            # Compute exact span algebraically
            raw_input, start_char, end_char = AlgebraicSpanCalculator.instantiate(
                template.pattern, cand["text_span"]
            )

            # Enforce project length SLA (<= 120 chars)
            if len(raw_input) > 120:
                retries += 1
                continue

            # Check global deduplication
            raw_clean = raw_input.strip()
            if deduplicator.register(raw_clean, split_name):
                rec = generate_positive_item(
                    item_id=item_id,
                    split_name=split_name,
                    target_category=target_cat,
                    template=template,
                    cand=cand,
                    start_char=start_char,
                    end_char=end_char,
                    raw_input=raw_clean,
                )
                records.append(rec)
                accepted = True
            else:
                retries += 1

    return records


# =============================================================================
# Legacy Baseline 3k Reproducibility Pipeline
# =============================================================================

STD_TEMPLATES_LEGACY = [
    ("{entity}", "standalone"),
    ("นัดคุยงาน {entity}", "meeting"),
    ("เจอกันที่ร้าน {entity} นะ", "casual_meet"),
    ("ส่งเอกสารสัญญา {entity}", "document_submission"),
    ("มีประชุมแผนงาน {entity}", "meeting"),
    ("หมอนัดตรวจสุขภาพ {entity}", "appointment"),
    ("เริ่มเปิดรับสมัครงาน {entity}", "announcement"),
    ("กำหนดส่งรายงาน {entity}", "deadline"),
    ("จองตั๋วเครื่องบินเดินทาง {entity}", "booking"),
    ("สัมภาษณ์รอบไฟนอล {entity}", "interview"),
    ("นัดช่างเข้ามาตรวจเช็ค {entity}", "appointment"),
    ("ปิดยอดขายประจำเดือน {entity}", "business"),
    ("ชำระค่าบริการ {entity}", "payment"),
    ("เริ่มอบรมพนักงานใหม่ {entity}", "training"),
    ("แจ้งเตือนนัดหมาย {entity}", "reminder"),
    ("พร้อมจัดส่งสินค้า {entity}", "delivery"),
    ("มีคอร์สอบรม {entity}", "training"),
    ("ส่งมอบงานงวดแรก {entity}", "milestone"),
    ("ตรวจรับบ้าน {entity}", "inspection"),
    ("นัดทานข้าว {entity}", "casual_meet"),
    ("เริ่มงานวันแรก {entity}", "work_start"),
    ("เปิดตัวโปรดักต์ใหม่ {entity}", "launch"),
    ("กำหนดการสัมมนา {entity}", "seminar"),
    ("เตรียมเอกสารยื่นภาษี {entity}", "tax"),
    ("ส่งมอบรถยนต์ {entity}", "handover"),
]

OOD_3A_TEMPLATES_LEGACY = [
    ("รบกวนส่งข้อมูลสรุปให้ก่อน{entity}ด้วยครับ", "disjoint_request"),
    ("ทีมไอทีจะทำการ maintenance ระบบ{entity}", "disjoint_ops"),
    ("จะโอนยอดเงินส่วนที่เหลือให้ภายใน{entity}นะคะ", "disjoint_finance"),
    ("ทางร้านจะจัดส่งพัสดุรอบด่วนช่วง{entity}", "disjoint_shipping"),
    ("สนใจลงทะเบียนร่วมกิจกรรมก่อน{entity}ไหมครับ", "disjoint_marketing"),
    ("ขออนุญาตคอนเฟิร์มนัดหมายเวลา{entity}", "disjoint_confirmation"),
    ("ฝ่ายบุคคลแจ้งให้เข้ามารายงานตัว{entity}", "disjoint_hr"),
    ("ขอเชิญเข้าร่วมฟังการบรรยายพิเศษ{entity}", "disjoint_academic"),
    ("ลูกค้าสะดวกรับของตอน{entity}หรือเปล่า", "disjoint_customer_service"),
    ("ขอยื่นเอกสารเบิกจ่ายงบประมาณ{entity}", "disjoint_accounting"),
    ("[URGENT] ส่งมอบโค้ดเวอร์ชันล่าสุด {entity}", "disjoint_bracket_noise"),
    ("ระบบจะทำการปิดปรับปรุงชั่วคราว: {entity}", "disjoint_colon_noise"),
    ("Order Confirmed -> เตรียมจัดส่ง {entity}", "disjoint_arrow_noise"),
    ("งานเสร็จเรียบร้อยพร้อมตรวจรับ {entity} !!!", "disjoint_exclamation_noise"),
    ("ขอนัดหมายล่วงหน้านะครับ... {entity}", "disjoint_ellipsis_noise"),
    ("แจ้งปิดปรับปรุงเซิร์ฟเวอร์ช่วง {entity}", "disjoint_ops"),
    ("โปรโมชั่นพิเศษสิ้นสุด {entity}", "disjoint_sales"),
    ("เรียนเชิญประชุมสามัญประจำปี {entity}", "disjoint_formal"),
    ("เปิดให้ดาวน์โหลดแอปพลิเคชัน {entity}", "disjoint_tech"),
    ("ปิดรับข้อเสนอโครงการ {entity}", "disjoint_procurement"),
]

OOD_3B_TEMPLATES_LEGACY = [
    ("เอ่อ... เจอกัน{entity}ละกันน้า", "chat_filler"),
    ("เค้าจะแวะไปหานะเตง{entity}", "chat_sweet"),
    ("ว่าไงพวก ว่างเจอกัน{entity}ปะ", "chat_buddy"),
    ("ส่งของให้ช่วง{entity}จ้า รอก่อนน้า", "chat_shop"),
    ("ประมาณ{entity}น่าจะถึงที่หมาย", "chat_approx"),
    ("เด๋วรีบทักไปหา{entity}เด้อ", "chat_slang"),
    ("ออนดิสคอร์ด{entity}นะทุกคน รออยู่", "chat_gamer"),
    ("แกรรรร เจอกัน{entity} ห้ามสายเด็ดขาด", "chat_peer"),
    ("เค้าติดงานอยู่ น่าจะว่างอีกที{entity}จ้า", "chat_excuse"),
    ("ขอเลื่อนนัดไปเป็น{entity}แทนได้ไหมมม", "chat_postpone"),
]


def _legacy_build_item(
    item_id: int,
    raw_input: str,
    text_span: str,
    ent_type: str,
    grain: str,
    iso_val: Optional[str],
    range_s: Optional[str],
    range_e: Optional[str],
    precision: str,
    confidence: float,
    category: str,
    ling_type: str,
    has_midnight: bool,
    notes: str,
) -> Dict[str, Any]:
    start_char = raw_input.find(text_span)
    if start_char == -1:
        raise ValueError(f"Span '{text_span}' not found in raw_input '{raw_input}'")
    end_char = start_char + len(text_span)
    assert raw_input[start_char:end_char] == text_span

    return {
        "id": item_id,
        "raw_input": raw_input,
        "ground_truth": {
            "text_span": text_span,
            "start_char": start_char,
            "end_char": end_char,
            "type": ent_type,
            "grain": grain,
            "iso_value": iso_val,
            "range_start": range_s,
            "range_end": range_e,
            "precision": precision,
            "confidence": confidence,
        },
        "category": category,
        "metadata": {
            "reference_time": REF_TIME_STR,
            "timezone": "Asia/Bangkok",
            "linguistic_type": ling_type,
            "has_midnight_crossing": has_midnight,
        },
        "notes": notes,
    }


def _legacy_generate_candidate_entity(rng: random.Random) -> Dict[str, Any]:
    cat_choice = rng.choices(
        ["relative_day_week", "thai_6hour_clock", "coarse_time_periods", "short_buddhist_era", "night_date_rollover", "compound"],
        weights=[25, 25, 15, 15, 10, 10],
        k=1,
    )[0]

    if cat_choice == "relative_day_week":
        sub_type = rng.choice(["deictic", "weekday", "numeric"])
        if sub_type == "deictic":
            span, day_offset, ling = rng.choice(RELATIVE_DAYS)
        elif sub_type == "weekday":
            span, day_offset, ling = rng.choice(WEEKDAY_OFFSETS)
        else:
            span, day_offset = rng.choice(NUMERIC_DAY_OFFSETS)
            ling = "numeric_offset"

        target_dt = DEFAULT_REF_TIME + timedelta(days=day_offset)
        iso_val = target_dt.replace(hour=12, minute=0, second=0).isoformat()
        return {
            "text_span": span,
            "type": "POINT",
            "grain": "day",
            "iso_value": iso_val,
            "range_start": None,
            "range_end": None,
            "precision": "exact",
            "confidence": 1.0,
            "category": "relative_day_week",
            "linguistic_type": ling,
            "has_midnight_crossing": False,
            "notes": f"Relative day/week offset {day_offset} days",
        }

    elif cat_choice == "thai_6hour_clock":
        clock_sub = rng.choice([
            CLOCK_MORNING, CLOCK_NOON, CLOCK_AFTERNOON,
            CLOCK_EVENING, CLOCK_NIGHT, CLOCK_MIDNIGHT, CLOCK_EARLY_MORNING
        ])
        span, h, m, ling = rng.choice(clock_sub)
        target_dt = DEFAULT_REF_TIME.replace(hour=h, minute=m, second=0)
        grain = "minute" if m != 0 else "hour"
        return {
            "text_span": span,
            "type": "POINT",
            "grain": grain,
            "iso_value": target_dt.isoformat(),
            "range_start": None,
            "range_end": None,
            "precision": "exact",
            "confidence": 1.0,
            "category": "thai_6hour_clock",
            "linguistic_type": ling,
            "has_midnight_crossing": False,
            "notes": f"Thai clock time {h:02d}:{m:02d}",
        }

    elif cat_choice == "coarse_time_periods":
        span, s_time, e_time, day_shift, is_mid, ling = rng.choice(COARSE_PERIODS)
        s_h, s_m, s_s = map(int, s_time.split(":"))
        e_h, e_m, e_s = map(int, e_time.split(":"))
        dt_s = DEFAULT_REF_TIME.replace(hour=s_h, minute=s_m, second=s_s)
        dt_e = (DEFAULT_REF_TIME + timedelta(days=day_shift)).replace(hour=e_h, minute=e_m, second=e_s)
        return {
            "text_span": span,
            "type": "RANGE",
            "grain": "auto",
            "iso_value": None,
            "range_start": dt_s.isoformat(),
            "range_end": dt_e.isoformat(),
            "precision": "coarse",
            "confidence": 0.80 if not is_mid else 0.75,
            "category": "coarse_time_periods",
            "linguistic_type": ling,
            "has_midnight_crossing": is_mid,
            "notes": f"Coarse range {s_time} to {e_time}",
        }

    elif cat_choice == "short_buddhist_era":
        sub_type = rng.choice(["year", "date"])
        if sub_type == "year":
            span, ce_yr, ling = rng.choice(BUDDHIST_ERA_YEARS)
            if span.startswith("ต้นปี"):
                return {
                    "text_span": span,
                    "type": "RANGE",
                    "grain": "auto",
                    "iso_value": None,
                    "range_start": f"{ce_yr:04d}-01-01T00:00:00+07:00",
                    "range_end": f"{ce_yr:04d}-04-30T23:59:59+07:00",
                    "precision": "coarse",
                    "confidence": 0.85,
                    "category": "short_buddhist_era",
                    "linguistic_type": ling,
                    "has_midnight_crossing": False,
                    "notes": f"Buddhist Era early-year range {span}",
                }
            elif span.startswith("กลางปี"):
                return {
                    "text_span": span,
                    "type": "RANGE",
                    "grain": "auto",
                    "iso_value": None,
                    "range_start": f"{ce_yr:04d}-05-01T00:00:00+07:00",
                    "range_end": f"{ce_yr:04d}-08-31T23:59:59+07:00",
                    "precision": "coarse",
                    "confidence": 0.85,
                    "category": "short_buddhist_era",
                    "linguistic_type": ling,
                    "has_midnight_crossing": False,
                    "notes": f"Buddhist Era mid-year range {span}",
                }
            elif span.startswith("ปลายปี"):
                return {
                    "text_span": span,
                    "type": "RANGE",
                    "grain": "auto",
                    "iso_value": None,
                    "range_start": f"{ce_yr:04d}-09-01T00:00:00+07:00",
                    "range_end": f"{ce_yr:04d}-12-31T23:59:59+07:00",
                    "precision": "coarse",
                    "confidence": 0.85,
                    "category": "short_buddhist_era",
                    "linguistic_type": ling,
                    "has_midnight_crossing": False,
                    "notes": f"Buddhist Era late-year range {span}",
                }
            else:
                iso_val = f"{ce_yr:04d}-01-01T00:00:00+07:00"
                return {
                    "text_span": span,
                    "type": "POINT",
                    "grain": "year",
                    "iso_value": iso_val,
                    "range_start": None,
                    "range_end": None,
                    "precision": "exact",
                    "confidence": 1.0,
                    "category": "short_buddhist_era",
                    "linguistic_type": ling,
                    "has_midnight_crossing": False,
                    "notes": f"Buddhist Era expression {span}",
                }
        else:
            span, iso_val, ling = rng.choice(BUDDHIST_ERA_DATES)
            return {
                "text_span": span,
                "type": "POINT",
                "grain": "day",
                "iso_value": iso_val,
                "range_start": None,
                "range_end": None,
                "precision": "exact",
                "confidence": 1.0,
                "category": "short_buddhist_era",
                "linguistic_type": ling,
                "has_midnight_crossing": False,
                "notes": f"Buddhist Era expression {span}",
            }

    elif cat_choice == "night_date_rollover":
        span, iso_val, ling = rng.choice(NIGHT_ROLLOVERS)
        return {
            "text_span": span,
            "type": "POINT",
            "grain": "minute",
            "iso_value": iso_val,
            "range_start": None,
            "range_end": None,
            "precision": "exact",
            "confidence": 0.95,
            "category": "night_date_rollover",
            "linguistic_type": ling,
            "has_midnight_crossing": True,
            "notes": f"Night-time day rollover {span}",
        }

    else:  # compound
        day_span, day_offset, _ = rng.choice(RELATIVE_DAYS[:4])
        clock_sub = rng.choice([CLOCK_MORNING, CLOCK_NOON, CLOCK_AFTERNOON, CLOCK_EVENING, CLOCK_NIGHT])
        c_span, h, m, _ = rng.choice(clock_sub)
        span = f"{day_span} {c_span}"
        target_dt = (DEFAULT_REF_TIME + timedelta(days=day_offset)).replace(hour=h, minute=m, second=0)
        return {
            "text_span": span,
            "type": "POINT",
            "grain": "minute" if m != 0 else "hour",
            "iso_value": target_dt.isoformat(),
            "range_start": None,
            "range_end": None,
            "precision": "exact",
            "confidence": 0.95,
            "category": "compound_datetime",
            "linguistic_type": "compound_relative_clock",
            "has_midnight_crossing": False,
            "notes": f"Compound relative day + clock {span}",
        }


def _legacy_generate_dataset_split(
    split_name: str,
    count: int,
    templates: List[Tuple[str, str]],
    rng: random.Random,
    start_id: int = 1,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i in range(count):
        cand = _legacy_generate_candidate_entity(rng)
        tmpl, tmpl_desc = rng.choice(templates)
        raw_input = tmpl.replace("{entity}", cand["text_span"])

        if len(raw_input) > 120:
            raw_input = cand["text_span"]

        rec = _legacy_build_item(
            item_id=start_id + i,
            raw_input=raw_input,
            text_span=cand["text_span"],
            ent_type=cand["type"],
            grain=cand["grain"],
            iso_val=cand["iso_value"],
            range_s=cand["range_start"],
            range_e=cand["range_end"],
            precision=cand["precision"],
            confidence=cand["confidence"],
            category=cand["category"],
            ling_type=cand["linguistic_type"],
            has_midnight=cand["has_midnight_crossing"],
            notes=f"{cand['notes']} [tmpl: {tmpl_desc}]",
        )
        records.append(rec)
    return records


def _legacy_generate_ood_split(
    count_3a: int,
    count_3b: int,
    rng: random.Random,
    start_id: int = 1,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    records_3a = _legacy_generate_dataset_split(
        split_name="test_ood_3a",
        count=count_3a,
        templates=OOD_3A_TEMPLATES_LEGACY,
        rng=rng,
        start_id=start_id,
    )
    records.extend(records_3a)

    cur_id = start_id + count_3a
    records_3b_chat = _legacy_generate_dataset_split(
        split_name="test_ood_3b_chat",
        count=count_3b - 30,
        templates=OOD_3B_TEMPLATES_LEGACY,
        rng=rng,
        start_id=cur_id,
    )
    records.extend(records_3b_chat)
    cur_id += len(records_3b_chat)

    edge_cases = [
        ("เจอกัน พน. นะเพื่อน", "พน.", "POINT", "day", "2026-11-09T12:00:00+07:00", None, None, "exact", 0.9, "relative_day_week", "slang_abbr_period", False, "Slang with period พน."),
        ("ส่งงาน มรน. เด้อ", "มรน.", "POINT", "day", "2026-11-10T12:00:00+07:00", None, None, "exact", 0.9, "relative_day_week", "slang_abbr_period", False, "Slang with period มรน."),
        ("เจอกัน 3ทุ่มกว่าๆ ที่ผับ", "3ทุ่มกว่าๆ", "POINT", "minute", "2026-11-08T21:15:00+07:00", None, None, "coarse", 0.75, "thai_6hour_clock", "slang_particle_kwa", False, "Colloquial 3ทุ่มกว่าๆ"),
        ("ไปถึง บ่าย2นิดๆ จ้า", "บ่าย2นิดๆ", "POINT", "minute", "2026-11-08T14:10:00+07:00", None, None, "coarse", 0.75, "thai_6hour_clock", "slang_particle_nid", False, "Colloquial บ่าย2นิดๆ"),
        ("ตื่นมาก็ สายโด่ง แล้ว", "สายโด่ง", "RANGE", "auto", None, "2026-11-08T10:00:00+07:00", "2026-11-08T12:00:00+07:00", "coarse", 0.7, "coarse_time_periods", "slang_coarse_idiom", False, "Idiomatic coarse สายโด่ง"),
        ("นัดกินข้าว เที่ยงเป๊ง นะ", "เที่ยงเป๊ง", "POINT", "minute", "2026-11-08T12:00:00+07:00", None, None, "exact", 0.85, "thai_6hour_clock", "slang_exact_noon", False, "Idiomatic exact noon เที่ยงเป๊ง"),
        ("แดดร่มลมตก บ่ายคล้อย ค่อยไป", "บ่ายคล้อย", "RANGE", "auto", None, "2026-11-08T14:30:00+07:00", "2026-11-08T16:30:00+07:00", "coarse", 0.7, "coarse_time_periods", "slang_coarse_idiom", False, "Idiomatic coarse บ่ายคล้อย"),
        ("ส่งของให้ พรุ่งนี้้ นะ", "พรุ่งนี้้", "POINT", "day", "2026-11-09T12:00:00+07:00", None, None, "exact", 0.8, "relative_day_week", "typo_double_tone", False, "Typo double tone mark"),
        ("เจอกัน พรุ่งนี้ ครับ", "พรุ่งนี้", "POINT", "day", "2026-11-09T12:00:00+07:00", None, None, "exact", 0.8, "relative_day_week", "typo_inverted_vowel", False, "Typo inverted unicode vowel order"),
        ("นัดคุยงาน มะรืนนีั", "มะรืนนีั", "POINT", "day", "2026-11-10T12:00:00+07:00", None, None, "exact", 0.8, "relative_day_week", "typo_shifted_key", False, "Typo shifted key symbol"),
        ("เวลานัดหมาย ๑0:00 น.", "๑0:00 น.", "POINT", "hour", "2026-11-08T10:00:00+07:00", None, None, "exact", 0.8, "thai_6hour_clock", "typo_mixed_digits", False, "OCR mixed Thai digit and Arabic zero"),
        ("รอเจอตอน บ่ายสองโมงครึ่งงงง", "บ่ายสองโมงครึ่งงงง", "POINT", "minute", "2026-11-08T14:30:00+07:00", None, None, "exact", 0.8, "thai_6hour_clock", "typo_elongated_vowel", False, "Chat prolonged consonant tail"),
        ("สะดวกเจอกัน พรุ่งนี้หรือมะรืนนี้ ไหม", "พรุ่งนี้หรือมะรืนนี้", "POINT", "day", "2026-11-09T12:00:00+07:00", None, None, "unknown", 0.5, "compound_datetime", "multi_ambiguity_choice", False, "Choice between tomorrow and day after"),
        ("เลือกระหว่าง จันทร์หรืออังคารหน้า ครับ", "จันทร์หรืออังคารหน้า", "POINT", "day", "2026-11-16T12:00:00+07:00", None, None, "unknown", 0.5, "compound_datetime", "multi_ambiguity_choice", False, "Disjunctive weekday reference"),
        ("มีเวลาช่วง บ่ายสองถึงบ่ายสี่ โมง", "บ่ายสองถึงบ่ายสี่", "RANGE", "hour", None, "2026-11-08T14:00:00+07:00", "2026-11-08T16:00:00+07:00", "exact", 0.85, "compound_datetime", "range_compound_explicit", False, "Explicit range duration afternoon"),
        ("นัดได้ทั้ง พุธหน้า หรือ พฤหัสหน้า", "พุธหน้า หรือ พฤหัสหน้า", "POINT", "day", "2026-11-11T12:00:00+07:00", None, None, "unknown", 0.5, "compound_datetime", "multi_ambiguity_choice", False, "Disjunctive future weekday choice"),
        ("เริ่มงานช่วง เช้าจรดเย็น ทุกวัน", "เช้าจรดเย็น", "RANGE", "auto", None, "2026-11-08T08:00:00+07:00", "2026-11-08T17:00:00+07:00", "coarse", 0.75, "coarse_time_periods", "coarse_compound_daylong", False, "Day-long coarse interval"),
        ("วันนัดหมาย 29 ก.พ. 66 ครับ", "29 ก.พ. 66", "POINT", "day", None, None, None, "unknown", 0.0, "short_buddhist_era", "calendar_invalid_leap", False, "Non-leap year impossible Feb 29"),
        ("กำหนดส่ง 31 เม.ย. 67 นี้", "31 เม.ย. 67", "POINT", "day", None, None, None, "unknown", 0.0, "short_buddhist_era", "calendar_invalid_30day_month", False, "April only has 30 days"),
        ("เซ็นสัญญา 30 ก.พ. 68 แน่นอน", "30 ก.พ. 68", "POINT", "day", None, None, None, "unknown", 0.0, "short_buddhist_era", "calendar_invalid_feb30", False, "Physically impossible 30 Feb"),
        ("ครบรอบเกษียณ ปี 81 ข้างหน้า", "ปี 81", "POINT", "year", "1938-01-01T00:00:00+07:00", None, None, "exact", 0.7, "short_buddhist_era", "century_pivot_overflow", False, "Year 81 pivots to 2481 (CE 1938) instead of 2581"),
        ("วางแผนระยะยาวถึง ปี 99 นู่นเลย", "ปี 99", "POINT", "year", "1956-01-01T00:00:00+07:00", None, None, "exact", 0.7, "short_buddhist_era", "century_pivot_distant", False, "Year 99 pivots to 2499 (CE 1956)"),
        ("สะดวกช่วงหัวค่ำ 6 โมงครึ่ง ถึง 3 ทุ่ม", "ช่วงหัวค่ำ", "RANGE", "auto", None, "2026-11-08T18:00:00+07:00", "2026-11-08T21:00:00+07:00", "coarse", 0.75, "coarse_time_periods", "coarse_custom_convention", False, "User defines dusk as 18:00 to 21:00 vs default 18:30-20:30"),
        ("แวะมาช่วงสายๆ สัก 8 โมงครึ่ง", "ช่วงสายๆ", "RANGE", "auto", None, "2026-11-08T08:30:00+07:00", "2026-11-08T11:00:00+07:00", "coarse", 0.75, "coarse_time_periods", "coarse_shifted_morning", False, "User defines late morning earlier at 08:30"),
        ("เจอกันช่วงดึกๆ 4 ทุ่ม ถึง ตี 1", "ช่วงดึกๆ", "RANGE", "auto", None, "2026-11-08T22:00:00+07:00", "2026-11-09T01:00:00+07:00", "coarse", 0.75, "coarse_time_periods", "coarse_truncated_midnight", True, "User defines late night end at 01:00 vs default 02:00"),
        ("นัดคุยช่วงเย็นๆ 5 โมง ถึง 1 ทุ่ม", "ช่วงเย็นๆ", "RANGE", "auto", None, "2026-11-08T17:00:00+07:00", "2026-11-08T19:00:00+07:00", "coarse", 0.75, "coarse_time_periods", "coarse_shifted_evening", False, "Shifted evening window 17:00 to 19:00"),
        ("ขอเรียนแจ้งให้ผู้เกี่ยวข้องทุกท่านทราบล่วงหน้าว่าทางเรามีความจำเป็นต้องแจ้งเลื่อนกำหนดการประชุมเนื่องจากติดภารกิจด่วนมาก โดยขอเลื่อนไปเป็น พรุ่งนี้", "พรุ่งนี้", "POINT", "day", "2026-11-09T12:00:00+07:00", None, None, "exact", 1.0, "relative_day_week", "length_exceeded_rejection", False, "Sentence length 147 chars (> 120 SLA bound)"),
        ("เอกสารข้อตกลงและสัญญาจ้างงานโครงการปรับปรุงระบบคลาวด์คอมพิวติ้งขององค์กรประจำปีงบประมาณได้รับการอนุมัติเรียบร้อยแล้วโดยมีกำหนดส่งมอบงาน มะรืนนี้", "มะรืนนี้", "POINT", "day", "2026-11-10T12:00:00+07:00", None, None, "exact", 1.0, "relative_day_week", "length_exceeded_rejection", False, "Sentence length 145 chars (> 120 SLA bound)"),
        ("รายงานสรุปผลการดำเนินงานไตรมาสที่สามได้รับการตรวจสอบและรับรองความถูกต้องจากคณะกรรมการฝ่ายบริหารเรียบร้อยแล้วพร้อมประกาศผลอย่างเป็นทางการ วันศุกร์หน้า", "วันศุกร์หน้า", "POINT", "day", "2026-11-13T12:00:00+07:00", None, None, "exact", 1.0, "relative_day_week", "length_exceeded_rejection", False, "Sentence length 150 chars (> 120 SLA bound)"),
        ("ตามที่ได้ประสานงานกับทีมงานวิศวกรผู้เชี่ยวชาญด้านความปลอดภัยทางไซเบอร์เพื่อเข้าตรวจสอบระบบและแก้ไขช่องโหว่ทั้งหมดนั้นทางทีมพร้อมเข้าดำเนินการช่วง ตอนสายๆ", "ตอนสายๆ", "RANGE", "auto", None, "2026-11-08T09:00:00+07:00", "2026-11-08T11:30:00+07:00", "coarse", 0.8, "coarse_time_periods", "length_exceeded_rejection", False, "Sentence length 155 chars (> 120 SLA bound)"),
    ]

    for raw, span, typ, gr, iso, rs, re, prec, conf, cat, ling, mid, note in edge_cases:
        rec = _legacy_build_item(
            item_id=cur_id,
            raw_input=raw,
            text_span=span,
            ent_type=typ,
            grain=gr,
            iso_val=iso,
            range_s=rs,
            range_e=re,
            precision=prec,
            confidence=conf,
            category=cat,
            ling_type=ling,
            has_midnight=mid,
            notes=note,
        )
        records.append(rec)
        cur_id += 1

    return records


# =============================================================================
# Cryptographic Checksum & File IO Routines
# =============================================================================

def compute_sha256(file_path: Path) -> str:
    """Compute hex-digest SHA-256 of a file using buffered 64 KiB reads."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, records: List[Dict[str, Any]], newline_format: str = "\n") -> None:
    """Write list of dictionaries to JSON Lines with strict byte-level newline control."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nl_bytes = newline_format.encode("utf-8")
    with open(path, "wb") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False).encode("utf-8") + nl_bytes)


# =============================================================================
# Execution Pipelines
# =============================================================================

def run_release_10k(
    out_dir: Path,
    seed: int,
    split_counts: Tuple[int, int, int, int] = (7000, 1000, 1000, 1000),
    verify_spans: bool = True,
    verify_dedup: bool = True,
    quiet: bool = False,
) -> None:
    """Generate 10,000-row synthetic release v1.1 for Hugging Face."""
    train_count, dev_count, id_count, ood_count = split_counts
    total_expected = sum(split_counts)

    if not quiet:
        print("=" * 70)
        print("THAI-TEMPORAL-NORMALIZER SYNTHETIC DATASET GENERATOR (RELEASE v1.1)")
        print(f"Output Directory: {out_dir}")
        print(f"Random Seed:      {seed}")
        print(f"Total Rows:       {total_expected} (Train: {train_count}, Dev: {dev_count}, ID: {id_count}, OOD: {ood_count})")
        print("=" * 70)

    rng = random.Random(seed)
    registry = TemplateRegistry()
    deduplicator = CrossSplitDeduplicator()

    # 1. Train Split
    if not quiet:
        print(f"[*] Generating Train split ({train_count} rows)...")
    train_records = generate_split_v1_1("train", train_count, registry, deduplicator, rng)
    train_file = out_dir / "train.jsonl"
    write_jsonl(train_file, train_records, newline_format="\n")

    # 2. Dev Split
    if not quiet:
        print(f"[*] Generating Dev split ({dev_count} rows)...")
    dev_records = generate_split_v1_1("dev", dev_count, registry, deduplicator, rng)
    dev_file = out_dir / "dev.jsonl"
    write_jsonl(dev_file, dev_records, newline_format="\n")

    # 3. In-Distribution Test Split
    if not quiet:
        print(f"[*] Generating ID Test split ({id_count} rows)...")
    test_id_records = generate_split_v1_1("test_id", id_count, registry, deduplicator, rng)
    test_id_file = out_dir / "test_id.jsonl"
    write_jsonl(test_id_file, test_id_records, newline_format="\n")

    # 4. Out-of-Distribution Test Split
    if not quiet:
        print(f"[*] Generating OOD Test split ({ood_count} rows)...")
    test_ood_records = generate_split_v1_1("test_ood", ood_count, registry, deduplicator, rng)
    test_ood_file = out_dir / "test_ood.jsonl"
    write_jsonl(test_ood_file, test_ood_records, newline_format="\n")

    all_splits = [
        ("train", train_file, train_records),
        ("dev", dev_file, dev_records),
        ("test_id", test_id_file, test_id_records),
        ("test_ood", test_ood_file, test_ood_records),
    ]

    total_generated = sum(len(recs) for _, _, recs in all_splits)
    assert total_generated == total_expected, f"Generated {total_generated} != {total_expected}"

    # Inline Span Offset Verification
    if verify_spans:
        if not quiet:
            print("[*] Verifying 100% exact character span integrity...")
        for split_name, fpath, recs in all_splits:
            for item in recs:
                gt = item.get("ground_truth")
                raw = item["raw_input"]
                if gt is not None:
                    s = gt["start_char"]
                    e = gt["end_char"]
                    span = gt["text_span"]
                    assert raw[s:e] == span, (
                        f"Span mismatch in {split_name} ({item['id']}): slice='{raw[s:e]}' != span='{span}'"
                    )
                else:
                    assert item["category"] == "negative_sample"
                    assert item["metadata"]["category"] == "negative_sample"
        if not quiet:
            print("[OK] Span Offset Verification: 100% exact match across all positive rows!")

    # Inline Deduplication Verification
    if verify_dedup:
        if not quiet:
            print("[*] Verifying zero cross-split duplicates...")
        all_texts = []
        for _, _, recs in all_splits:
            for item in recs:
                all_texts.append(item["raw_input"].strip())
        assert len(all_texts) == total_expected
        assert len(set(all_texts)) == total_expected, (
            f"Found {total_expected - len(set(all_texts))} duplicates in generated dataset!"
        )
        if not quiet:
            print(f"[OK] Deduplication Verification: Exactly 0 duplicates! Unique texts = {len(set(all_texts))}")

    # Compute SHA-256 Checksums and Manifest
    manifest_info: Dict[str, Any] = {
        "dataset_name": "DuckerMaster/Thai-Natural-DateTime-Dataset",
        "version": "th-temporal-v1.1.0",
        "license": "CC-BY-4.0",
        "total_records": total_expected,
        "seed": seed,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "splits": {},
        "files": {},
    }

    if not quiet:
        print("\n" + "=" * 70)
        print("DATASET CHECKSUM MANIFEST (SHA-256)")
        print("=" * 70)

    for split_key, fpath, recs in all_splits:
        sha = compute_sha256(fpath)
        size_b = fpath.stat().st_size
        file_info = {
            "records": len(recs),
            "sha256": sha,
            "size_bytes": size_b,
        }
        manifest_info["files"][fpath.name] = file_info
        manifest_info["splits"][split_key] = {
            "filename": fpath.name,
            "records": len(recs),
            "sha256": sha,
            "size_bytes": size_b,
        }
        if not quiet:
            print(f"{fpath.name:<18} ({len(recs):>5} rows): {sha}")

    checksum_file = out_dir / "checksums.json"
    with open(checksum_file, "w", encoding="utf-8") as f:
        json.dump(manifest_info, f, indent=2, ensure_ascii=False)

    if not quiet:
        print("=" * 70)
        print(f"[OK] Checksum manifest written to {checksum_file}")
        print(f"[SUCCESS] v1.1 Dataset generation completed successfully ({total_expected} rows).")


def run_baseline_3k(
    out_dir: Path,
    seed: int,
    quiet: bool = False,
) -> None:
    """Reproduce the Phase 2 baseline 3,000-row benchmark with CRLF (\\r\\n)."""
    if not quiet:
        print(f"Generating Phase 2 baseline synthetic dataset with seed={seed} in {out_dir}...")

    rng = random.Random(seed)

    # 1. Train set (1,500 rows)
    train_records = _legacy_generate_dataset_split("train", 1500, STD_TEMPLATES_LEGACY, rng, start_id=1)
    train_file = out_dir / "train_1500.jsonl"
    write_jsonl(train_file, train_records, newline_format="\r\n")
    write_jsonl(out_dir / "train.jsonl", train_records, newline_format="\r\n")
    if not quiet:
        print(f"[OK] Generated {len(train_records)} rows -> {train_file.name}")

    # 2. Dev set (500 rows)
    dev_records = _legacy_generate_dataset_split("dev", 500, STD_TEMPLATES_LEGACY, rng, start_id=1501)
    dev_file = out_dir / "dev_500.jsonl"
    write_jsonl(dev_file, dev_records, newline_format="\r\n")
    write_jsonl(out_dir / "dev.jsonl", dev_records, newline_format="\r\n")
    if not quiet:
        print(f"[OK] Generated {len(dev_records)} rows -> {dev_file.name}")

    # 3. ID Test set (500 rows)
    test_id_records = _legacy_generate_dataset_split("test_id", 500, STD_TEMPLATES_LEGACY, rng, start_id=2001)
    test_id_file = out_dir / "test_id_500.jsonl"
    write_jsonl(test_id_file, test_id_records, newline_format="\r\n")
    write_jsonl(out_dir / "test_id.jsonl", test_id_records, newline_format="\r\n")
    if not quiet:
        print(f"[OK] Generated {len(test_id_records)} rows -> {test_id_file.name}")

    # 4. OOD Blind Test set (500 rows)
    test_ood_records = _legacy_generate_ood_split(count_3a=400, count_3b=100, rng=rng, start_id=2501)
    test_ood_file = out_dir / "test_ood_500.jsonl"
    write_jsonl(test_ood_file, test_ood_records, newline_format="\r\n")
    write_jsonl(out_dir / "test_ood.jsonl", test_ood_records, newline_format="\r\n")
    if not quiet:
        print(f"[OK] Generated {len(test_ood_records)} rows -> {test_ood_file.name}")

    total_rows = len(train_records) + len(dev_records) + len(test_id_records) + len(test_ood_records)
    assert total_rows == 3000, f"Expected exactly 3000 rows, got {total_rows}"

    all_files = [train_file, dev_file, test_id_file, test_ood_file]
    for p in all_files:
        with open(p, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                raw = item["raw_input"]
                gt = item["ground_truth"]
                s = gt["start_char"]
                e = gt["end_char"]
                span = gt["text_span"]
                assert raw[s:e] == span, (
                    f"Integrity violation in {p.name} line {line_idx}: '{raw[s:e]}' != '{span}'"
                )

    manifest_info: Dict[str, Any] = {
        "dataset_name": "DuckerMaster/Thai-Natural-DateTime-Dataset",
        "version": "th-temporal-v1.0.0-draft",
        "license": "CC-BY-4.0",
        "total_records": 3000,
        "seed": seed,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }

    for p in all_files:
        sha = compute_sha256(p)
        manifest_info["files"][p.name] = {
            "records": sum(1 for line in open(p, "r", encoding="utf-8") if line.strip()),
            "sha256": sha,
            "size_bytes": p.stat().st_size,
        }

    checksum_file = out_dir / "checksums.json"
    with open(checksum_file, "w", encoding="utf-8") as f:
        json.dump(manifest_info, f, indent=2, ensure_ascii=False)

    if not quiet:
        print(f"[OK] Checksum manifest written to {checksum_file}")


# =============================================================================
# CLI Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic temporal dataset generator for Thai-Temporal-Normalizer."
    )
    parser.add_argument(
        "--mode",
        choices=["release_10k", "baseline_3k"],
        default="release_10k",
        help="Generation mode: 'release_10k' for 10,000-row dataset, or 'baseline_3k' to reproduce 3,000-row Phase 2 benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write output JSONL files and checksums.json. Defaults to data/v1.1 for release_10k, or tests/data for baseline_3k.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed for deterministic generation.",
    )
    parser.add_argument(
        "--total-rows",
        type=int,
        default=None,
        help="Total rows to generate (default: 10000 for release_10k, 3000 for baseline_3k).",
    )
    parser.add_argument(
        "--split-counts",
        nargs=4,
        type=int,
        default=None,
        metavar=("TRAIN", "DEV", "ID", "OOD"),
        help="Split counts for Train, Dev, Test ID, Test OOD.",
    )
    parser.add_argument(
        "--verify-spans",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify exact character spans after generation.",
    )
    parser.add_argument(
        "--verify-dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify zero duplicate raw_input texts across all splits after generation.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose progress logs.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.mode == "release_10k":
        out_dir = args.output_dir if args.output_dir is not None else project_root / "data" / "v1.1"
        split_counts = tuple(args.split_counts) if args.split_counts is not None else (7000, 1000, 1000, 1000)
        total_rows = args.total_rows if args.total_rows is not None else sum(split_counts)
        if sum(split_counts) != total_rows:
            raise ValueError(f"Sum of split counts {split_counts} ({sum(split_counts)}) != total_rows ({total_rows})")

        run_release_10k(
            out_dir=out_dir,
            seed=args.seed,
            split_counts=split_counts,
            verify_spans=args.verify_spans,
            verify_dedup=args.verify_dedup,
            quiet=args.quiet,
        )
    else:  # baseline_3k
        out_dir = args.output_dir if args.output_dir is not None else project_root / "tests" / "data"
        run_baseline_3k(
            out_dir=out_dir,
            seed=args.seed,
            quiet=args.quiet,
        )


if __name__ == "__main__":
    main()
