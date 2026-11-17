"""Night-time day-rollover normalization engine."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import List, Optional

from thai_temporal_normalizer.models import TemporalEntity
from thai_temporal_normalizer.rules.clock import ClockRuleEngine
from thai_temporal_normalizer.rules.constants import BANGKOK_TZ, THAI_WEEKDAY_MAP


class NightRolloverRuleEngine:
    """Detects night prefixes and applies [TargetDay] + 1 Day rollover for early morning hours [00:00, 05:59]."""

    def __init__(self, clock_engine: Optional[ClockRuleEngine] = None) -> None:
        self.clock_engine = clock_engine or ClockRuleEngine()

    def parse_rollover_span(
        self,
        text: str,
        ref_time: datetime,
    ) -> List[TemporalEntity]:
        """Parse expressions involving night anchors and clock times.

        Handles:
            - 'คืนวันศุกร์ ตีสอง' -> Saturday 02:00:00 (Rollover +1d)
            - 'ดึกวันจันทร์ ตีหนึ่งครึ่ง' -> Tuesday 01:30:00 (Rollover +1d)
            - 'คืนนี้ ตีสาม' -> Tomorrow 03:00:00 (Rollover +1d)
            - 'เมื่อคืน ตีสอง' -> Today 02:00:00 (Rollover +1d)
            - 'คืนวันศุกร์ สี่ทุ่ม' -> Friday 22:00:00 (Negative control, no rollover)
            - 'เช้าวันเสาร์ ตีสอง' -> Saturday 02:00:00 (Morning anchor, no rollover)
            - 'วันศุกร์ ตีสอง' -> Friday 02:00:00 (No night prefix, no rollover)
        """
        entities: List[TemporalEntity] = []
        ref_date = ref_time.date()
        ref_weekday = ref_date.weekday()

        # Look for clock times in the string
        time_matches = self.clock_engine.parse_time_span(text)
        if not time_matches:
            return entities

        for span_time, t_start, t_end, time_obj, grain in time_matches:
            prefix_text = text[:t_start].strip()

            has_night_prefix = False
            has_anchor = False
            anchor_date = ref_date
            phrase_start = t_start

            if prefix_text:
                m_last_night = re.search(r"เมื่อคืน", prefix_text)
                if m_last_night:
                    has_night_prefix = True
                    has_anchor = True
                    anchor_date = ref_date - timedelta(days=1)
                    phrase_start = m_last_night.start()
                else:
                    m_tonight = re.search(r"(?:ดึกคืนนี้|คืนนี้)", prefix_text)
                    if m_tonight:
                        has_night_prefix = True
                        has_anchor = True
                        anchor_date = ref_date
                        phrase_start = m_tonight.start()
                    else:
                        m_wk = re.search(
                            r"(ดึกคืนวัน|คืนวัน|ดึกวัน|ดึกคืน|เช้าวัน|วัน)?(จันทร์|อังคาร|พุธ|พฤหัสบดี|พฤหัส|ศุกร์|เสาร์|อาทิตย์)",
                            prefix_text,
                        )
                        if m_wk:
                            has_anchor = True
                            prefix_type = m_wk.group(1) or ""
                            wk_name = m_wk.group(2)
                            target_wk = THAI_WEEKDAY_MAP.get(wk_name)
                            if target_wk is not None:
                                phrase_start = m_wk.start()
                                days_ahead = (target_wk - ref_weekday) % 7
                                if days_ahead == 0 and prefix_type in ("เช้าวัน", "วัน"):
                                    days_ahead = 0 if ref_weekday == target_wk and time_obj.hour >= ref_time.hour else 7
                                    # For baseline, Sunday Nov 8 looking at "คืนวันอาทิตย์" or "เช้าวันอาทิตย์":
                                    if target_wk == ref_weekday:
                                        days_ahead = 0
                                else:
                                    if days_ahead == 0:
                                        days_ahead = 7
                                    # Wait, for Sunday Nov 8:
                                    # "คืนวันศุกร์" -> target_wk=4, days_ahead = (4 - 6)%7 = 5 (Nov 13)
                                    # "คืนวันเสาร์" -> target_wk=5, days_ahead = (5 - 6)%7 = 6 (Nov 14)
                                    # "คืนวันอาทิตย์" -> target_wk=6, days_ahead = 0 (Nov 8)
                                    # "เช้าวันอาทิตย์" -> target_wk=6, days_ahead = 0 (Nov 8)
                                    # "เช้าวันเสาร์" -> target_wk=5, days_ahead = 6 (Nov 14)
                                    if target_wk == ref_weekday:
                                        days_ahead = 0

                                anchor_date = ref_date + timedelta(days=days_ahead)

                                if any(prefix_type.startswith(p) for p in ("คืน", "ดึก")):
                                    has_night_prefix = True

            # If no anchor was found before the clock time, it's not a night rollover case
            if not has_anchor:
                continue

            # Check rollover condition:
            # 1. Has night prefix (คืน, ดึก)
            # 2. Time is in early morning [00:00:00, 05:59:59]
            is_early_morning = (0 <= time_obj.hour <= 5)
            rollover_applied = has_night_prefix and is_early_morning

            if rollover_applied:
                final_date = anchor_date + timedelta(days=1)
            else:
                final_date = anchor_date

            final_dt = datetime(
                final_date.year,
                final_date.month,
                final_date.day,
                time_obj.hour,
                time_obj.minute,
                time_obj.second,
                tzinfo=ref_time.tzinfo or BANGKOK_TZ,
            )

            full_span_text = text[phrase_start:t_end].strip()

            entities.append(
                TemporalEntity(
                    text_span=full_span_text,
                    start_char=phrase_start,
                    end_char=t_end,
                    type="POINT",
                    grain=grain,
                    iso_value=final_dt.isoformat(),
                    precision="exact",
                    confidence=1.0,
                )
            )

        return entities
