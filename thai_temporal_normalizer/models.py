"""Data models and type definitions for Thai Temporal Normalizer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Literal, Optional, Tuple


class WeekdayAnchorPolicy(str, Enum):
    """Disambiguation policy for relative weekday references like 'พุธหน้า'."""

    NEXT_OCCURRENCE = "next_occurrence"
    NEXT_WEEK = "next_week"


@dataclass
class TemporalEntity:
    """Normalized temporal entity matching PRD Section 7.1 specification."""

    text_span: str
    start_char: int
    end_char: int
    type: Literal["POINT", "RANGE", "DURATION"]
    iso_value: Optional[str] = None
    range_start: Optional[str] = None
    range_end: Optional[str] = None
    precision: Literal["exact", "coarse", "unknown"] = "exact"
    confidence: float = 1.0
    grain: str = "auto"

    def __post_init__(self) -> None:
        valid_types = ("POINT", "RANGE", "DURATION")
        if self.type not in valid_types:
            raise ValueError(f"Invalid type '{self.type}'. Must be one of {valid_types}")

        valid_precisions = ("exact", "coarse", "unknown")
        if self.precision not in valid_precisions:
            raise ValueError(f"Invalid precision '{self.precision}'. Must be one of {valid_precisions}")

        if self.start_char < 0:
            raise ValueError(f"start_char must be non-negative, got {self.start_char}")
        if self.end_char < self.start_char:
            raise ValueError(
                f"end_char ({self.end_char}) cannot be less than start_char ({self.start_char})"
            )

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

    @property
    def span(self) -> Tuple[int, int]:
        """Character offset span tuple (start_char, end_char)."""
        return (self.start_char, self.end_char)

    @property
    def normalized_range(self) -> Tuple[Optional[str], Optional[str]]:
        """Normalized range tuple (range_start, range_end)."""
        return (self.range_start, self.range_end)

    @property
    def value(self) -> Optional[str]:
        """Alias for iso_value."""
        return self.iso_value

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TemporalEntity:
        """Instantiate entity from dictionary."""
        return cls(
            text_span=data["text_span"],
            start_char=data["start_char"],
            end_char=data["end_char"],
            type=data["type"],
            iso_value=data.get("iso_value"),
            range_start=data.get("range_start"),
            range_end=data.get("range_end"),
            precision=data.get("precision", "exact"),
            confidence=float(data.get("confidence", 1.0)),
            grain=data.get("grain", "auto"),
        )
