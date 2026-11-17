"""Local Hugging Face Dataset Loading Script for Thai Natural DateTime Dataset (v1.1.0).

This script implements an offline Hugging Face `GeneratorBasedBuilder` for the
10,000-row benchmark release of `DuckerMaster/Thai-Natural-DateTime-Dataset`.

Features:
- Pure offline loading from local directory `data/v1.1/` (zero external network requests).
- Four evaluation splits:
  - train (7,000 rows)
  - dev (1,000 rows)
  - test_id (1,000 rows)
  - test_ood (1,000 rows)
- Strongly-typed PyArrow schema supporting nullable `ground_truth` (for 1,000 negative samples).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import datasets

_DESCRIPTION = """\
DuckerMaster/Thai-Natural-DateTime-Dataset is a standardized, benchmark-grade Thai \
natural language temporal annotation dataset. Version 1.1.0 scales the synthetic \
dataset to 10,000 rows across 4 splits (train, dev, test_id, test_ood) with 100% exact \
character span offsets, semantic time filtering across 7 context domains, 10% negative \
samples with verified null ground truth, 15% temporal bounds/durations/recurrence, and \
5% chat colloquialisms.
"""

_HOMEPAGE = "https://github.com/DuckerMaster/Thai-Temporal-Normalizer"
_LICENSE = "CC-BY-4.0"
_VERSION = datasets.Version("1.1.0")

_CITATION = """\
@dataset{duckermaster_thai_natural_datetime_2026,
  author       = {DuckerMaster},
  title        = {Thai-Natural-DateTime-Dataset: A Standardized Benchmark for Colloquial Thai Temporal Normalization},
  year         = {2026},
  version      = {th-temporal-v1.1.0},
  publisher    = {GitHub},
  howpublished = {\\url{https://github.com/DuckerMaster/Thai-Temporal-Normalizer}},
  license      = {CC-BY-4.0}
}
"""


class ThaiNaturalDatetimeConfig(datasets.BuilderConfig):
    """BuilderConfig for ThaiNaturalDatetime."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ThaiNaturalDatetimeConfig."""
        kwargs.setdefault("version", _VERSION)
        super().__init__(**kwargs)


class ThaiNaturalDatetime(datasets.GeneratorBasedBuilder):
    """Thai Natural DateTime Dataset generator builder."""

    VERSION = _VERSION
    BUILDER_CONFIGS = [
        ThaiNaturalDatetimeConfig(
            name="default",
            version=_VERSION,
            description="Default configuration for Thai Natural DateTime Dataset v1.1.0 (10,000 rows)",
        ),
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self) -> datasets.DatasetInfo:
        """Return dataset metadata and schema."""
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "raw_input": datasets.Value("string"),
                "ground_truth": {
                    "text_span": datasets.Value("string"),
                    "start_char": datasets.Value("int32"),
                    "end_char": datasets.Value("int32"),
                    "type": datasets.Value("string"),
                    "normalized_value": datasets.Value("string"),
                    "grain": datasets.Value("string"),
                    "iso_value": datasets.Value("string"),
                    "range_start": datasets.Value("string"),
                    "range_end": datasets.Value("string"),
                    "precision": datasets.Value("string"),
                    "confidence": datasets.Value("float32"),
                },
                "category": datasets.Value("string"),
                "metadata": {
                    "split": datasets.Value("string"),
                    "category": datasets.Value("string"),
                    "semantic_domain": datasets.Value("string"),
                    "reference_time": datasets.Value("string"),
                    "timezone": datasets.Value("string"),
                    "linguistic_type": datasets.Value("string"),
                    "has_midnight_crossing": datasets.Value("bool"),
                    "negative_type": datasets.Value("string"),
                },
                "notes": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: Any
    ) -> list[datasets.SplitGenerator]:
        """Define split generators for train, dev, test_id, and test_ood."""
        base_dir = Path(__file__).resolve().parent
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": str(base_dir / "train.jsonl")},
            ),
            datasets.SplitGenerator(
                name=datasets.NamedSplit("dev"),
                gen_kwargs={"filepath": str(base_dir / "dev.jsonl")},
            ),
            datasets.SplitGenerator(
                name=datasets.NamedSplit("test_id"),
                gen_kwargs={"filepath": str(base_dir / "test_id.jsonl")},
            ),
            datasets.SplitGenerator(
                name=datasets.NamedSplit("test_ood"),
                gen_kwargs={"filepath": str(base_dir / "test_ood.jsonl")},
            ),
        ]

    def _generate_examples(
        self, filepath: str
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Yield examples from the JSONL files."""
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                clean_line = line.strip()
                if not clean_line:
                    continue
                row = json.loads(clean_line)
                row_id = str(row.get("id", f"row_{idx}"))

                gt = row.get("ground_truth")
                meta = row.get("metadata", {})

                # Ensure ground_truth struct conforms or is None
                gt_struct = None
                if gt is not None:
                    gt_struct = {
                        "text_span": gt.get("text_span"),
                        "start_char": gt.get("start_char"),
                        "end_char": gt.get("end_char"),
                        "type": gt.get("type"),
                        "normalized_value": gt.get("normalized_value"),
                        "grain": gt.get("grain"),
                        "iso_value": gt.get("iso_value"),
                        "range_start": gt.get("range_start"),
                        "range_end": gt.get("range_end"),
                        "precision": gt.get("precision"),
                        "confidence": (
                            float(gt["confidence"])
                            if gt.get("confidence") is not None
                            else None
                        ),
                    }

                meta_struct = {
                    "split": meta.get("split"),
                    "category": meta.get("category"),
                    "semantic_domain": meta.get("semantic_domain"),
                    "reference_time": meta.get("reference_time"),
                    "timezone": meta.get("timezone", "Asia/Bangkok"),
                    "linguistic_type": meta.get("linguistic_type"),
                    "has_midnight_crossing": meta.get("has_midnight_crossing", False),
                    "negative_type": meta.get("negative_type", None),
                }

                example = {
                    "id": row_id,
                    "raw_input": row.get("raw_input", ""),
                    "ground_truth": gt_struct,
                    "category": row.get("category", ""),
                    "metadata": meta_struct,
                    "notes": row.get("notes", ""),
                }

                yield row_id, example


def load_dataset_local(split: str | None = None) -> datasets.Dataset | datasets.DatasetDict:
    """Load the local 10,000-row Thai Natural DateTime dataset offline via GeneratorBasedBuilder.

    Args:
        split: Optional split name ('train', 'dev', 'test_id', 'test_ood').
            If None, returns all splits as a DatasetDict.

    Returns:
        Dataset or DatasetDict.
    """
    builder = ThaiNaturalDatetime()
    builder.download_and_prepare()
    dataset_dict = builder.as_dataset()
    if split is not None:
        return dataset_dict[split]
    return dataset_dict


if __name__ == "__main__":
    ds = load_dataset_local()
    print("Standalone builder dataset:", ds)
    total = sum(len(ds[s]) for s in ds)
    print(f"Total rows: {total}")
    assert total == 10000, f"Expected 10,000 rows, got {total}"
    print("Self-test passed!")
