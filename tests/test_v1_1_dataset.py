"""Comprehensive 5-Tier Verification Suite for Thai-Temporal-Normalizer v1.1 (10k Dataset).

Requirement-driven opaque-box verification suite validating:
- Tier 1: Split Counts (7,000/1,000/1,000/1,000 = 10,000) & Zero Duplicate Texts.
- Tier 2: Exact Character Span Offset Accuracy (100% positive rows match raw_input[start:end] == span).
- Tier 3: Negative Sample Ground Truth (10% / 1,000 rows, null GT) & FP Immunity (0 False Positives).
- Tier 4: Distribution & Semantic Realism (70/15/5/10 stratification, 0 office templates in wee hours).
- Tier 5: Cryptographic SHA-256 Manifest Integrity & Baseline Safety (3,000-row benchmark untouched).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from thai_temporal_normalizer import parse_thai_temporal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V1_1_DIR = PROJECT_ROOT / "data" / "v1.1"
BASELINE_DIR = PROJECT_ROOT / "tests" / "data"

SPLIT_SPECS = {
    "train": 7000,
    "dev": 1000,
    "test_id": 1000,
    "test_ood": 1000,
}

EXPECTED_TOTAL_ROWS = 10000

GOLDEN_BASELINE_HASHES = {
    "train_1500.jsonl": "9f7e52cf17ce228f12c7ff0d4bf4531e9bb1db0eb678fa1d1bdccdc5fab88381",
    "dev_500.jsonl": "f4cf1b5ffa1cfe26f58c9285e2bfd65f20f89b1b5e7537f9d956eb8bab2b75ab",
    "test_id_500.jsonl": "4484ca4829941ff16da5c461c5b358ec286309d15fffc7211405854ae65c9c8e",
    "test_ood_500.jsonl": "44e6990c9f6e0ab0d96b695cf9ff5ad33a2466a94f3a1dd7d96d6e676ba1aad2",
}


def compute_file_sha256(path: Path) -> str:
    """Compute hex-digest SHA-256 of a file using buffered 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# Pytest Fixtures (Module-scoped for Fast In-Memory Execution)
# =============================================================================

@pytest.fixture(scope="module")
def v1_1_split_records() -> Dict[str, List[Dict[str, Any]]]:
    """Load and parse all 4 splits from data/v1.1/*.jsonl into memory once."""
    splits: Dict[str, List[Dict[str, Any]]] = {}
    for split_name in SPLIT_SPECS:
        split_path = V1_1_DIR / f"{split_name}.jsonl"
        assert split_path.exists(), f"Release file {split_path} does not exist!"
        records = []
        with open(split_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                clean_line = line.strip()
                if clean_line:
                    try:
                        records.append(json.loads(clean_line))
                    except json.JSONDecodeError as exc:
                        pytest.fail(f"Corrupt JSON at {split_path.name}:{line_idx}: {exc}")
        splits[split_name] = records
    return splits


@pytest.fixture(scope="module")
def v1_1_all_records(v1_1_split_records: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Flatten all records across all 4 release splits."""
    all_recs: List[Dict[str, Any]] = []
    for split_name, recs in v1_1_split_records.items():
        all_recs.extend(recs)
    return all_recs


@pytest.fixture(scope="module")
def v1_1_checksums_manifest() -> Dict[str, Any]:
    """Load and parse data/v1.1/checksums.json."""
    manifest_path = V1_1_DIR / "checksums.json"
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def baseline_checksums_manifest() -> Dict[str, Any]:
    """Load and parse tests/data/checksums.json."""
    manifest_path = BASELINE_DIR / "checksums.json"
    assert manifest_path.exists(), f"Missing baseline manifest: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Tier 1 — Split Counts & Zero Duplicates
# =============================================================================

class TestTier1SplitCountsAndZeroDuplicates:
    """Validate 10,000-row volume, exact split quotas, and global 0 duplication."""

    def test_release_files_exist(self):
        """All 4 release JSONL files and checksum manifest must exist in data/v1.1."""
        assert V1_1_DIR.exists(), f"Directory {V1_1_DIR} does not exist!"
        for split_name in SPLIT_SPECS:
            file_path = V1_1_DIR / f"{split_name}.jsonl"
            assert file_path.is_file(), f"Missing release split file: {file_path}"
            assert file_path.stat().st_size > 0, f"File {file_path} is empty!"
        assert (V1_1_DIR / "checksums.json").is_file(), "Missing data/v1.1/checksums.json"

    def test_split_row_counts(self, v1_1_split_records: Dict[str, List[Dict[str, Any]]]):
        """Assert exact row counts: Train=7,000, Dev=1,000, ID=1,000, OOD=1,000 (total 10,000)."""
        total = 0
        for split_name, expected_count in SPLIT_SPECS.items():
            actual_count = len(v1_1_split_records[split_name])
            assert actual_count == expected_count, (
                f"Split '{split_name}' count mismatch: expected {expected_count}, got {actual_count}"
            )
            total += actual_count
        assert total == EXPECTED_TOTAL_ROWS, f"Total records mismatch: expected {EXPECTED_TOTAL_ROWS}, got {total}"

    def test_record_ids_unique_globally(self, v1_1_all_records: List[Dict[str, Any]]):
        """Assert every record has a unique ID across the entire 10,000-row dataset."""
        ids = [rec["id"] for rec in v1_1_all_records if "id" in rec]
        assert len(ids) == EXPECTED_TOTAL_ROWS, "Some records lack 'id' key!"
        assert len(set(ids)) == EXPECTED_TOTAL_ROWS, (
            f"Found {EXPECTED_TOTAL_ROWS - len(set(ids))} duplicate record IDs!"
        )

    def test_raw_input_zero_duplicates_within_splits(
        self, v1_1_split_records: Dict[str, List[Dict[str, Any]]]
    ):
        """Assert exactly 0 duplicate raw_input texts within any individual split."""
        for split_name, recs in v1_1_split_records.items():
            texts = [r["raw_input"].strip() for r in recs]
            duplicates = len(texts) - len(set(texts))
            assert duplicates == 0, (
                f"Split '{split_name}' contains {duplicates} duplicate raw_input texts!"
            )

    def test_raw_input_zero_duplicates_across_all_splits(
        self, v1_1_all_records: List[Dict[str, Any]]
    ):
        """Assert exactly 0 duplicate raw_input texts across all splits (strictly 10,000 unique texts)."""
        texts = [r["raw_input"].strip() for r in v1_1_all_records]
        assert len(texts) == EXPECTED_TOTAL_ROWS
        unique_texts = set(texts)
        assert len(unique_texts) == EXPECTED_TOTAL_ROWS, (
            f"Cross-split deduplication failed! Total rows: {len(texts)}, Unique texts: {len(unique_texts)}. "
            f"Duplicates: {len(texts) - len(unique_texts)}"
        )


# =============================================================================
# Tier 2 — Exact Character Span Offset Accuracy
# =============================================================================

class TestTier2ExactCharacterSpanOffsetAccuracy:
    """Validate 100% exact character span offset accuracy and coordinate validity."""

    def test_positive_rows_exact_character_span_slice(
        self, v1_1_all_records: List[Dict[str, Any]]
    ):
        """Assert raw_input[start_char:end_char] == text_span for 100% of positive rows."""
        mismatches: List[str] = []
        positive_count = 0

        for rec in v1_1_all_records:
            gt = rec.get("ground_truth")
            if gt is None:
                continue

            positive_count += 1
            raw = rec["raw_input"]
            start_c = gt["start_char"]
            end_c = gt["end_char"]
            span = gt["text_span"]

            actual_slice = raw[start_c:end_c]
            if actual_slice != span:
                mismatches.append(
                    f"ID={rec['id']}: raw[{start_c}:{end_c}]='{actual_slice}' != text_span='{span}'"
                )

        assert len(mismatches) == 0, (
            f"Encountered {len(mismatches)} span offset mismatches out of {positive_count} positive rows!\n"
            + "\n".join(mismatches[:10])
        )
        assert positive_count == 9000, f"Expected 9,000 positive rows, found {positive_count}"

    def test_span_coordinates_bounds_and_validity(
        self, v1_1_all_records: List[Dict[str, Any]]
    ):
        """Assert span coordinates are non-empty, within string bounds, and start < end."""
        for rec in v1_1_all_records:
            gt = rec.get("ground_truth")
            if gt is None:
                continue

            raw = rec["raw_input"]
            start_c = gt["start_char"]
            end_c = gt["end_char"]
            span = gt["text_span"]

            assert isinstance(start_c, int) and isinstance(end_c, int), (
                f"Record {rec['id']} coordinates are not integers: start={start_c}, end={end_c}"
            )
            assert 0 <= start_c < end_c <= len(raw), (
                f"Record {rec['id']} invalid coordinate bounds: 0 <= {start_c} < {end_c} <= {len(raw)}"
            )
            assert len(span) == (end_c - start_c), (
                f"Record {rec['id']} span length mismatch: len('{span}')={len(span)} != {end_c - start_c}"
            )
            assert len(span) > 0, f"Record {rec['id']} has empty text_span!"

    def test_positive_ground_truth_schema_completeness(
        self, v1_1_all_records: List[Dict[str, Any]]
    ):
        """Assert positive samples conform to required ground_truth schema."""
        valid_types = {"POINT", "RANGE", "DURATION", "RECURRENCE"}
        for rec in v1_1_all_records:
            gt = rec.get("ground_truth")
            if gt is None:
                continue

            assert "text_span" in gt, f"Record {rec['id']} missing text_span in ground_truth"
            assert "start_char" in gt, f"Record {rec['id']} missing start_char in ground_truth"
            assert "end_char" in gt, f"Record {rec['id']} missing end_char in ground_truth"
            assert "type" in gt, f"Record {rec['id']} missing type in ground_truth"
            assert gt["type"] in valid_types, f"Record {rec['id']} invalid type: {gt['type']}"
            assert "normalized_value" in gt, f"Record {rec['id']} missing normalized_value"


# =============================================================================
# Tier 3 — Negative Sample Ground Truth & False Positive Immunity
# =============================================================================

class TestTier3NegativeSampleGroundTruthAndFPImmunity:
    """Validate negative samples constitute ~10%, have null GT, and 0 false positives."""

    def test_negative_samples_count_and_proportions(
        self,
        v1_1_all_records: List[Dict[str, Any]],
        v1_1_split_records: Dict[str, List[Dict[str, Any]]],
    ):
        """Assert negative samples constitute exactly 10% (1,000 rows) overall and 10% in each split."""
        all_negs = [r for r in v1_1_all_records if r.get("ground_truth") is None]
        assert len(all_negs) == 1000, f"Expected exactly 1,000 negative samples, got {len(all_negs)}"

        expected_neg_per_split = {
            "train": 700,
            "dev": 100,
            "test_id": 100,
            "test_ood": 100,
        }
        for split_name, expected_neg in expected_neg_per_split.items():
            split_negs = [r for r in v1_1_split_records[split_name] if r.get("ground_truth") is None]
            assert len(split_negs) == expected_neg, (
                f"Split '{split_name}' negative count mismatch: expected {expected_neg}, got {len(split_negs)}"
            )

    def test_negative_samples_null_ground_truth_and_metadata(
        self, v1_1_all_records: List[Dict[str, Any]]
    ):
        """Assert all negative samples have ground_truth is None and proper metadata annotations."""
        valid_neg_types = {"currency", "phone_number", "tracking_code", "general_text"}
        negative_count = 0

        for rec in v1_1_all_records:
            if rec.get("ground_truth") is None:
                negative_count += 1
                assert rec.get("category") == "negative_sample", (
                    f"Negative record {rec['id']} category is not 'negative_sample'!"
                )
                meta = rec.get("metadata", {})
                assert meta.get("category") == "negative_sample", (
                    f"Negative record {rec['id']} metadata.category is not 'negative_sample'!"
                )
                neg_type = meta.get("negative_type")
                assert neg_type in valid_neg_types, (
                    f"Negative record {rec['id']} invalid negative_type: {neg_type}"
                )

        assert negative_count == 1000, f"Expected 1,000 negative records, counted {negative_count}"

    def test_negative_samples_false_positive_immunity(
        self, v1_1_all_records: List[Dict[str, Any]]
    ):
        """Assert negative samples pass through parse_thai_temporal with 0 false positive extractions."""
        false_positives: List[Tuple[str, str, List[Any]]] = []
        negative_recs = [r for r in v1_1_all_records if r.get("ground_truth") is None]

        for rec in negative_recs:
            raw_text = rec["raw_input"]
            parsed = parse_thai_temporal(raw_text)
            if len(parsed) > 0:
                false_positives.append((rec["id"], raw_text, parsed))

        assert len(false_positives) == 0, (
            f"Normalizer produced {len(false_positives)} false positive extractions on negative samples!\n"
            + "\n".join([f"ID={rec_id}: text='{text}', extracted={[e.text_span for e in entities]}" for rec_id, text, entities in false_positives[:10]])
        )


# =============================================================================
# Tier 4 — Distribution & Semantic Realism
# =============================================================================

class TestTier4DistributionAndSemanticRealism:
    """Validate linguistic category proportions and semantic diurnal realism."""

    def test_category_distribution_dataset_wide(
        self, v1_1_all_records: List[Dict[str, Any]]
    ):
        """Assert category distribution matches ~70% standard, ~15% bounds, ~5% chat, ~10% negatives."""
        cat_counts: Dict[str, int] = {}
        for rec in v1_1_all_records:
            cat = rec["metadata"]["category"]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        total = len(v1_1_all_records)
        assert total == EXPECTED_TOTAL_ROWS

        assert cat_counts.get("standard", 0) == 7000, (
            f"Expected 7,000 standard rows (70%), got {cat_counts.get('standard')}"
        )
        assert cat_counts.get("bound_duration_recurrence", 0) == 1500, (
            f"Expected 1,500 bound_duration_recurrence rows (15%), got {cat_counts.get('bound_duration_recurrence')}"
        )
        assert cat_counts.get("chat_colloquial", 0) == 500, (
            f"Expected 500 chat_colloquial rows (5%), got {cat_counts.get('chat_colloquial')}"
        )
        assert cat_counts.get("negative_sample", 0) == 1000, (
            f"Expected 1,000 negative_sample rows (10%), got {cat_counts.get('negative_sample')}"
        )

    def test_category_distribution_per_split(
        self, v1_1_split_records: Dict[str, List[Dict[str, Any]]]
    ):
        """Assert category distribution holds proportionally across all 4 splits."""
        expected_breakdown = {
            "train": {"standard": 4900, "bound_duration_recurrence": 1050, "chat_colloquial": 350, "negative_sample": 700},
            "dev": {"standard": 700, "bound_duration_recurrence": 150, "chat_colloquial": 50, "negative_sample": 100},
            "test_id": {"standard": 700, "bound_duration_recurrence": 150, "chat_colloquial": 50, "negative_sample": 100},
            "test_ood": {"standard": 700, "bound_duration_recurrence": 150, "chat_colloquial": 50, "negative_sample": 100},
        }

        for split_name, exp_counts in expected_breakdown.items():
            actual_counts: Dict[str, int] = {}
            for rec in v1_1_split_records[split_name]:
                cat = rec["metadata"]["category"]
                actual_counts[cat] = actual_counts.get(cat, 0) + 1

            for cat, exp_c in exp_counts.items():
                act_c = actual_counts.get(cat, 0)
                assert act_c == exp_c, (
                    f"Split '{split_name}' category '{cat}' count mismatch: expected {exp_c}, got {act_c}"
                )

    def test_semantic_realism_no_business_wee_hour_pairings(
        self, v1_1_all_records: List[Dict[str, Any]]
    ):
        """Assert 0 business/office templates are paired with wee-hour times (01:00-05:00)."""
        wee_hour_violations: List[Tuple[str, str, str, str]] = []
        wee_hours = {1, 2, 3, 4, 5}

        for rec in v1_1_all_records:
            meta = rec.get("metadata", {})
            domain = meta.get("semantic_domain")
            notes = rec.get("notes", "")

            # Identify office business contexts
            is_office = (domain == "office_business") or ("[tmpl: TMPL_BUS_" in notes)
            if not is_office:
                continue

            gt = rec.get("ground_truth")
            if not gt:
                continue

            # Check all time-bearing fields for clock hours
            for field in ("iso_value", "range_start", "range_end"):
                val = gt.get(field)
                if val and isinstance(val, str) and "T" in val:
                    # Match standard ISO-8601 time: YYYY-MM-DDTHH:MM:SS
                    m = re.search(r"T(\d{2}):(\d{2})", val)
                    if m:
                        hour = int(m.group(1))
                        if hour in wee_hours:
                            wee_hour_violations.append((rec["id"], rec["raw_input"], field, val))

        assert len(wee_hour_violations) == 0, (
            f"Found {len(wee_hour_violations)} semantic realism violations where business/office templates "
            f"were paired with wee-hour times (01:00-05:00)!\n"
            + "\n".join([f"ID={rec_id}: field={f} val={v} raw='{text}'" for rec_id, text, f, v in wee_hour_violations[:10]])
        )


# =============================================================================
# Tier 5 — SHA-256 Manifest Integrity & Baseline Safety
# =============================================================================

class TestTier5ChecksumManifestIntegrityAndBaselineSafety:
    """Validate cryptographic checksums for v1.1 and baseline 3k benchmark preservation."""

    def test_v1_1_checksum_manifest_matches_files(
        self, v1_1_checksums_manifest: Dict[str, Any]
    ):
        """Assert data/v1.1/checksums.json matches the actual SHA-256 digests and file sizes."""
        assert v1_1_checksums_manifest.get("version") == "th-temporal-v1.1.0"
        assert v1_1_checksums_manifest.get("total_records") == EXPECTED_TOTAL_ROWS
        assert v1_1_checksums_manifest.get("seed") == 42

        files_info = v1_1_checksums_manifest.get("files", {})
        assert len(files_info) == 4, f"Expected 4 release files in manifest, got {len(files_info)}"

        for filename, info in files_info.items():
            file_path = V1_1_DIR / filename
            assert file_path.is_file(), f"Manifest lists {filename} but file does not exist!"
            actual_sha = compute_file_sha256(file_path)
            expected_sha = info["sha256"]
            assert actual_sha == expected_sha, (
                f"SHA-256 mismatch for release file {filename}: expected {expected_sha}, got {actual_sha}"
            )
            assert file_path.stat().st_size == info["size_bytes"], (
                f"File size mismatch for {filename}: expected {info['size_bytes']}, got {file_path.stat().st_size}"
            )

    def test_baseline_3k_files_and_checksum_invariance(
        self, baseline_checksums_manifest: Dict[str, Any]
    ):
        """Assert tests/data/ 3,000-row baseline remains untouched and identical to golden checksums."""
        assert BASELINE_DIR.exists(), f"Baseline directory {BASELINE_DIR} missing!"
        assert baseline_checksums_manifest.get("total_records") == 3000

        total_baseline_rows = 0
        for filename, golden_sha in GOLDEN_BASELINE_HASHES.items():
            baseline_file = BASELINE_DIR / filename
            assert baseline_file.is_file(), f"Baseline file {filename} missing from tests/data/!"

            # 1. SHA-256 verification against golden hardcoded values
            actual_sha = compute_file_sha256(baseline_file)
            assert actual_sha == golden_sha, (
                f"Baseline file {filename} SHA-256 changed! Expected {golden_sha}, got {actual_sha}"
            )

            # 2. SHA-256 verification against tests/data/checksums.json manifest
            manifest_info = baseline_checksums_manifest["files"].get(filename)
            assert manifest_info is not None, f"Baseline manifest missing entry for {filename}"
            assert actual_sha == manifest_info["sha256"], (
                f"Baseline manifest mismatch for {filename}: {manifest_info['sha256']} != {actual_sha}"
            )

            # 3. Row count check
            with open(baseline_file, "r", encoding="utf-8") as f:
                row_count = sum(1 for line in f if line.strip())
            total_baseline_rows += row_count

        assert total_baseline_rows == 3000, (
            f"Baseline total rows mismatch: expected 3,000, got {total_baseline_rows}"
        )
