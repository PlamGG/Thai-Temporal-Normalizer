"""Unit and integration tests for Phase 3 packaging, distribution, and documentation."""

from __future__ import annotations

import hashlib
import json
import tarfile
import zipfile
from pathlib import Path

from scripts.colab_verification import (
    benchmark_latency_sla,
    evaluate_baseline_150,
    verify_linguistic_phenomena,
    verify_security_guardrails,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_ROOT / "dist"
DATA_DIR = PROJECT_ROOT / "tests" / "data"


def test_distribution_packages_built() -> None:
    """Verify that both wheel and sdist exist in dist/ and have valid structures."""
    assert DIST_DIR.exists(), f"dist directory does not exist at {DIST_DIR}"

    wheels = list(DIST_DIR.glob("*.whl"))
    assert len(wheels) >= 1, "Expected at least one .whl file in dist/"
    wheel_path = wheels[0]
    assert "py3-none-any" in wheel_path.name, f"Expected pure Python wheel, got {wheel_path.name}"

    # Verify wheel can be read as valid zip and contains required modules
    with zipfile.ZipFile(wheel_path, "r") as zf:
        namelist = zf.namelist()
        assert any(n.endswith("normalizer.py") for n in namelist)
        assert any(n.endswith("models.py") for n in namelist)
        assert any(n.endswith("redos_guard.py") for n in namelist)
        assert any(n.endswith("METADATA") for n in namelist)

    sdists = list(DIST_DIR.glob("*.tar.gz"))
    assert len(sdists) >= 1, "Expected at least one .tar.gz source distribution in dist/"
    sdist_path = sdists[0]

    # Verify sdist can be read as valid tar.gz and contains pyproject.toml
    with tarfile.open(sdist_path, "r:gz") as tf:
        namelist = tf.getnames()
        assert any("pyproject.toml" in n for n in namelist)


def test_readme_documentation_completeness() -> None:
    """Verify README.md contains all mandated sections and empirical benchmark tables."""
    readme_path = PROJECT_ROOT / "README.md"
    assert readme_path.exists(), "README.md is missing"

    content = readme_path.read_text(encoding="utf-8")
    # Architectural and MWE coverage
    assert "6-Hour Clock Parsing" in content
    assert "Relative Day & Week Expressions" in content
    assert "Dynamic Short Buddhist Era (BE) Century Pivot" in content
    assert "Night-Time Day Rollover" in content
    assert "Coarse Time Range Resolution" in content
    assert "Custom Range Configuration Injection" in content

    # Benchmark tables and SLAs
    assert "dateparser" in content
    assert "pythainlp_wrapper" in content
    assert "simple_regex" in content
    assert "0.9960" in content  # ID Span F1
    assert "0.9203" in content  # OOD Span F1
    assert "0.2023 ms" in content  # p95 latency
    assert "22.75 MB" in content  # peak RSS

    # Security & API
    assert "ReDoS Guardrails" in content
    assert "parse_thai_temporal" in content
    assert "TemporalEntity" in content


def test_data_card_completeness_and_checksum_integrity() -> None:
    """Verify DATA_CARD.md exists, lists all checksums and PDPA compliance statement."""
    data_card_path = PROJECT_ROOT / "DATA_CARD.md"
    assert data_card_path.exists(), "DATA_CARD.md is missing"

    content = data_card_path.read_text(encoding="utf-8")
    assert "DuckerMaster/Thai-Natural-DateTime-Dataset" in content
    assert "CC-BY-4.0" in content
    assert "Personal Data Protection Act (PDPA) B.E. 2562" in content
    assert "3,000" in content

    checksum_file = DATA_DIR / "checksums.json"
    assert checksum_file.exists(), "checksums.json is missing"

    with open(checksum_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for fname, meta in manifest["files"].items():
        assert meta["sha256"] in content, f"Checksum for {fname} not found in DATA_CARD.md"
        actual_file = DATA_DIR / fname
        assert actual_file.exists()
        calc_sha = hashlib.sha256(actual_file.read_bytes()).hexdigest()
        assert calc_sha == meta["sha256"], f"Checksum mismatch for {fname}"


def test_colab_verification_end_to_end() -> None:
    """Verify all verification functions in colab_verification.py pass."""
    assert verify_linguistic_phenomena() is True
    assert evaluate_baseline_150() is True
    assert benchmark_latency_sla() is True
    assert verify_security_guardrails() is True
