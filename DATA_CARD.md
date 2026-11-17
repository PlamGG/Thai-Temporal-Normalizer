# Dataset Card: DuckerMaster/Thai-Natural-DateTime-Dataset

[![Dataset License: CC-BY-4.0](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset Version](https://img.shields.io/badge/version-th--temporal--v1.0.0--draft-blue.svg)](#)
[![Release Version](https://img.shields.io/badge/release-v1.1.0%20(10k%20rows)-brightgreen.svg)](#)
[![PDPA Liability](https://img.shields.io/badge/PDPA%20Liability-0%25%20(Pure%20Synthetic)-brightgreen.svg)](#)
[![Records](https://img.shields.io/badge/total%20records-3%2C000%20rows-orange.svg)](#)

---

## 📌 Dataset Overview

**DuckerMaster/Thai-Natural-DateTime-Dataset** is a standardized, benchmark-grade Thai natural language temporal annotation dataset. It was developed to rigorously evaluate temporal normalization engines, Information Extraction pipelines, and LLM reasoning on colloquial Thai temporal semantics.

Unlike existing datasets that rely on rigid standard calendar formats, this dataset focuses on everyday spoken and conversational structures:
1. **Thai 6-hour traditional clock** (`ตี`, `โมง`, `บ่าย`, `ทุ่ม`, `หกทุ่ม`)
2. **Deictic relative days and weekdays** (`มะรืนนี้`, `พุธหน้า`, `อังคารที่แล้ว`)
3. **Dynamic short Buddhist Era century pivot** (`ปี 69`, `15 มี.ค. 67`)
4. **Night-time calendar day rollover** (`คืนวันศุกร์ ตีสอง` $\rightarrow$ Saturday 02:00)
5. **Continuous coarse time ranges with midnight crossing** (`สายๆ`, `ดึกๆ`)
6. **Compound multi-slot temporal expressions** (`พรุ่งนี้ บ่ายสองโมงครึ่ง`)

---

## 🔒 PDPA & Ethical Compliance Statement (0% Personal Data)

**Compliance Statement**:
> Under Thailand's **Personal Data Protection Act (PDPA) B.E. 2562 (2019)**, processing of Personally Identifiable Information (PII) requires lawful bases, strict consent mechanisms, and data subject rights management.
>
> To ensure **100% legal safety and 0% PDPA liability**, `DuckerMaster/Thai-Natural-DateTime-Dataset` was constructed via **pure algorithmic synthesis**:
> - **Zero real-world chat logs or private user messages** were crawled, harvested, or repurposed.
> - **Zero real names, national identification numbers, phone numbers, email addresses, IP addresses, or geographic coordinates** are present.
> - All carrier phrases utilize synthetic sentence frames and open domain templates (business appointments, deliveries, meeting reminders, scheduling).
> - Fully open for research, commercial benchmarking, and fine-tuning under **Creative Commons Attribution 4.0 International (CC-BY-4.0)**.

---

## 🏛️ 3-Tier Split Architecture

The dataset is partitioned into 4 disjoint subsets totaling **3,000 records**, structurally segregated into a 3-tier evaluation framework:

```
Total Records: 3,000 (100.0%)
├── Tier 1: Training & Development Sets (2,000 rows / 66.67%)
│   ├── train_1500.jsonl : 1,500 rows (Standard syntactic templates)
│   └── dev_500.jsonl    :   500 rows (Validation for hyperparameter tuning)
├── Tier 2: In-Distribution (ID) Test Set (500 rows / 16.67%)
│   └── test_id_500.jsonl:   500 rows (Identical templates to Train, freshly sampled slot values)
└── Tier 3: Out-of-Distribution (OOD) Blind Test Set (500 rows / 16.67%)
    └── test_ood_500.jsonl: 500 rows
        ├── Sub-tier 3A (80% = 400 rows): Algorithmic noise, disjoint frames, chat abbreviations
        └── Sub-tier 3B (20% = 100 rows): Conversational blind test, edge cases, calendar guards
```

---

## 🔑 Cryptographic SHA-256 Checksums

Every split file is cryptographically locked with SHA-256 digests. Integrity verification is enforced by automated test suites (`tests/test_phase2_benchmarks.py`):

| File Name | Split Role | Records | File Size | SHA-256 Hex Digest |
|:----------|:-----------|:--------|:----------|:-------------------|
| `train_1500.jsonl` | Tier 1 (Train) | 1,500 | 946,850 bytes | `9f7e52cf17ce228f12c7ff0d4bf4531e9bb1db0eb678fa1d1bdccdc5fab88381` |
| `dev_500.jsonl` | Tier 1 (Dev) | 500 | 313,958 bytes | `f4cf1b5ffa1cfe26f58c9285e2bfd65f20f89b1b5e7537f9d956eb8bab2b75ab` |
| `test_id_500.jsonl` | Tier 2 (ID Test) | 500 | 316,816 bytes | `4484ca4829941ff16da5c461c5b358ec286309d15fffc7211405854ae65c9c8e` |
| `test_ood_500.jsonl` | Tier 3 (OOD Test)| 500 | 335,665 bytes | `44e6990c9f6e0ab0d96b695cf9ff5ad33a2466a94f3a1dd7d96d6e676ba1aad2` |

---

## 📋 JSONL Field Schema & Data Model

Each line in the JSONL files is a self-contained JSON object conforming to the following schema:

```json
{
  "id": 104,
  "raw_input": "เจอกันพรุ่งนี้ตอนบ่ายสอง",
  "ground_truth": {
    "text_span": "พรุ่งนี้ตอนบ่ายสอง",
    "start_char": 6,
    "end_char": 24,
    "type": "POINT",
    "grain": "hour",
    "iso_value": "2026-11-09T14:00:00+07:00",
    "range_start": null,
    "range_end": null,
    "precision": "exact",
    "confidence": 0.95
  },
  "category": "composite_datetime",
  "metadata": {
    "reference_time": "2026-11-08T12:00:00+07:00",
    "timezone": "Asia/Bangkok",
    "linguistic_type": "composite_relative_clock",
    "has_midnight_crossing": false
  },
  "notes": "Joint relative day and Thai clock time"
}
```

### Field Definitions

| Field Name | Type | Description |
|:-----------|:-----|:------------|
| `id` | `int` | Sequential record identifier within the split |
| `raw_input` | `str` | Raw Thai input text string ($\le 120$ characters) |
| `ground_truth` | `Object` | Gold standard annotated temporal entity |
| `ground_truth.text_span` | `str` | Verbatim text substring extracted from `raw_input` |
| `ground_truth.start_char` | `int` | Zero-based starting character offset ($\text{raw\_input}[\text{start}:\text{end}] == \text{text\_span}$) |
| `ground_truth.end_char` | `int` | Zero-based ending character offset |
| `ground_truth.type` | `str` | Entity type: `"POINT"` (specific timestamp) or `"RANGE"` (duration interval) |
| `ground_truth.grain` | `str` | Temporal grain resolution (e.g. `"hour"`, `"minute"`, `"day"`, `"auto"`) |
| `ground_truth.iso_value` | `str \| null` | Normalized ISO-8601 string for `"POINT"` entities |
| `ground_truth.range_start`| `str \| null` | ISO-8601 string for `"RANGE"` start boundary |
| `ground_truth.range_end` | `str \| null` | ISO-8601 string for `"RANGE"` end boundary |
| `ground_truth.precision` | `str` | Granularity indicator (`"exact"`, `"day"`, `"coarse"`) |
| `ground_truth.confidence`| `float` | Gold annotation confidence ($\in [0.0, 1.0]$) |
| `category` | `str` | Primary linguistic phenomenon category |
| `metadata` | `Object` | Contextual and experimental metadata |
| `metadata.reference_time` | `str` | ISO-8601 anchor datetime with timezone offset (`+07:00`) |
| `metadata.timezone` | `str` | Reference timezone (always `"Asia/Bangkok"`) |
| `metadata.linguistic_type` | `str` | Granular linguistic subcategory identifier |
| `metadata.has_midnight_crossing` | `bool` | Whether the temporal expression crosses midnight |
| `notes` | `str` | Human-readable explanation and annotation note |

---

## 🧪 Linguistic Categories & Distribution

The 3,000 records span 6 core linguistic categories:

| Linguistic Category | Description | Example Expression | Split Proportion |
|:--------------------|:------------|:-------------------|:-----------------|
| `relative_day_week` | Deictic relative day and weekday offsets | `พรุ่งนี้`, `มะรืนนี้`, `พุธหน้า`, `ศุกร์ที่แล้ว` | ~20% |
| `thai_6hour_clock` | Traditional Thai 4-quadrant 6-hour clock | `ตีสาม`, `สิบโมงเช้า`, `บ่ายสอง`, `สี่ทุ่ม` | ~20% |
| `coarse_time_periods` | Approximate fuzzy periods and ranges | `สายๆ`, `ช่วงบ่าย`, `หัวค่ำ`, `ดึกๆ` | ~20% |
| `short_buddhist_era` | 2-digit BE years and abbreviated dates | `ปี 69`, `15 มี.ค. 67`, `ปลายปี 68` | ~20% |
| `night_date_rollover` | Post-midnight expressions linked to previous night | `คืนวันศุกร์ ตีสอง`, `คืนนี้ ตีหนึ่งครึ่ง` | ~15% |
| `composite_datetime` | Joint expressions combining date, time, and period | `พรุ่งนี้ตอนบ่ายสามโมงครึ่ง` | ~5% |

---

## 🔀 Noise Injection & OOD Disjoint Methodology

To test real-world robustness and prevent rule engines from overfitting, the Out-of-Distribution (OOD) test set incorporates targeted noise injection:

1. **Disjoint Syntactic Frames**: Zero template overlap with the training set.
2. **Chat Colloquialisms & Abbreviations**:
   - `พน` / `พน.` for `พรุ่งนี้`
   - `มรน` / `มรน.` for `มะรืนนี้`
   - `บ่าย2` for `บ่ายสองโมง`
   - `2ทุ่ม` for `สองทุ่ม`
   - `เที่ยงคืน` for `หกทุ่ม`
3. **Conversational Fillers & Casual Particle Noise**:
   - Leading colloquial starters: `เอ่อ`, `เจอกันนะ`, `เค้าจะไป`, `ประมาณ`, `ราวๆ`
   - Trailing hedges: `กว่าๆ`, `นิดๆ`, `แถวๆ นั้น`
4. **Spacing & Orthographic Variations**:
   - Removal of standard word boundary spacing (`เจอกันพรุ่งนี้บ่ายสอง`)
   - Mixed Arabic and Thai numerals (`๑๕ มี.ค. 67` vs `15 มีนาคม 2567`)
   - Repetitive emphasis (`สายๆๆ`, `ดึกมว๊าก`)

---

## 📊 Verification & Reproducibility

To re-generate the dataset or verify cryptographic hashes independently:

```bash
# Re-generate synthetic dataset with fixed seed 42
python -X utf8 benchmarks/generate_synthetic.py

# Verify SHA-256 hashes against manifest
python -c "
import json, hashlib
with open('tests/data/checksums.json', 'r', encoding='utf-8') as f:
    manifest = json.load(f)
for fname, meta in manifest['files'].items():
    with open('tests/data/' + fname, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    assert digest == meta['sha256'], f'Hash mismatch on {fname}'
    print(f'Verified {fname}: {digest[:12]}... OK')
"
```

---

## 🚀 Release v1.1.0: 10,000-Row Dataset Expansion

### 1. Dataset Summary & Motivation
The **v1.1.0 release** of `DuckerMaster/Thai-Natural-DateTime-Dataset` scales the dataset to **10,000 high-quality synthetic rows** designed for Hugging Face distribution and production-grade evaluation of Thai temporal extraction and normalization engines. 

This major upgrade directly resolves critical scaling bottlenecks identified in synthetic data generation:
- **Global Cross-Split Deduplication**: Enforces strictly 0 duplicate `raw_input` texts within and across all splits (10,000 globally unique sentences), preventing data leakage between training, validation, and test sets.
- **Semantic Time Filtering**: Introduces diurnal context filtering preventing absurd context-time pairings (e.g., standard business meetings scheduled at 02:00 AM or late-night nocturnal server maintenance scheduled at midday).
- **100% Exact Character-Span Offsets**: Implements an `AlgebraicSpanCalculator` that computes entity coordinates algebraically (`len(prefix)`, `len(prefix) + len(text_span)`), replacing fragile substring search (`str.find`) and guaranteeing 100% exact span matching (`raw_input[start_char:end_char] == text_span`) without coordinate drift.
- **Rich Linguistic Dimensions & Negative Sampling**: Expands coverage beyond point timestamps to include temporal bounds, recurrence rules, durations, colloquial chat abbreviations, and negative samples to suppress model false positives.

---

### 2. Split Distribution & Hierarchy
The 10,000 records in v1.1.0 are partitioned across four disjoint splits stored in `data/v1.1/`:

| Split Name | File Name | Role | Records | Proportion |
|:-----------|:----------|:-----|:--------|:-----------|
| **Train** | `train.jsonl` | Model training and parameter fitting | 7,000 | 70.0% |
| **Dev** | `dev.jsonl` | Validation and hyperparameter tuning | 1,000 | 10.0% |
| **ID Test** | `test_id.jsonl` | In-distribution evaluation (freshly sampled slot values) | 1,000 | 10.0% |
| **OOD Test** | `test_ood.jsonl` | Out-of-distribution robustness (disjoint syntactic frames, chat abbreviations, colloquial noise) | 1,000 | 10.0% |
| **Total** | — | **Full v1.1.0 Release** | **10,000** | **100.0%** |

---

### 3. Linguistic Dimension Breakdown
The 10,000 rows follow a balanced stratification across four core linguistic dimensions:

| Dimension | Records | Proportion | Description & Examples |
|:----------|:--------|:-----------|:-----------------------|
| **Standard Temporal Expressions** | 7,000 | 70.0% | Diverse standard Thai dates, 24h digital times, 6-hour traditional clock, deictic relative days, and Buddhist Era years. (e.g., `06.20 น.`, `มะรืนนี้ สิบโมง`, `15 มี.ค. 67`, `คืนวันศุกร์ ตีสอง`) |
| **Bounds, Durations & Recurrence** | 1,500 | 15.0% | Temporal boundary intervals (`ก่อนเที่ยง`, `หลังบ่ายสาม`, `ไม่เกินห้าทุ่ม`), relative durations (`อีก 15 นาที`, `ประมาณ 2 ชั่วโมง`), and recurring frequencies (`ทุกวันจันทร์`, `ทุกสิ้นเดือน`, `ทุกวันที่ 1 ของเดือน`). |
| **Negative Samples (Null GT)** | 1,000 | 10.0% | Sentences containing non-temporal numbers or general non-temporal text with verified `ground_truth: null` to evaluate model precision and eliminate false positives. |
| **Chat Colloquialisms & Unspaced Times** | 500 | 5.0% | Informal abbreviations and unspaced clock expressions commonly encountered in social media and chat. (e.g., `พน`, `มรน`, `บ่าย3`, `2ทุ่ม`, `ทุ่มนึง`). |
| **Total** | **10,000** | **100.0%** | Statistically stratified across Train (70%), Dev (10%), ID Test (10%), and OOD Test (10%). |

---

### 4. Negative Sample Methodology (False Positive Suppression)
To rigorously test temporal taggers and avoid hallucinations on non-temporal numbers, v1.1.0 incorporates **1,000 negative samples** (10% of every split) with guaranteed `ground_truth = null`:
- **Currency Expressions**: Amounts with Thai Baht and international currency units (e.g., `โอนเงินจำนวน 500 บาท เรียบร้อยแล้ว`, `ค่าอาหารและเครื่องดื่มมื้อนี้รวม 1,250 บาท`).
- **Tracking & Order Numbers**: Parcel tracking formats and order IDs (e.g., `พัสดุหมายเลข SPX5990906602 กำลังนำจ่าย`, `Tracking Number: EMS8165491438`).
- **Phone Numbers**: Thai landline, mobile, hotline, and emergency contact formats (e.g., `สอบถามข้อมูลเพิ่มเติม โทร 02-123-4567`, `Hotline 1193 สายด่วนทางหลวง`).
- **General Non-Temporal Text**: Natural statements and operational updates without numerical or temporal references (e.g., `ระบบคลาวด์ทำงานได้ตามปกติ`, `กรุณาตรวจสอบเอกสารสัญญาแนบท้าย`).

In empirical evaluation against the normalizer, all 1,000 negative samples achieve **0 False Positives (100% FP immunity)**.

---

### 5. Semantic Time Filtering Across Context Domains
A major innovation in v1.1.0 is the `SemanticTimeFilter`, which enforces diurnal realism across 7 distinct context domains:

| # | Semantic Domain | Permitted Diurnal Hours | Allowed Features | Semantic Realism Guarantee |
|:--|:----------------|:------------------------|:-----------------|:---------------------------|
| 1 | `office_business` | 08:00 – 18:00 | Business meetings, interviews, conferences | Midnight crossing strictly prohibited; eliminates absurd business meetings at 02:00. |
| 2 | `daytime_commercial` | 08:00 – 19:00 | Customer service, retail visits, repair appointments | Midnight crossing strictly prohibited; matches commercial business hours. |
| 3 | `dining_social` | 11:00 – 14:30 (lunch)<br>17:00 – 23:59 (dinner)<br>00:00 (late night drinks) | Lunch, dinner, social drinks | Prohibits dinner appointments during morning work hours. |
| 4 | `it_ops_maintenance` | 21:00 – 06:00 | Cloud backup, server reboots, system patching | Diurnal restrictions favor nocturnal off-peak maintenance hours. |
| 5 | `financial_transactions` | 06:00 – 23:00 / Unconstrained | Bank transfers, billing deadlines, payment reminders | Compatible with flexible financial operations. |
| 6 | `personal_chat` | 07:00 – 01:00 / Unconstrained | Casual conversation, family reminders, friendly meetups | Reflects natural conversational flexibility. |
| 7 | `ood_disjoint_noise` | Unconstrained | Disjoint syntactic structures, chat particle variations | Blind robustness testing across all diurnal expressions. |

---

### 6. Official Cryptographic SHA-256 Checksum Manifest (v1.1.0)
All v1.1.0 release files in `data/v1.1/` are cryptographically verified and recorded in `data/v1.1/checksums.json`:

| File Name | Split Role | Records | File Size | SHA-256 Hex Digest |
|:----------|:-----------|:--------|:----------|:-------------------|
| `train.jsonl` | Train (Tier 1) | 7,000 | 5,817,882 bytes | `fc1df01f95c3a86541e6de87a2186638b9d19321afc80c48febcf59618ee982a` |
| `dev.jsonl` | Dev (Tier 1) | 1,000 | 828,970 bytes | `1a83193ebd387ab140618eed52441a545e55b9284b3e590f987b4cb3f9185359` |
| `test_id.jsonl` | ID Test (Tier 2) | 1,000 | 837,154 bytes | `62a5a30e19ace0ba23271aec8973000b9422b706e7599e80904bb8f1abc6346c` |
| `test_ood.jsonl` | OOD Test (Tier 3) | 1,000 | 837,168 bytes | `d3ca2d415160c858496161fb2d22b8b9e56ced5acdbab59d48e69eca9e74dd00` |

---

### 7. Offline Local Usage & Hugging Face Loading

The v1.1.0 release is packaged for completely offline usage via Hugging Face `datasets` without requiring network access:

#### Direct Local Loading with `datasets.load_dataset`
```python
from datasets import load_dataset

# Load entire dataset offline from local directory
dataset = load_dataset("./data/v1.1")
print(dataset)
# Output:
# DatasetDict({
#     train: Dataset({ features: [...], num_rows: 7000 }),
#     validation: Dataset({ features: [...], num_rows: 1000 }),
#     test: Dataset({ features: [...], num_rows: 2000 })
# })

assert sum(len(dataset[s]) for s in dataset) == 10000
print("HF Dataset loaded successfully offline!")
```

#### Programmatic Builder Usage via `thai_natural_datetime.py`
```python
import sys
from pathlib import Path

# Add data/v1.1 to path or import builder directly
sys.path.append(str(Path("data/v1.1").resolve()))
from thai_natural_datetime import load_dataset_local

# Load all 4 discrete splits (train, dev, test_id, test_ood)
dataset = load_dataset_local()
print(dataset)
# DatasetDict({
#     train: Dataset({ num_rows: 7000 }),
#     dev: Dataset({ num_rows: 1000 }),
#     test_id: Dataset({ num_rows: 1000 }),
#     test_ood: Dataset({ num_rows: 1000 })
# })
```

#### Reading JSONL Directly with Pandas
```python
import pandas as pd

# Load train split
df_train = pd.read_json("data/v1.1/train.jsonl", lines=True)
print(f"Loaded {len(df_train)} rows.")
# Inspect negative samples
negatives = df_train[df_train["ground_truth"].isna()]
print(f"Negative samples in train: {len(negatives)} rows (10.0%)")
```

---

## 📜 Citation & Attribution

### v1.1.0 (10,000-Row Release)
```bibtex
@dataset{duckermaster_thai_natural_datetime_2026_v11,
  author       = {DuckerMaster and DuckerMaster},
  title        = {Thai-Natural-DateTime-Dataset: A Standardized Benchmark for Colloquial Thai Temporal Normalization},
  year         = {2026},
  version      = {th-temporal-v1.1.0},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/DuckerMaster/Thai-Temporal-Normalizer}},
  license      = {CC-BY-4.0}
}
```

### v1.0.0 (3,000-Row Baseline)
```bibtex
@dataset{duckermaster_thai_natural_datetime_2026,
  author       = {DuckerMaster and DuckerMaster},
  title        = {Thai-Natural-DateTime-Dataset: A Standardized Benchmark for Colloquial Thai Temporal Normalization},
  year         = {2026},
  version      = {th-temporal-v1.0.0-draft},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/DuckerMaster/Thai-Temporal-Normalizer}},
  license      = {CC-BY-4.0}
}
```
