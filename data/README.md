# Thai Natural DateTime Dataset (Quick Sample)

This repository includes a 10-sample preview of the dataset.

The full **10,000-row benchmark dataset** (Train, Validation, Test ID, and Blind Test OOD) is officially hosted on **Hugging Face Hub**:
👉 **[DuckerMaster/Thai-Natural-DateTime-Dataset](https://huggingface.co/datasets/DuckerMaster/Thai-Natural-DateTime-Dataset)**

### How to Load Full Dataset via Python:
```python
from datasets import load_dataset

# Load full 10k dataset directly from Hugging Face
dataset = load_dataset("DuckerMaster/Thai-Natural-DateTime-Dataset")

print(dataset["train"][0])
```

### Sample Record Preview:
```json
{
  "id": "train_0001",
  "raw_input": "ทีมงานยืนยันกำหนดการประชุมเป็น วันพรุ่งนี้ บ่ายสองโมงครึ่ง ผ่าน Google Meet",
  "ground_truth": {
    "text_span": "วันพรุ่งนี้ บ่ายสองโมงครึ่ง",
    "start_char": 31,
    "end_char": 58,
    "type": "POINT",
    "normalized_value": "2026-11-09T14:30:00+07:00",
    "grain": "minute",
    "iso_value": "2026-11-09T14:30:00+07:00"
  },
  "category": "compound_point"
}
```
