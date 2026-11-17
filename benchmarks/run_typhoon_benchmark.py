import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

# 1. Load Environment
with open('Thai-Temporal-Normalizer/.env', encoding='utf-8') as f:
    env = dict(line.strip().split('=', 1) for line in f if '=' in line and not line.startswith('#'))

TYPHOON_KEY = env.get('TYPHOON_API_KEY')
TYPHOON_MODEL = 'typhoon-v2.5-30b-a3b-instruct'

# 2. Load 150 Baseline Dataset
dataset_path = 'Thai-Temporal-Normalizer/tests/data/baseline_150.jsonl'
records = []
with open(dataset_path, encoding='utf-8') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print(f"Loaded {len(records)} test cases from {dataset_path}")

SYSTEM_PROMPT = """You are an expert Thai temporal extraction engine.
Given a Thai temporal expression and reference time, extract the temporal meaning and return ONLY a valid JSON object with these keys:
{
  "type": "POINT" or "RANGE",
  "iso_value": "YYYY-MM-DDTHH:MM:SS+07:00" (or null if RANGE),
  "range_start": "YYYY-MM-DDTHH:MM:SS+07:00" (or null if POINT),
  "range_end": "YYYY-MM-DDTHH:MM:SS+07:00" (or null if POINT)
}
DO NOT explain. Output raw JSON only."""

def call_typhoon(text, ref_time):
    url = "https://api.opentyphoon.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TYPHOON_KEY}",
        "Content-Type": "application/json"
    }
    user_prompt = f'Expression: "{text}"\nReference Time: {ref_time}\nTimezone: Asia/Bangkok\nJSON:'
    payload = {
        "model": TYPHOON_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 120
    }
    
    t0 = time.perf_counter()
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            data = json.loads(resp.read().decode('utf-8'))
            content = data['choices'][0]['message']['content'].strip()
            # Clean markdown code blocks if any
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(content), elapsed_ms, None
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return None, elapsed_ms, str(e)

# Run benchmark
results = []
correct_count = 0
category_stats = {}
latencies = []

print("Running Typhoon benchmark on 150 items...")
for idx, r in enumerate(records, 1):
    raw_input = r['raw_input']
    gt = r['ground_truth']
    ref_time = r['metadata']['reference_time']
    cat = r['category']
    
    if cat not in category_stats:
        category_stats[cat] = {"total": 0, "correct": 0, "errors": 0}
    category_stats[cat]["total"] += 1
    
    pred_json, latency, err = call_typhoon(raw_input, ref_time)
    latencies.append(latency)
    
    is_correct = False
    if pred_json and not err:
        if gt['type'] == 'POINT':
            # Check ISO value match (date or datetime prefix)
            gt_val = gt['iso_value']
            pred_val = pred_json.get('iso_value')
            if pred_val and (pred_val == gt_val or pred_val[:10] == gt_val[:10]):
                is_correct = True
        elif gt['type'] == 'RANGE':
            # Check Range existence
            if pred_json.get('range_start') and pred_json.get('range_end'):
                is_correct = True
                
    if is_correct:
        correct_count += 1
        category_stats[cat]["correct"] += 1
    else:
        category_stats[cat]["errors"] += 1
        
    results.append({
        "id": r['id'],
        "input": raw_input,
        "category": cat,
        "ground_truth": gt,
        "prediction": pred_json,
        "latency_ms": latency,
        "is_correct": is_correct,
        "error": err
    })
    
    if idx % 25 == 0 or idx == len(records):
        print(f"[{idx}/{len(records)}] Processed. Running Acc: {correct_count/idx*100:.1f}%, Avg Latency: {sum(latencies)/len(latencies):.1f}ms")
    time.sleep(0.1) # polite rate limit

latencies.sort()
p50 = latencies[len(latencies)//2]
p95 = latencies[int(len(latencies)*0.95)]
mean_lat = sum(latencies)/len(latencies)

summary = {
    "model": TYPHOON_MODEL,
    "total": len(records),
    "correct": correct_count,
    "accuracy": correct_count / len(records),
    "error_rate": 1.0 - (correct_count / len(records)),
    "latency_mean_ms": mean_lat,
    "latency_p50_ms": p50,
    "latency_p95_ms": p95,
    "per_category": category_stats,
    "raw_results": results
}

output_path = 'Thai-Temporal-Normalizer/benchmarks/typhoon_benchmark_results.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n" + "="*50)
print(f"Typhoon Benchmark Completed!")
print(f"Overall Accuracy: {correct_count}/{len(records)} ({correct_count/len(records)*100:.2f}%)")
print(f"Error Rate: {(1-correct_count/len(records))*100:.2f}%")
print(f"Latency: Mean={mean_lat:.1f}ms, p50={p50:.1f}ms, p95={p95:.1f}ms")
print("Per Category Accuracy:")
for cat, s in category_stats.items():
    print(f"  - {cat}: {s['correct']}/{s['total']} ({s['correct']/s['total']*100:.1f}%)")
print("="*50)
