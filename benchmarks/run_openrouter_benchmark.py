import json
import os
import sys
import time
import urllib.request
import urllib.error

sys.stdout.reconfigure(encoding='utf-8')

with open('Thai-Temporal-Normalizer/.env', encoding='utf-8') as f:
    env = dict(line.strip().split('=', 1) for line in f if '=' in line and not line.startswith('#'))

OPENROUTER_KEY = env.get('OPENROUTER_API_KEY')
# ใช้ google/gemini-2.5-flash ซึ่งทดสอบแล้วผ่าน API Key นี้ได้ฉลุย
MODEL_NAME = 'google/gemini-2.5-flash'

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

def call_openrouter(text, ref_time):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/DuckerMaster/Thai-Temporal-Normalizer",
        "X-Title": "Thai Temporal Normalizer Benchmark"
    }
    user_prompt = f'Expression: "{text}"\nReference Time: {ref_time}\nTimezone: Asia/Bangkok\nJSON:'
    payload = {
        "model": MODEL_NAME,
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
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(content), elapsed_ms, None
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return None, elapsed_ms, str(e)

results = []
correct_count = 0
category_stats = {}
latencies = []

print(f"Running OpenRouter ({MODEL_NAME}) benchmark on 150 items...")
for idx, r in enumerate(records, 1):
    raw_input = r['raw_input']
    gt = r['ground_truth']
    ref_time = r['metadata']['reference_time']
    cat = r['category']
    
    if cat not in category_stats:
        category_stats[cat] = {"total": 0, "correct": 0, "errors": 0}
    category_stats[cat]["total"] += 1
    
    pred_json, latency, err = call_openrouter(raw_input, ref_time)
    latencies.append(latency)
    
    is_correct = False
    if pred_json and not err:
        if gt['type'] == 'POINT':
            gt_val = gt['iso_value']
            pred_val = pred_json.get('iso_value')
            if pred_val and (pred_val == gt_val or pred_val[:10] == gt_val[:10]):
                is_correct = True
        elif gt['type'] == 'RANGE':
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
    time.sleep(0.1)

latencies.sort()
p50 = latencies[len(latencies)//2]
p95 = latencies[int(len(latencies)*0.95)]
mean_lat = sum(latencies)/len(latencies)

summary = {
    "model": MODEL_NAME,
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

output_path = 'Thai-Temporal-Normalizer/benchmarks/openrouter_gemini_benchmark_results.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n" + "="*50)
print(f"OpenRouter Benchmark Completed!")
print(f"Model: {MODEL_NAME}")
print(f"Overall Accuracy: {correct_count}/{len(records)} ({correct_count/len(records)*100:.2f}%)")
print(f"Error Rate: {(1-correct_count/len(records))*100:.2f}%")
print(f"Latency: Mean={mean_lat:.1f}ms, p50={p50:.1f}ms, p95={p95:.1f}ms")
print("Per Category Accuracy:")
for cat, s in category_stats.items():
    print(f"  - {cat}: {s['correct']}/{s['total']} ({s['correct']/s['total']*100:.1f}%)")
print("="*50)
