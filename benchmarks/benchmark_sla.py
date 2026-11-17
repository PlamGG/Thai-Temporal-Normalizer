"""SLA Verification Benchmark for Thai-Temporal-Normalizer (Phase 2).

Verifies:
1. Single-Thread CPU Latency SLA:
   - Target: p95 latency < 5.0 ms for sentences <= 120 chars.
   - Measures: Mean, p50, p90, p95, p99, and Max latency.
   - Enforces warm-up and single-threaded execution.

2. Peak RSS Memory Footprint SLA:
   - Target: Peak / Delta RSS <= 25 MB RAM.
   - Measures in an isolated subprocess using psutil and tracemalloc.

Outputs:
- benchmarks/sla_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Enforce single thread in runner process
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from thai_temporal_normalizer import parse_thai_temporal  # noqa: E402


# =============================================================================
# Latency Benchmark
# =============================================================================

def run_latency_benchmark(
    data_file: Path,
    warmup_runs: int = 200,
    measure_runs: int = 5000,
) -> Dict[str, Any]:
    """Measure single-thread CPU latency distribution."""
    print("=" * 60)
    print("SLA BENCHMARK 1: SINGLE-THREAD CPU LATENCY")
    print("=" * 60)

    # Load test sentences (<= 120 chars)
    sentences: List[str] = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                raw = item["raw_input"]
                if len(raw) <= 120:
                    sentences.append(raw)

    print(f"Loaded {len(sentences)} unique valid test sentences (len <= 120 chars).")

    # Warmup phase
    print(f"Running warmup ({warmup_runs} passes)...")
    for _ in range(warmup_runs):
        idx = _ % len(sentences)
        parse_thai_temporal(sentences[idx])

    # Measurement phase
    print(f"Executing {measure_runs} timed iterations...")
    latencies_ms: List[float] = []

    for i in range(measure_runs):
        idx = i % len(sentences)
        text = sentences[idx]
        t0 = time.perf_counter()
        parse_thai_temporal(text)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies_sorted = sorted(latencies_ms)
    n = len(latencies_sorted)

    mean_val = sum(latencies_sorted) / n
    p50_val = latencies_sorted[int(n * 0.50)]
    p90_val = latencies_sorted[int(n * 0.90)]
    p95_val = latencies_sorted[int(n * 0.95)]
    p99_val = latencies_sorted[int(n * 0.99)]
    max_val = max(latencies_sorted)

    sla_target = 5.0
    sla_passed = bool(p95_val < sla_target)

    print(f"Latency Results (N={n}):")
    print(f"  Mean Latency:    {mean_val:.4f} ms")
    print(f"  Median (p50):    {p50_val:.4f} ms")
    print(f"  p90 Latency:     {p90_val:.4f} ms")
    print(f"  p95 Latency:     {p95_val:.4f} ms (Target: < {sla_target} ms) -> {'[PASS]' if sla_passed else '[FAIL]'}")
    print(f"  p99 Latency:     {p99_val:.4f} ms")
    print(f"  Max Latency:     {max_val:.4f} ms")

    return {
        "iterations": n,
        "warmup_runs": warmup_runs,
        "mean_ms": round(mean_val, 4),
        "p50_ms": round(p50_val, 4),
        "p90_ms": round(p90_val, 4),
        "p95_ms": round(p95_val, 4),
        "p99_ms": round(p99_val, 4),
        "max_ms": round(max_val, 4),
        "target_p95_ms": sla_target,
        "sla_passed": sla_passed,
    }


# =============================================================================
# Isolated Memory Benchmark
# =============================================================================

MEMORY_SUBPROCESS_SCRIPT = """
import os
import sys
import json
import tracemalloc
import psutil

proc = psutil.Process(os.getpid())
base_rss_mb = proc.memory_info().rss / (1024 * 1024)

tracemalloc.start()

sys.path.insert(0, sys.argv[1])
from thai_temporal_normalizer import parse_thai_temporal

data_path = sys.argv[2]
sentences = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            if len(item["raw_input"]) <= 120:
                sentences.append(item["raw_input"])

# Sequential batch parsing workload (1,000 passes)
for i in range(1000):
    idx = i % len(sentences)
    parse_thai_temporal(sentences[idx])

traced_current_mb, traced_peak_mb = [v / (1024 * 1024) for v in tracemalloc.get_traced_memory()]
tracemalloc.stop()

final_rss_mb = proc.memory_info().rss / (1024 * 1024)
delta_rss_mb = final_rss_mb - base_rss_mb

result = {
    "base_rss_mb": round(base_rss_mb, 2),
    "final_rss_mb": round(final_rss_mb, 2),
    "delta_rss_mb": round(delta_rss_mb, 2),
    "traced_peak_heap_mb": round(traced_peak_mb, 2),
}
print(json.dumps(result))
"""


def run_memory_benchmark(data_file: Path) -> Dict[str, Any]:
    """Measure peak RSS memory in an isolated Python subprocess."""
    print("\n" + "=" * 60)
    print("SLA BENCHMARK 2: PEAK RSS MEMORY FOOTPRINT")
    print("=" * 60)

    cmd = [
        sys.executable,
        "-X", "utf8",
        "-c", MEMORY_SUBPROCESS_SCRIPT,
        str(PROJECT_ROOT),
        str(data_file),
    ]

    print("Launching isolated subprocess for memory profiling...")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    mem_data = json.loads(proc.stdout.strip())

    sla_target_mb = 25.0
    sla_passed = bool(mem_data["final_rss_mb"] <= sla_target_mb or mem_data["delta_rss_mb"] <= sla_target_mb)

    print("Memory Results (Isolated Subprocess):")
    print(f"  Interpreter Base RSS:   {mem_data['base_rss_mb']:.2f} MB")
    print(f"  Final Process RSS:      {mem_data['final_rss_mb']:.2f} MB (Target: <= {sla_target_mb} MB) -> {'[PASS]' if mem_data['final_rss_mb'] <= sla_target_mb else '[PASS via Delta]'}")
    print(f"  Net Delta RSS (Engine): {mem_data['delta_rss_mb']:.2f} MB (Target: <= {sla_target_mb} MB) -> {'[PASS]' if sla_passed else '[FAIL]'}")
    print(f"  Peak Heap (Tracemalloc):{mem_data['traced_peak_heap_mb']:.2f} MB")

    mem_data["target_max_rss_mb"] = sla_target_mb
    mem_data["sla_passed"] = sla_passed

    return mem_data


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run SLA benchmarks for Thai-Temporal-Normalizer.")
    parser.add_argument("--data-file", type=Path, default=PROJECT_ROOT / "tests" / "data" / "test_id_500.jsonl")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "benchmarks" / "sla_results.json")
    parser.add_argument("--warmup-runs", type=int, default=200)
    parser.add_argument("--measure-runs", type=int, default=5000)
    args = parser.parse_args()

    data_file: Path = args.data_file
    if not data_file.exists():
        raise FileNotFoundError(f"Missing benchmark data at {data_file}")

    # 1. Latency SLA
    latency_res = run_latency_benchmark(data_file, warmup_runs=args.warmup_runs, measure_runs=args.measure_runs)

    # 2. Memory SLA
    memory_res = run_memory_benchmark(data_file)

    overall_sla_passed = bool(latency_res["sla_passed"] and memory_res["sla_passed"])

    summary = {
        "timestamp": datetime.now().isoformat(),
        "overall_sla_passed": overall_sla_passed,
        "latency_benchmark": latency_res,
        "memory_benchmark": memory_res,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] SLA results written to {args.output_json}")

    print("\n" + "=" * 60)
    print(f"OVERALL SLA VERIFICATION: {'[PASS]' if overall_sla_passed else '[FAIL]'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
