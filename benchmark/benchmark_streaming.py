"""
benchmark_streaming.py
-----------------------
Performance benchmark: Single Decision Tree vs Random Forest
under streaming conditions. Compares both accuracy and runtime.

Run:
    python benchmark/benchmark_streaming.py

Author: NumCompute-Stream
"""


# <-------- Imports --------->

import time
import numpy as np
import sys
import os


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numcompute_stream.trees import DecisionTreeClassifier
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.streaming import StreamTrainer, chunk_data
from numcompute_stream.metrics import accuracy


# ---------------------------------------------------------------------------
# Generate synthetic dataset
# ---------------------------------------------------------------------------


# <------- Generating synthetic dataset with non-linear decision boundary --------->

def generate_data(n_samples=2000, n_features=10, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    # Non-linear decision boundary
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 1.5).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Streaming benchmark runner
# ---------------------------------------------------------------------------


# <------- Run benchmark for a given model under streaming conditions --------->

def run_benchmark(model, X, y, chunk_size=100, model_name="Model", verbose=True):
    """Train model chunk-by-chunk and measure time + accuracy."""
    trainer = StreamTrainer(model, verbose=False)
    chunk_metrics = []
    chunk_times = []

    chunks = list(chunk_data(X, y, chunk_size=chunk_size, shuffle=True, random_state=0))

    eval_chunks = chunks[-3:]
    train_chunks = chunks[:-3]

    for X_c, y_c in train_chunks:
        t0 = time.perf_counter()
        trainer.fit_chunk(X_c, y_c)
        t1 = time.perf_counter()
        chunk_times.append(t1 - t0)

    # Evaluate on held-out chunks
    eval_accs = []
    for X_e, y_e in eval_chunks:
        preds = model.predict(X_e)
        eval_accs.append(accuracy(y_e, preds))

    # Compute mean accuracy and timing metrics
    
    mean_eval_acc = float(np.mean(eval_accs))
    mean_chunk_time = float(np.mean(chunk_times))
    total_time = float(np.sum(chunk_times))


# <------- Print benchmark results --------->

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Model: {model_name}")
        print(f"{'='*50}")
        print(f"  Chunks trained        : {len(train_chunks)}")
        print(f"  Chunk size            : {chunk_size}")
        print(f"  Mean eval accuracy    : {mean_eval_acc:.4f}")
        print(f"  Mean time per chunk   : {mean_chunk_time*1000:.2f} ms")
        print(f"  Total training time   : {total_time:.4f} s")
        print(f"{'='*50}")


# <------- Return benchmark results as a dictionary --------->

    return {
        "model": model_name,
        "mean_eval_accuracy": mean_eval_acc,
        "mean_chunk_time_ms": mean_chunk_time * 1000,
        "total_time_s": total_time,
        "n_chunks": len(train_chunks),
    }


# ---------------------------------------------------------------------------
# Vectorised vs loop baseline comparison
# ---------------------------------------------------------------------------


# <------- Benchmark NumPy vectorised mean vs Python loop mean --------->

def benchmark_vectorised_vs_loop(n=100_000):
    """Compare NumPy vectorised mean vs Python loop mean."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(n)

    # Vectorised
    t0 = time.perf_counter()
    vec_mean = np.mean(data)
    t1 = time.perf_counter()
    vec_time = t1 - t0

    # Loop-based
    t0 = time.perf_counter()
    total = 0.0
    for v in data:
        total += v
    loop_mean = total / n
    t1 = time.perf_counter()
    loop_time = t1 - t0


# <------- Print benchmark results with labels --------->

    print(f"\n{'='*50}")
    print("  Vectorised vs Loop Mean (n={:,})".format(n))
    print(f"{'='*50}")
    print(f"  Vectorised mean   : {vec_mean:.6f}  | time: {vec_time*1000:.4f} ms")
    print(f"  Loop-based mean   : {loop_mean:.6f}  | time: {loop_time*1000:.4f} ms")
    print(f"  Speedup           : {loop_time/vec_time:.1f}x")
    print(f"{'='*50}")


# <------- Return benchmark results as a dictionary --------->

    return {
        "vectorised_time_ms": vec_time * 1000,
        "loop_time_ms": loop_time * 1000,
        "speedup": loop_time / vec_time,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# <------- Main entry point to run benchmarks --------->

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  NumCompute-Stream Benchmarking Suite")
    print("=" * 50)

    # --- Generate data ---
    X, y = generate_data(n_samples=2000, n_features=10)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # --- Model benchmarks ---
    results = []

    r1 = run_benchmark(
        DecisionTreeClassifier(max_depth=5, random_state=0),
        X, y, chunk_size=100,
        model_name="DecisionTree (max_depth=5)"
    )
    results.append(r1)

    r2 = run_benchmark(
        RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0),
        X, y, chunk_size=100,
        model_name="RandomForest (n=10, max_depth=5)"
    )
    results.append(r2)


# <------- Run benchmark for Random Forest with more trees and deeper depth --------->

    r3 = run_benchmark(
        RandomForestClassifier(n_estimators=20, max_depth=7, random_state=0),
        X, y, chunk_size=100,
        model_name="RandomForest (n=20, max_depth=7)"
    )
    results.append(r3)

    # --- Summary table ---
    print("\n\nSUMMARY TABLE")
    print(f"{'Model':<40} {'Accuracy':>10} {'Time/chunk(ms)':>16} {'Total(s)':>10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['model']:<40} "
            f"{r['mean_eval_accuracy']:>10.4f} "
            f"{r['mean_chunk_time_ms']:>16.2f} "
            f"{r['total_time_s']:>10.4f}"
        )

    # --- Vectorised vs loop ---
    benchmark_vectorised_vs_loop(n=100_000)


    print("\n[Benchmark complete]")