import numpy as np
import time

from numcompute.stats import mean
from numcompute.sort_search import topk


# benchamrking modules consist of functions that compare the performance of loop-based implementations vs vectorized implementations in NumCompute and this measures the execution time 
# of both approaches and calculate the speedup achieved by using vectorized operations.


# =========================
# HELPER: TIMER
# =========================

def time_function(func, *args, repeat=5):                   # time function takes a function and its arguments and runs it multiple times then returns the minimum execution time. 
                                                            # This helps to get a more accurate measure of the function's performance by reducing the impact of outliers.

    times = []

    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)

    return min(times) 


# =========================
# LOOP vs VECTORIZED (MEAN)
# =========================

def loop_mean(X):                                           # loop mean returns the mean of an array by iterating through each element and summing them up, then dividing by the count of elements.

    total = 0.0
    count = 0
    for x in X:
        total += x
        count += 1
    return total / count


def benchmark_mean():                                       # benchmark mean generates a large random array and then times both the loop-based mean and the vectorized mean from NumCompute, returning the results in a dictionary format.

    X = np.random.rand(1_000_000)

    t_loop = time_function(loop_mean, X)
    t_vec = time_function(mean, X)

    return {
        "task": "Mean",
        "loop_time": t_loop,
        "vectorized_time": t_vec,
        "speedup": t_loop / t_vec
    }


# =========================
# LOOP vs VECTORIZED (TOP-K)
# =========================

def loop_topk(X, k):                                        # loop topk returns the top k elements from an array by sorting it in reverse order and slicing the first k elements and is
                                                            #  not suitable to find the top k elements, especially for large arrays.

    return sorted(X, reverse=True)[:k]


def benchmark_topk():                                       # benchmark topk generates a large random array and then times both the loop-based top-k and the vectorized top-k from 
                                                            # NumCompute and this returning the results in a dictionary format.

    X = np.random.rand(1_000_000)
    k = 10

    t_loop = time_function(loop_topk, X, k)
    t_vec = time_function(topk, X, k)

    return {
        "task": "Top-K",
        "loop_time": t_loop,
        "vectorized_time": t_vec,
        "speedup": t_loop / t_vec
    }


# =========================
# RUN ALL BENCHMARKS
# =========================

def run_benchmarks():                                        # run benchmarks for mean and top-k functions collects the results and prints them in a formatted table
                                                             # showing the execution times and speedup for each task.
    results = []

    results.append(benchmark_mean())
    results.append(benchmark_topk())

    print("\n=== Benchmark Results ===")
    print(f"{'Task':<10} {'Loop (s)':<12} {'Vectorized (s)':<18} {'Speedup':<10}")

    for r in results:
        print(f"{r['task']:<10} {r['loop_time']:<12.6f} {r['vectorized_time']:<18.6f} {r['speedup']:<10.2f}")

    return results


if __name__ == "__main__":
    run_benchmarks()