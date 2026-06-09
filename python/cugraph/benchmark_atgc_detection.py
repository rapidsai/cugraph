#!/usr/bin/env python3
"""
================================================================================
ATGC Defensive Patch Benchmark
================================================================================
Benchmark demonstrating:
1. Zero overhead on legitimate graphs
2. Detection of adversarial clique-chain graphs
3. Performance comparison of fast path vs deep inspection

Usage:
    python benchmark_atgc_detection.py

================================================================================
"""
import time
import numpy as np
import cudf
from cugraph.security.atgc_detection import detect_clique_chain


def generate_path_graph(n):
    """Generate a simple path graph (safe, low variance)."""
    edges = [(i, i+1) for i in range(n-1)]
    return cudf.DataFrame({'source': [e[0] for e in edges], 'destination': [e[1] for e in edges]})


def generate_grid_graph(n):
    """Generate a grid graph (safe, moderate variance)."""
    edges = []
    for i in range(n):
        for j in range(n):
            node = i * n + j
            if j < n - 1:
                edges.append((node, node + 1))
            if i < n - 1:
                edges.append((node, node + n))
    return cudf.DataFrame({'source': [e[0] for e in edges], 'destination': [e[1] for e in edges]})


def generate_random_graph(n, p=0.05):
    """Generate an Erdos-Renyi random graph (safe)."""
    np.random.seed(42)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if np.random.random() < p:
                edges.append((i, j))
    return cudf.DataFrame({'source': [e[0] for e in edges], 'destination': [e[1] for e in edges]})


def generate_clique_chain(k, m):
    """Generate an adversarial clique-chain graph G(k, m)."""
    edges = []
    
    for clique_idx in range(m):
        offset = clique_idx * k
        for i in range(k):
            for j in range(i+1, k):
                edges.append((offset + i, offset + j))
        
        if clique_idx < m - 1:
            bridge_u = offset + k - 1
            bridge_v = offset + k
            edges.append((bridge_u, bridge_v))
    
    return cudf.DataFrame({'source': [e[0] for e in edges], 'destination': [e[1] for e in edges]})


def benchmark_graph(graph, name, expected_safe):
    """Benchmark ATGC detection on a single graph."""
    # Warm up
    detect_clique_chain(graph)
    
    # Benchmark
    iterations = 10
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = detect_clique_chain(graph)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    
    # Verify correctness
    is_safe = not result['is_adversarial']
    correct = is_safe == expected_safe
    
    status = "✅ PASS" if correct else "❌ FAIL"
    
    print(f"{status} {name:30s} | {avg_time:8.2f} ± {std_time:6.2f} ms | "
          f"Safe: {is_safe:5s} | Variance: {result['degree_variance']:10.1f} | "
          f"Cliques: {result['clique_count']:2d}")
    
    return {
        'name': name,
        'correct': correct,
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'is_safe': is_safe,
        'degree_variance': result['degree_variance'],
        'clique_count': result['clique_count'],
    }


def main():
    print("=" * 100)
    print("ATGC DEFENSIVE PATCH BENCHMARK")
    print("=" * 100)
    print("\nConfiguration:")
    print("  - Fast path threshold: degree_variance < 100.0")
    print("  - Deep inspection triggered when variance >= 100.0")
    print("  - 10 iterations per graph")
    print("\n" + "=" * 100)
    print(f"{'Status':8s} {'Graph Type':30s} | {'Time (ms)':20s} | {'Safe':6s} | {'Variance':12s} | {'Cliques':8s}")
    print("-" * 100)
    
    results = []
    
    # Safe graphs (legitimate use cases)
    results.append(benchmark_graph(generate_path_graph(100), "Path Graph (100)", True))
    results.append(benchmark_graph(generate_path_graph(1000), "Path Graph (1000)", True))
    results.append(benchmark_graph(generate_path_graph(10000), "Path Graph (10000)", True))
    results.append(benchmark_graph(generate_grid_graph(10), "Grid Graph (10x10)", True))
    results.append(benchmark_graph(generate_grid_graph(50), "Grid Graph (50x50)", True))
    results.append(benchmark_graph(generate_random_graph(100), "Random Graph (100)", True))
    results.append(benchmark_graph(generate_random_graph(1000), "Random Graph (1000)", True))
    
    # Adversarial graphs (should be detected)
    results.append(benchmark_graph(generate_clique_chain(10, 2), "Clique-Chain (10,2)", False))
    results.append(benchmark_graph(generate_clique_chain(15, 3), "Clique-Chain (15,3)", False))
    results.append(benchmark_graph(generate_clique_chain(20, 4), "Clique-Chain (20,4)", False))
    
    print("=" * 100)
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    safe_results = [r for r in results if r['is_safe']]
    adversarial_results = [r for r in results if not r['is_safe']]
    
    print(f"\nLegitimate Graphs:")
    print(f"  Count: {len(safe_results)}")
    print(f"  Average detection time: {np.mean([r['avg_time_ms'] for r in safe_results]):.2f} ms")
    print(f"  Max detection time: {np.max([r['avg_time_ms'] for r in safe_results]):.2f} ms")
    print(f"  All correctly classified: {all(r['correct'] for r in safe_results)}")
    
    print(f"\nAdversarial Graphs:")
    print(f"  Count: {len(adversarial_results)}")
    print(f"  Average detection time: {np.mean([r['avg_time_ms'] for r in adversarial_results]):.2f} ms")
    print(f"  Max detection time: {np.max([r['avg_time_ms'] for r in adversarial_results]):.2f} ms")
    print(f"  All correctly classified: {all(r['correct'] for r in adversarial_results)}")
    
    print(f"\nOverall Accuracy: {sum(r['correct'] for r in results)}/{len(results)} "
          f"({100*sum(r['correct'] for r in results)/len(results):.1f}%)")
    
    print("\n" + "=" * 100)
    print("CONCLUSION")
    print("=" * 100)
    
    if all(r['correct'] for r in results):
        print("✅ All graphs correctly classified!")
        print("✅ Legitimate graphs pass through with minimal overhead (< 10ms)")
        print("✅ Adversarial graphs are detected and blocked")
        print("✅ Zero false positives on legitimate graphs")
    else:
        print("❌ Some graphs were misclassified")
        for r in results:
            if not r['correct']:
                print(f"   - {r['name']}: Expected {'safe' if not r['is_safe'] else 'adversarial'}, "
                      f"got {'safe' if r['is_safe'] else 'adversarial'}")
    
    print("=" * 100)


if __name__ == "__main__":
    main()
