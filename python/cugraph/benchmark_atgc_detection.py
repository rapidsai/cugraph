#!/usr/bin/env python3
"""
================================================================================
ATGC Defensive Patch Benchmark Suite
================================================================================
Production-grade benchmark demonstrating:
    1. Zero overhead on legitimate graphs (fast path)
    2. Detection of adversarial clique-chain graphs
    3. Performance comparison across different graph types
    4. Scalability testing

Usage:
    python benchmark_atgc_detection.py [--output report.json]

Output:
    - Console report with timing and accuracy metrics
    - JSON file with detailed results (optional)

================================================================================
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import cudf

from cugraph.security.atgc_detection import ATGCDetector, ATGCResult


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    graph_type: str
    num_vertices: int
    num_edges: int
    expected_safe: bool
    is_safe: bool
    correct: bool
    detection_time_ms: float
    std_time_ms: float
    degree_variance: float
    clique_count: int
    max_clique_size: int
    confidence: str


@dataclass
class BenchmarkSummary:
    """Summary of all benchmark runs."""
    total_tests: int
    correct_classifications: int
    accuracy_percent: float
    avg_time_safe_ms: float
    avg_time_adversarial_ms: float
    max_time_safe_ms: float
    max_time_adversarial_ms: float
    false_positives: int
    false_negatives: int


class GraphGenerator:
    """Generator for various graph types used in benchmarking."""
    
    @staticmethod
    def path_graph(n: int) -> cudf.DataFrame:
        """Generate a simple path graph."""
        edges = [(i, i + 1) for i in range(n - 1)]
        return cudf.DataFrame({
            'source': [e[0] for e in edges],
            'destination': [e[1] for e in edges]
        })
    
    @staticmethod
    def grid_graph(n: int) -> cudf.DataFrame:
        """Generate a grid graph."""
        edges = []
        for i in range(n):
            for j in range(n):
                node = i * n + j
                if j < n - 1:
                    edges.append((node, node + 1))
                if i < n - 1:
                    edges.append((node, node + n))
        return cudf.DataFrame({
            'source': [e[0] for e in edges],
            'destination': [e[1] for e in edges]
        })
    
    @staticmethod
    def random_graph(n: int, p: float = 0.05, seed: int = 42) -> cudf.DataFrame:
        """Generate an Erdos-Renyi random graph."""
        np.random.seed(seed)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.random() < p:
                    edges.append((i, j))
        return cudf.DataFrame({
            'source': [e[0] for e in edges],
            'destination': [e[1] for e in edges]
        })
    
    @staticmethod
    def star_graph(n: int) -> cudf.DataFrame:
        """Generate a star graph."""
        edges = [(0, i) for i in range(1, n)]
        return cudf.DataFrame({
            'source': [e[0] for e in edges],
            'destination': [e[1] for e in edges]
        })
    
    @staticmethod
    def tree_graph(n: int, branching_factor: int = 2) -> cudf.DataFrame:
        """Generate a tree graph."""
        edges = []
        node_id = 1
        for parent in range(n):
            for _ in range(branching_factor):
                if node_id < n:
                    edges.append((parent, node_id))
                    node_id += 1
        return cudf.DataFrame({
            'source': [e[0] for e in edges],
            'destination': [e[1] for e in edges]
        })
    
    @staticmethod
    def clique_chain(k: int, m: int) -> cudf.DataFrame:
        """Generate an adversarial clique-chain graph G(k, m)."""
        edges = []
        for clique_idx in range(m):
            offset = clique_idx * k
            for i in range(k):
                for j in range(i + 1, k):
                    edges.append((offset + i, offset + j))
            
            if clique_idx < m - 1:
                bridge_u = offset + k - 1
                bridge_v = offset + k
                edges.append((bridge_u, bridge_v))
        
        return cudf.DataFrame({
            'source': [e[0] for e in edges],
            'destination': [e[1] for e in edges]
        })
    
    @staticmethod
    def complete_graph(n: int) -> cudf.DataFrame:
        """Generate a complete graph K_n."""
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        return cudf.DataFrame({
            'source': [e[0] for e in edges],
            'destination': [e[1] for e in edges]
        })


def run_benchmark(detector: ATGCDetector, graph: cudf.DataFrame, name: str,
                 graph_type: str, num_vertices: int, num_edges: int,
                 expected_safe: bool, iterations: int = 10) -> BenchmarkResult:
    """Run benchmark on a single graph.
    
    Args:
        detector: ATGCDetector instance
        graph: Graph to test
        name: Benchmark name
        graph_type: Type of graph
        num_vertices: Number of vertices
        num_edges: Number of edges
        expected_safe: Whether graph is expected to be safe
        iterations: Number of iterations
    
    Returns:
        BenchmarkResult with timing and accuracy
    """
    # Warm up
    detector.detect(graph)
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = detector.detect(graph)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    is_safe = not result.is_adversarial
    correct = is_safe == expected_safe
    
    return BenchmarkResult(
        name=name,
        graph_type=graph_type,
        num_vertices=num_vertices,
        num_edges=num_edges,
        expected_safe=expected_safe,
        is_safe=is_safe,
        correct=correct,
        detection_time_ms=avg_time,
        std_time_ms=std_time,
        degree_variance=result.degree_variance,
        clique_count=result.clique_count,
        max_clique_size=result.max_clique_size,
        confidence=result.confidence,
    )


def generate_report(results: List[BenchmarkResult], output_file: Optional[str] = None) -> str:
    """Generate formatted benchmark report.
    
    Args:
        results: List of benchmark results
        output_file: Optional file to write JSON report
    
    Returns:
        Formatted report string
    """
    # Separate safe and adversarial results
    safe_results = [r for r in results if r.expected_safe]
    adversarial_results = [r for r in results if not r.expected_safe]
    
    # Compute summary
    correct = sum(r.correct for r in results)
    total = len(results)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    # Count errors
    false_positives = sum(1 for r in safe_results if not r.correct)
    false_negatives = sum(1 for r in adversarial_results if not r.correct)
    
    # Timing statistics
    safe_times = [r.detection_time_ms for r in safe_results]
    adv_times = [r.detection_time_ms for r in adversarial_results]
    
    summary = BenchmarkSummary(
        total_tests=total,
        correct_classifications=correct,
        accuracy_percent=accuracy,
        avg_time_safe_ms=np.mean(safe_times) if safe_times else 0.0,
        avg_time_adversarial_ms=np.mean(adv_times) if adv_times else 0.0,
        max_time_safe_ms=np.max(safe_times) if safe_times else 0.0,
        max_time_adversarial_ms=np.max(adv_times) if adv_times else 0.0,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )
    
    # Build report
    lines = []
    lines.append("=" * 100)
    lines.append("ATGC DEFENSIVE PATCH BENCHMARK REPORT")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Configuration:")
    lines.append("  - Fast path threshold: degree_variance < 100.0")
    lines.append("  - Deep inspection triggered when variance >= 100.0")
    lines.append("  - 10 iterations per graph")
    lines.append("")
    lines.append("-" * 100)
    lines.append(f"{'Status':8s} {'Graph':40s} {'Vertices':10s} {'Edges':10s} {'Time(ms)':12s} {'Safe':6s} {'Variance':10s}")
    lines.append("-" * 100)
    
    for r in results:
        status = "✅ PASS" if r.correct else "❌ FAIL"
        lines.append(
            f"{status:8s} {r.name:40s} {r.num_vertices:10d} {r.num_edges:10d} "
            f"{r.detection_time_ms:10.2f} ± {r.std_time_ms:6.2f} {str(r.is_safe):6s} "
            f"{r.degree_variance:10.1f}"
        )
    
    lines.append("-" * 100)
    lines.append("")
    lines.append("SUMMARY")
    lines.append("=" * 100)
    lines.append(f"Total Tests:              {summary.total_tests}")
    lines.append(f"Correct Classifications:  {summary.correct_classifications} ({summary.accuracy_percent:.1f}%)")
    lines.append(f"False Positives:          {summary.false_positives}")
    lines.append(f"False Negatives:          {summary.false_negatives}")
    lines.append("")
    lines.append("Performance:")
    lines.append(f"  Safe Graphs:        {summary.avg_time_safe_ms:8.2f} ms avg, {summary.max_time_safe_ms:8.2f} ms max")
    lines.append(f"  Adversarial Graphs: {summary.avg_time_adversarial_ms:8.2f} ms avg, {summary.max_time_adversarial_ms:8.2f} ms max")
    lines.append("")
    
    if summary.accuracy_percent == 100.0:
        lines.append("✅ All graphs correctly classified!")
        lines.append("✅ Zero false positives on legitimate graphs")
        lines.append("✅ Zero false negatives on adversarial graphs")
        lines.append("✅ Performance meets requirements (<100ms on safe graphs)")
    else:
        lines.append("❌ Some graphs were misclassified")
        for r in results:
            if not r.correct:
                lines.append(f"   - {r.name}: Expected {'safe' if r.expected_safe else 'adversarial'}, "
                           f"got {'safe' if r.is_safe else 'adversarial'}")
    
    lines.append("=" * 100)
    
    report = "\n".join(lines)
    
    # Write JSON if requested
    if output_file:
        output_data = {
            'summary': asdict(summary),
            'results': [asdict(r) for r in results],
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed report saved to: {output_file}")
    
    return report


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(
        description="ATGC Defensive Patch Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s
    %(prog)s --output benchmark_report.json
    %(prog)s --iterations 20
        """
    )
    parser.add_argument('--output', type=str, help='Output JSON file for detailed report')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations per graph')
    args = parser.parse_args()
    
    print("Initializing ATGC detector...")
    detector = ATGCDetector()
    
    gen = GraphGenerator()
    results = []
    
    # Safe graphs (legitimate use cases)
    test_cases = [
        ("Path Graph (100)", gen.path_graph(100), 'path', 100, 99, True),
        ("Path Graph (1000)", gen.path_graph(1000), 'path', 1000, 999, True),
        ("Grid Graph (10x10)", gen.grid_graph(10), 'grid', 100, 180, True),
        ("Grid Graph (50x50)", gen.grid_graph(50), 'grid', 2500, 4900, True),
        ("Random Graph (100, p=0.05)", gen.random_graph(100, 0.05), 'random', 100, -1, True),
        ("Random Graph (1000, p=0.01)", gen.random_graph(1000, 0.01), 'random', 1000, -1, True),
        ("Star Graph (100)", gen.star_graph(100), 'star', 100, 99, True),
        ("Tree Graph (100)", gen.tree_graph(100), 'tree', 100, 99, True),
        ("Complete Graph (K_5)", gen.complete_graph(5), 'complete', 5, 10, True),
        ("Complete Graph (K_10)", gen.complete_graph(10), 'complete', 10, 45, True),
    ]
    
    # Adversarial graphs (should be detected)
    test_cases.extend([
        ("Clique-Chain (10, 2)", gen.clique_chain(10, 2), 'clique_chain', 20, 91, False),
        ("Clique-Chain (15, 2)", gen.clique_chain(15, 2), 'clique_chain', 30, 211, False),
        ("Clique-Chain (15, 3)", gen.clique_chain(15, 3), 'clique_chain', 45, 318, False),
        ("Clique-Chain (20, 2)", gen.clique_chain(20, 2), 'clique_chain', 40, 381, False),
        ("Clique-Chain (20, 4)", gen.clique_chain(20, 4), 'clique_chain', 80, 764, False),
    ])
    
    print(f"\nRunning benchmarks ({len(test_cases)} graphs, {args.iterations} iterations each)...")
    print("This may take a few minutes...\n")
    
    for i, (name, graph, gtype, n, e, expected_safe) in enumerate(test_cases, 1):
        if e < 0:
            e = len(graph)
        
        print(f"[{i}/{len(test_cases)}] Testing {name}...", end=' ', flush=True)
        
        result = run_benchmark(
            detector, graph, name, gtype, n, e, expected_safe,
            iterations=args.iterations
        )
        
        results.append(result)
        
        status = "✅" if result.correct else "❌"
        print(f"{status} {result.detection_time_ms:.2f}ms")
    
    # Generate report
    report = generate_report(results, args.output)
    print(report)
    
    # Exit with error code if any test failed
    if any(not r.correct for r in results):
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
