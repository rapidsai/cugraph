# Benchmarks

## NetworkX vs. nx-cugraph
We ran several commonly used graph algorithms on both `networkx` and `nx-cugraph`. Here are the results


<figure>

![bench-image](../_static/bc_benchmark.png)

<figcaption style="text-align: center;">Results from running this <a
href="https://github.com/rapidsai/cugraph/blob/HEAD/benchmarks/nx-cugraph/pytest-based/bench_algos.py">Benchmark</a><span
class="title-ref"></span></figcaption>
</figure>

## Reproducing Benchmarks

Below are the steps to reproduce the results on your own.

1. Clone the latest <https://github.com/rapidsai/cugraph>

2. Follow the instructions to build and activate an environment

4. Install the latest `nx-cugraph` by following the [Installation Guide](installation.md)

5. Follow the instructions written in the README [here](https://github.com/rapidsai/cugraph/blob/HEAD/benchmarks/nx-cugraph/pytest-based)
