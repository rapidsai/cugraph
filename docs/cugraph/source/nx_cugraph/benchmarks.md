# Benchmarks

## NetworkX vs. nx-cugraph
We ran several commonly used graph algorithms on both `networkx` and `nx-cugraph`. Here are the results


<figure>

<!-- ![bench-image](link) -->

<figcaption style="text-align: center;">Results of the <a
href="https://github.com/rapidsai/cugraph">Benchmark</a> including <span
class="title-ref">nx-cugraph</span></figcaption>
</figure>

## Reproducing Benchmarks

Below are the steps to reproduce the results on your workstation. These are documented in <https://github.com/rapidsai/cugraph/blob/HEAD/benchmarks/nx-cugraph/pytest-based>.

1. Clone the latest <https://github.com/rapidsai/cugraph>

2. Follow the instructions to build an environment

3. Activate the environment

4. Install the latest `nx-cugraph` by following the [guide](installation.md)

5. Follow the instructions written in the README here: `cugraph/benchmarks/nx-cugraph/pytest-based/`
