## `nx-cugraph` Benchmarks

### Overview

This directory contains a set of scripts designed to benchmark NetworkX with the `nx-cugraph` backend and deliver a report that summarizes the speed-up and runtime deltas over default NetworkX.

Our current benchmarks provide the following datasets:

| Dataset     | Nodes | Edges | Directed |
| --------    | ------- | ------- | ------- |
| netscience  | 1,461    | 5,484 | Yes |
| email-Eu-core  | 1,005    | 25,571 | Yes |
| amazon0302  | 262111  | 1234877 | Yes |
| cit-Patents  | 3,774,768    | 16,518,948 | Yes |
| hollywood  | 1,139,905    | 57,515,616 | No |
| soc-LiveJournal1  | 4,847,571    | 68,993,773 | Yes |



### Scripts

#### 1. `run-main-benchmarks.sh`
This script allows users to run a small set of commonly-used algorithms across multiple datasets and backends. All results are stored inside a sub-directory (`logs/`) and output files are named based on the combination of parameters for that benchmark.

NOTE:
 - If running with all algorithms and datasets using NetworkX without an accelerated backend, this script may take a few hours to finish running.
 - The `betweenness_centrality` benchmark will run with values `[10, 20, 50, 100, 500, 1000]` by default. You can specify only specific k-values to be run by editing `bc_k_values` (line 46) to be passed as a [pytest keyword object](https://docs.pytest.org/en/6.2.x/usage.html#specifying-tests-selecting-tests).

**Usage:**
 - Run with `--cpu-only`:
  ```bash
  ./run-main-benchmarks.sh --cpu-only
  ```
 - Run with `--gpu-only`:
  ```bash
  ./run-main-benchmarks.sh --gpu-only
  ```
 - Run without any arguments (all backends):
  ```bash
  ./run-main-benchmarks.sh
  ```

#### 2. `create_results_summary_page.py`
This script is designed to be run after `run-main-benchmarks.sh` in order to generate an HTML page displaying a results table comparing default NetworkX to nx-cugraph. The script also provides information about the current system, so it should be run on the machine on which benchmarks were run.

**Usage:**
  ```bash
  python create_results_summary_page.py > report.html
  ```
