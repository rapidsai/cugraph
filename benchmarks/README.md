# cuGraph benchmarks

## Overview

This directory contains source and configuration files for benchmarking
`cuGraph`.  The sources are currently intended to benchmark `cuGraph` via the
python API, but this is not a requirement, and future updates may include
benchmarks written in C++ or other languages.

The benchmarks here assume specific datasets are present in the `datasets`
directory under the root of the `cuGraph` source tree.

## Prerequisites
### Python
* cugraph built and installed (or `cugraph` sources and built C++ extensions
  available on `PYTHONPATH`)

* rapids-pytest-benchmark pytest plugin (`conda install -c rlratzel
  rapids-pytest-benchmark`)
  * NOTE: the `rlratzel` channel is temporary! This plugin will eventually be
    moved to a more standard channel

* specific datasets installed in <cugraph>/datasets (see benchmark sources in
  this dir for details)

## Usage (Python)
### Python
* Run `pytest --help` (with the rapids-pytest-benchmark plugin installed) for
  the full list of options

* See also the `pytest.ini` file in this directory for examples of how to enable
  options by default and define marks

## Examples
### Python
* Run all the benchmarks and print their names on a separate line (`-v`), and generate a report to stdout
```
(rapids) user@machine:/cugraph/benchmarks> pytest -v
```

* Run all the benchmarks but stop if an error or failure is encountered
```
(rapids) user@machine:/cugraph/benchmarks> pytest -x
```

* Run all the benchmarks but do not reinit RMM with different configurations
```
(rapids) user@machine:/cugraph/benchmarks> pytest --no-rmm-reinit
```

* Show what benchmarks would be run with the given options, but do not run them
```
(rapids) user@machine:/cugraph/benchmarks> pytest --collect-only
```

* Show the markers that can be specified to include/omit benchmarks
```
(rapids) user@machine:/cugraph/benchmarks> pytest --markers
```

* Run all the benchmarks, generate a report to stdout, and save results for future comparisons
```
(rapids) user@machine:/cugraph/benchmarks> pytest -v --benchmark-autosave
```

* Run all the benchmarks as done above, but also compare to the last run
```
(rapids) user@machine:/cugraph/benchmarks> pytest -v --benchmark-autosave --benchmark-compare
```

* Run the benchmarks against only "small" datasets
```
(rapids) user@machine:/cugraph/benchmarks> pytest -v -m small
```

* Run the benchmarks against only "small" and "directed" datasets
```
(rapids) user@machine:/cugraph/benchmarks> pytest -v -m "small and directed"
```

* Run only the SSSP benchmarks against only "small" and "directed" datasets
```
(rapids) user@machine:/cugraph/benchmarks> pytest -v -m "small and directed" -k sssp
```

* Run only the SSSP and BFS benchmarks against only "small" and "directed" datasets
```
(rapids) user@machine:/cugraph/benchmarks> pytest -v -m "small and directed" -k "sssp or bfs"
```

* Run the benchmarks for everything except pagerank
```
(rapids) user@machine:/cugraph/benchmarks> pytest -v -k "not pagerank"
```

* Run only benchmarks for the "ETL" steps
```
(rapids) user@machine:/cugraph/benchmarks> pytest -v -m ETL
```

* Run benchmarks for only the "ETL" steps and generate a histogram plot
```
(rapids) user@machine:/cugraph/benchmarks> pytest -v -m ETL --benchmark-histogram
```

* Run the benchmarks but keep RMM managed memory enabled and pool allocation disabled
```
(rapids) user@machine:/cugraph/benchmarks> pytest -v -m "managedmem_on and poolallocator_off"
```
