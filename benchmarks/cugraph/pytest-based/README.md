# cuGraph benchmarks

## Overview

The sources are currently intended to benchmark `cuGraph` via the python API
but future updates may include benchmarks written in C++ or other languages.

The benchmarks here assume specific datasets are present in the `datasets`
directory under the root of the `cuGraph` source tree.

## Prerequisites
### Python
* cugraph built and installed (or `cugraph` sources and built C++ extensions
  available on `PYTHONPATH`)

* rapids-pytest-benchmark pytest plugin (`conda install -c rapidsai
  rapids-pytest-benchmark`)

* The benchmark datasets downloaded and installed in <cugraph>/datasets. Run the
script below from the <cugraph>/datasets directory:
```
cd <cugraph>/datasets
./get_test_data.sh --benchmark
```

## Usage (Python)
### Python
* Run `pytest --help` (with the rapids-pytest-benchmark plugin installed) for
  the full list of options

* See also the `pytest.ini` file in this directory for examples of how to enable
  options by default and define marks

## Examples
### Python
_**NOTE: these commands must be run from the `<cugraph_root>/benchmarks` directory.**_
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
