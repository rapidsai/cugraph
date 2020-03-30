# cuGraph Benchmarking

This directory contains utilities for writing and running benchmarks for cuGraph.


## Prerequisites

* An environment capable of running Python applications that use cuGraph.  A
  conda environment containing packages in the cuGraph env.yaml, or a RAPIDS
  `runtime` or `devel` Docker container.
  * NOTE: A RAPIDS `runtime` container contains the complete set of packages to
    satisfy every RAPIDS component, installed in a conda environment named
    `rapids`.  A `devel` continer also contains the packages needed for RAPIDS
    in a `rapids` conda environment, but also the complete toolchain used to
    build RAPIDS from source, the source files, and the intermediate build
    artifacts. `devel` containers are ideal for developers working on RAPIDS,
    and `runtime` containers are better suited for users of RAPIDS that don't
    need a toolchain or sources.
  * For developers using benchmarks to investigate performance-oriented changes,
    a `devel` container is probably a better choice.

* The existing benchmarks require datasets which can be obtained using the
  script in `<cugraph src dir>/datasets/get_test_data.sh`
  ```
  cd <cugraph src dir>/datasets
  ./get_test_data.sh
  ```

## Overview

The current benchmark running script, by default, assumes all benchmarks will be
run on the dataset name passed in.  To run against multiple datasets, multiple
invocations of the script are required.  The current implementation of the
script creates a single graph object from the dataset passed in and runs one or
more benchmarks on that - different datasets require new graphs to be created,
and the script currently only creates a single graph upfront.  The script also
treates the dataset read and graph creation as individual benchmarks and reports
results for those steps too.

There are two scripts to be aware of when running benchmarks; a python script
named `<cugraph src dir>/python/utils/run_benchmarks.py`, and a shell script
named `<cugraph src dir>/python/utils/run_benchmarks.sh`, both described more
below.  The python script is more general purpose in that it allows a user to
pass in a variety of different options and does not assume a particular set of
datasets exist, while the shell script is intended for easier use for common
invocations in that it assumes a specific set of options and datasets.  The
shell script assumes the datasets downloaded and installed by the
`<cugraph src dir>/datasets/get_test_data.sh` script are in place.


## Running benchmarks

### Quick start

The examples below assumes a bash shell in a RAPIDS `devel` container:
```
# get datasets
cd /rapids/cugraph/datasets
./get_test_data.sh

# run benchmarks
cd /rapids/cugraph/python/utils
./run_benchmarks.sh
```

### `<cugraph src dir>/python/utils/run_benchmarks.py`

The run_benchmarks.py script allows a user to run specific benchmarks on
specific datasets using different options.  The options vary based on the
benchmark being run, and typically have reasonable defaults.  For more
information, see the `--help` output of `run_benchmarks.py`.


## Writing new benchmarks

### Quick start

* Write a new function to run as a benchmark
  * The function need not perform any measurements - those will be handled by
    the `Benchmark` class which wraps it.

  * The function can take args with the understanding that they will need to be
    passed by the runner.  The runner already has a Graph object (created by
    reading the dataset) available.  Any other arg will need to be provided by
    either a custom command line arg, a global, or some other means.
    ```
      def my_new_benchmark(graphObj, arg1):
         graphObj.my_new_algo(arg1)
    ```

  * The above is an oversimplified example, and in the case above, the
    `my_new_algo()` method of the Graph object itself could serve as the
    callable which is wrapped by a `Benchmark` object (this is how most of the
    benchmarks are done in `run_benchmarks.py`).  A separate function like the
    above is only needed if a series of operations are to be benchmarked
    together.

* Add the new function to the runner

  * The easiest way is to write the function inside `run_benchmarks.py`. A more
    scalable way would be to write it in a separate module and `import` it into
    `run_benchmarks.py`.

  * Update `getBenchmarks()` in `run_benchmarks.py` to add an instance of a
    `Benchmark` object that wraps the new benchmark function - see
    `run_benchmarks.py` for more details and examples.

    * If the new benchmark function requires special args that are passed in via
      the command line, also update `parseCLI()` to add the new options.
      ```
      from my_module import my_new_benchmark

      def getBenchmarks(G, edgelist_gdf, args):
         benches = [
           Benchmark(name="my_new_benchmark",
                     func=my_new_benchmark,
                     args=(G, args.arg1)),
           ...
      ```

      if the new algo is the only operation to be benchmarked, and is perhaps
      just a new method in the cugraph module (like most other algos), then an
      easier approach could just be:

      ```
      def getBenchmarks(G, edgelist_gdf, args):
         benches = [
           Benchmark(name="my_new_benchmark",
                     func=cugraph.my_new_algo,
                     args=(G, args.arg1)),
           ...
      ```

### benchmark runner

The `run_benchmarks.py` script sets up a standard way to read in command-line
options (in most cases to be used to provide options to the underlying algos),
read the dataset specified to create an instance of a Graph class, and run the
specified algos on the Graph instance.  This script is intended to be modified
to customize the setup needed for different benchmarks, but idally only the
`getBenchmarks()` and sometimes the `parseCLI()` functions will change.

### The cuGraph Benchmark class

The `Benchmark` class is defined in `benchmark.py`, and it simply uses a series
of decorators to wrap the algo function call in timers and other calls to take
measurements to be included in the benchmark output.

The current metrics included are execution time (using the system monotonic
timer), GPU memory, and GPU utilization.  Each metric is defined in
`benchmark.py`, where new metrics can be added and applied.  The `Benchmark`
class simply defines the standard set of metrics that will be applied to each
algo, like so:
```
class Benchmark(WrappedFunc):
    wrappers = [logExeTime, logGpuMetrics, printLastResult]
```

See `benchmark.py` for more details about the `WrappedFunc` base class.
