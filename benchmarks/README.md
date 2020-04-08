# cuGraph benchmarks

## Overview

This directory contains source and configuration files for benchmarking
`cuGraph`.  The sources are currently intended to benchmark `cuGraph` via the
python API, but this is not a requirement, and future updates may include
benchmarks written in C++ or other languages.

The benchmarks here assume specific datasets are present in the `datasets` directory under the root of the `cuGraph` source tree.

## Prerequisites

* `pytest` and the `pytest-benchmark` plugin
* cugraph built and installed (or `cugraph` sources and built C++ extensions available on `PYTHONPATH`)
* *<show dependencies needed for GPU metrics (should be just pynvml)>*
* *<show optional dependencies to install for plotting>*

## Usage

* *<show example using pytest with pytest-benchmark plugin here>*
* *<show plotting example>*
