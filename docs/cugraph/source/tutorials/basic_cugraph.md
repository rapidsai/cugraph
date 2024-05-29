# Getting started with cuGraph

## Required hardware/software

CuGraph is part of [Rapids](https://docs.rapids.ai/user-guide) and has the following system requirements:
 * NVIDIA GPU, Volta architecture or later, with [compute capability](https://developer.nvidia.com/cuda-gpus) 7.0+
 * CUDA 11.2, 11.4, 11.5, 11.8, 12.0 or 12.2
 * Python version 3.9, 3.10, or 3.11
 * NetworkX >= version 3.3 or newer in order to use use [NetworkX Configs](https://networkx.org/documentation/stable/reference/backends.html#module-networkx.utils.configs) **This is required for use of nx-cuGraph, [see below](#cugraph-using-networkx-code).**

## Installation
The latest RAPIDS System Requirements documentation is located [here](https://docs.rapids.ai/install#system-req).

This includes several ways to set up cuGraph
* From Unix
    * [Conda](https://docs.rapids.ai/install#wsl-conda)
    * [Docker](https://docs.rapids.ai/install#wsl-docker)
    * [pip](https://docs.rapids.ai/install#wsl-pip)

* In windows you must install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) and then choose one of the following:
    * [Conda](https://docs.rapids.ai/install#wsl-conda)
    * [Docker](https://docs.rapids.ai/install#wsl-docker)
    * [pip](https://docs.rapids.ai/install#wsl-pip)

* Build From Source

To build from source, check each RAPIDS GitHub README for set up and build instructions. Further links are provided in the [selector tool](https://docs.rapids.ai/install#selector). If additional help is needed reach out on our [Slack Channel](https://rapids-goai.slack.com/archives/C5E06F4DC).

## CuGraph Using NetworkX Code
While the steps above are required to use the full suite of cuGraph graph analytics, cuGraph is now supported as a NetworkX backend using [nx-cugraph](https://docs.rapids.ai/api/cugraph/nightly/nx_cugraph/nx_cugraph/).
Nx-cugraph offers those with existing NetworkX code, a **zero code change** option with a growing list of supported algorithms.


## Cugraph API Example
Coming soon !


Until then, [the cuGraph notebook repository](https://github.com/rapidsai/cugraph/blob/main/notebooks/README.md) has many examples of loading graph data and running algorithms in Jupyter notebooks. The [cuGraph test code](https://github.com/rapidsai/cugraph/tree/main/python/cugraph/cugraph/tests) gives examples of python scripts settng up and calling cuGraph algorithms. A simple example of [testing the degree centrality algorithm](https://github.com/rapidsai/cugraph/blob/main/python/cugraph/cugraph/tests/centrality/test_degree_centrality.py) is a good place to start. Some of these examples show [multi-GPU tests/examples with larger data sets](https://github.com/rapidsai/cugraph/blob/main/python/cugraph/cugraph/tests/centrality/test_degree_centrality_mg.py) as well.
