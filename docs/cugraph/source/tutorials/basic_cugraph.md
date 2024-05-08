## Required hardware/software

CuGraph is part of [Rapids](https://docs.rapids.ai/user-guide)
It has the following system requirements
 * NVIDIA GPU, Volta architecture or later, with [compute capability](https://developer.nvidia.com/cuda-gpus) 7.0+
 * CUDA 11.2, 11.4, 11.5, 11.8, 12.0 or 12.2
 * Python version 3.9, 3.10, or 3.11
 * NetworkX >= version 3.0 (version 3.3 or higher recommended) **This if for use of nx-cuGraph, [see below](#cugraph-using-networkx-code).**

## Installation
The latest RAPIDS System Requirements documentation is located [here](https://docs.rapids.ai/install#system-req)
This includes several ways to set up for cuGraph
* From Unix
    * [Conda](https://docs.rapids.ai/install#wsl-conda)
    * [Docker](https://docs.rapids.ai/install#wsl-docker)
    * [pip](https://docs.rapids.ai/install#wsl-pip)
* To use RAPIDS in windows you must install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
    * [Conda](https://docs.rapids.ai/install#wsl-conda)
    * [Docker](https://docs.rapids.ai/install#wsl-docker)
    * [pip](https://docs.rapids.ai/install#wsl-pip)
Build From Source

## CuGraph Using NetworkX Code
While the steps above are required to use the full suite of cuGraph graph analytics, cuGraph is now supported as a NetworkX backend using [nx-cugraph](https://docs.rapids.ai/api/cugraph/nightly/nx_cugraph/nx_cugraph/).
This is much simpler but limits users to the current but growing list of suppored algorithms.


## Cugraph API demo