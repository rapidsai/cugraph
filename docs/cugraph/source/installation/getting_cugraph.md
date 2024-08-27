
# Getting cuGraph Packages

Start by reading the [RAPIDS Instalation guide](https://docs.rapids.ai/install)
and checkout the [RAPIDS install selector](https://rapids.ai/start.html) for a pick list of install options.


There are 4 ways to get cuGraph packages:
1. [Quick start with Docker Repo](#docker)
2. [Conda Installation](#conda)
3. [Pip Installation](#pip)
4. [Build from Source](./source_build.md)


<br>

## Docker
The RAPIDS Docker containers contain all RAPIDS packages, including all from cuGraph, as well as all required supporting packages. To download a RAPIDS container, please see the [Docker Hub page for rapidsai/base](https://hub.docker.com/r/rapidsai/base), choosing a tag based on the NVIDIA CUDA version you're running. Also, the [rapidsai/notebooks](https://hub.docker.com/r/rapidsai/notebooks) container provides a ready to run Docker container with example notebooks and data, showcasing how you can utilize all of the RAPIDS libraries: cuDF, cuML, and cuGraph.

<br>


## Conda
It is easy to install cuGraph using conda. You can get a minimal conda installation with [Miniconda](https://conda.io/miniconda.html) or get the full installation with [Anaconda](https://www.anaconda.com/download).

cuGraph Conda packages
 * cugraph - this will also import:
   * pylibcugraph
   * libcugraph
 * cugraph-service-client
 * cugraph-service-server
 * cugraph-dgl
 * cugraph-pyg
 * cugraph-equivariant
 * nx-cugraph

Replace the package name in the example below to the one you want to install.


Install and update cuGraph using the conda command:

```bash
conda install -c rapidsai -c conda-forge -c nvidia cugraph cuda-version=12.0
```

Alternatively, use `cuda-version=11.8` for packages supporting CUDA 11.

Note: This conda installation only applies to Linux and Python versions 3.10/3.11.

<br>

## PIP
cuGraph, and all of RAPIDS, is available via pip.

```
pip install cugraph-cu12 --extra-index-url=https://pypi.nvidia.com
```

Replace `-cu12` with `-cu11` for packages supporting CUDA 11.

Also available:
 * cugraph-dgl-cu12
 * cugraph-pyg-cu12
 * cugraph-equivariant-cu12
 * nx-cugraph-cu12

<br>
