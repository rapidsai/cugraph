
# Getting cuGraph Packages

There are 4 ways to get cuGraph packages:
1. [Quick start with Docker Repo](#docker)
2. [Conda Installation](#conda)
3. [Pip Installation](#pip)
4. [Build from Source](#SOURCE)

Or checkout the [RAPIDS install selector](https://rapids.ai/start.html) for a pick list of install options.

<br>

## Docker
The RAPIDS Docker containers contain all RAPIDS packages, including all from cuGraph, as well as all required supporting packages.   To download a container, please see the [Docker Repository](https://hub.docker.com/r/rapidsai/rapidsai/), choosing a tag based on the NVIDIA CUDA version you’re running.  This provides a ready to run Docker container with example notebooks and data, showcasing how you can utilize all of the RAPIDS libraries: cuDF, cuML, and cuGraph.

<br>


## Conda
It is easy to install cuGraph using conda. You can get a minimal conda installation with [Miniconda](https://conda.io/miniconda.html) or get the full installation with [Anaconda](https://www.anaconda.com/download).

cuGraph Conda packages
 * cugraph - this will also import:
   * pylibcugraph
   * libcugraph
 * cugraph_service_client
 * cugraph_service_server
 * cugraph_dgl
 * cugraph_pyg

Replace the package name in the example below to the one you want to install.


Install and update cuGraph using the conda command:

```bash
conda install -c rapidsai -c numba -c conda-forge -c nvidia cugraph cudatoolkit=11.8
```

Note: This conda installation only applies to Linux and Python versions 3.8/3.10.

<br>

## PIP
cuGraph, and all of RAPIDS, is available via pip.

```
pip install cugraph-cu11 --extra-index-url=https://pypi.ngc.nvidia.com
```

pip packages for other packages are being worked and should be available in early 2023

<br>

## SOURCE
cuGraph can be build directly from source. First check to make sure you have or can configure a supported environment.
Instructions for building from source is in our [source build](./SOURCEBUILD.md) page.
