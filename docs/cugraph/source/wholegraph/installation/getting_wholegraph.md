
# Getting the WholeGraph Packages

Start by reading the [RAPIDS Instalation guide](https://docs.rapids.ai/install)
and checkout the [RAPIDS install selector](https://rapids.ai/start.html) for a pick list of install options.


There are 4 ways to get WholeGraph packages:
1. [Quick start with Docker Repo](#docker)
2. [Conda Installation](#conda)
3. [Pip Installation](#pip)
4. [Build from Source](./source_build.md)


<br>

## Docker
The RAPIDS Docker containers (as of Release 23.10) contain all RAPIDS packages, including WholeGraph, as well as all required supporting packages.   To download a container, please see the [Docker Repository](https://hub.docker.com/r/rapidsai/rapidsai/), choosing a tag based on the NVIDIA CUDA version you’re running.  This provides a ready to run Docker container with example notebooks and data, showcasing how you can utilize all of the RAPIDS libraries.

<br>


## Conda
It is easy to install WholeGraph using conda. You can get a minimal conda installation with [Miniconda](https://conda.io/miniconda.html) or get the full installation with [Anaconda](https://www.anaconda.com/download).

WholeGraph conda packages
 * libwholegraph
 * pylibwholegraph

Replace the package name in the example below to the one you want to install.


Install and update WholeGraph using the conda command:

```bash
conda install -c rapidsai -c conda-forge -c nvidia wholegraph cudatoolkit=11.8
```

<br>

## PIP
wholegraph, and all of RAPIDS, is available via pip.

```
pip install wholegraph-cu11 --extra-index-url=https://pypi.nvidia.com
```

<br>
