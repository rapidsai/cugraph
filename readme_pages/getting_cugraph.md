
# Getting cuGraph

There are 3 ways to get cuGraph :
1. [Quick start with Docker Repo](#docker)
2. [Conda Installation](#conda)
3. [Build from Source](../SOURCEBUILD.md)

<br><br>

## Docker 
Please see the [Docker Repository](https://hub.docker.com/r/rapidsai/rapidsai/), choosing a tag based on the NVIDIA CUDA version you’re running. This provides a ready to run Docker container with example notebooks and data, showcasing how you can utilize all of the RAPIDS libraries: cuDF, cuML, and cuGraph.

<br>

## Conda 
It is easy to install cuGraph using conda. You can get a minimal conda installation with [Miniconda](https://conda.io/miniconda.html) or get the full installation with [Anaconda](https://www.anaconda.com/download).

Install and update cuGraph using the conda command:

```bash
# CUDA 11.4
conda install -c nvidia -c rapidsai -c numba -c conda-forge cugraph cudatoolkit=11.4

# CUDA 11.5
conda install -c nvidia -c rapidsai -c numba -c conda-forge cugraph cudatoolkit=11.5

For CUDA > 11.5, please use the 11.5 environment
```

Note: This conda installation only applies to Linux and Python versions 3.8/3.9.