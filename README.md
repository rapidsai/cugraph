# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuGraph - GPU Graph Analytics</div>

[![Build Status](http://18.191.94.64/buildStatus/icon?job=cugraph-master)](http://18.191.94.64/job/cugraph-master/)  [![Documentation Status](https://readthedocs.org/projects/cugraph/badge/?version=latest)](https://cugraph.readthedocs.io/en/latest/)

The [RAPIDS](https://rapids.ai) cuGraph library is a collection of graph analytics that process GPU Dataframe - see [cuDF](https://github.com/rapidsai/cudf). cuGraph aims at provides a NetworkX-like API that will be familiar to data scientists, so they can now build GPU-accelerated workflows more easily.

 For more project details, see [rapids.ai](https://rapids.ai/).

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/cudf/blob/master/README.md) ensure you are on the `master` branch.



## Quick Start

Please see the [Demo Docker Repository](https://hub.docker.com/r/rapidsai/rapidsai/), choosing a tag based on the NVIDIA CUDA version you’re running. This provides a ready to run Docker container with example notebooks and data, showcasing how you can utilize all of the RAPIDS libraries: cuDF, cuML, and cuGraph.

## 



## Install cuGraph

### Conda (Coming Soon)

It is easy to install cuGraph using conda. You can get a minimal conda installation with [Miniconda](https://conda.io/miniconda.html) or get the full installation with [Anaconda](https://www.anaconda.com/download).

Install and update cuGraph using the conda command:

```bash
# CUDA 9.2
conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cugraph

# CUDA 10.0
conda install -c nvidia/label/cuda10.0 -c rapidsai/label/cuda10.0 -c numba -c conda-forge -c defaults cugraph
```

Note: This conda installation only applies to Linux and Python versions 3.6/3.7.

### Pip (Coming Soon)

It is easy to install cuGraph using pip. You must specify the CUDA version to ensure you install the right package.

```bash
# CUDA 9.2
pip install cugraph-cuda92

# CUDA 10.0.
pip install cugraph-cuda100
```






## Development Setup

The following instructions are for developers and contributors to cuGraph OSS development. These instructions are tested on Linux Ubuntu 16.04 & 18.04. Use these instructions to build cuGraph from source and contribute to its development.  Other operating systems may be compatible, but are not currently tested.

The following instructions are tested on Linux systems.

Compiler requirement:

* `gcc`     version 5.4+
* `nvcc`    version 9.2
* `cmake`   version 3.12



CUDA requirement:

* CUDA 9.2+
* NVIDIA driver 396.44+
* Pascal architecture or better

You can obtain CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).



Since `cmake` will download and build Apache Arrow you may need to install Boost C++ (version 1.58+) before running
`cmake`:

```bash
# Install Boost C++ for Ubuntu 16.04/18.04
$ sudo apt-get install libboost-all-dev
```

or

```bash
# Install Boost C++ for Conda
$ conda install -c conda-forge boost
```



## Building cuGraph from source

To install cuGraph from source, ensure the dependencies are met and follow the steps below:

- Clone the repository and submodules

```bash
CUGRAPH_HOME=$(pwd)/cugraph
git clone https://github.com/rapidsai/cugraph.git $CUGRAPH_HOME

# Next load all the submodules
cd $CUGRAPH_HOME
git submodule update --init --remote --recursive
```

- Create the conda development environment `cugraph_dev`

```bash
# create the conda environment (assuming in base `cugraph` directory)
conda env create --name cugraph_dev --file conda/environments/cugraph_dev.yml

# activate the environment
conda activate cugraph_dev 
```

- Create the conda development environment `cugraph_nightly`

If you wish to use nightly RAPIDS builds then you can use the following conda environment:

```bash
# create the conda environment (assuming in base `cugraph` directory)
conda env create --name cugraph_nightly --file conda/environments/cugraph_nightly.yml

# activate the environment
conda activate cugraph_nightly 
```



The environment can be updated as development includes/changes the depedencies. To do so, run:

```bash
conda env update --name cugraph_dev --file conda/environments/cugraph_dev.yml
conda activate cugraph_dev 
```



This installs the required `cmake`, `cudf`, `pyarrow` and other
dependencies into the `cugraph_dev` conda environment and activates it.

Build and install `libcugraph`. CMake depends on the `nvcc` executable being on your path or defined in `$CUDACXX`.

This project uses cmake for building the C/C++ library. To configure cmake,
run:

```bash
cd cpp	      # enter cpp directory
mkdir build   # create build directory for out-of-source build
cd build      # enter the build directory
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX 
```

To build the C/C++ code
```bash
make          #This should produce a shared library named `libcugraph.so`
make install  #The default locations are `$CMAKE_INSTALL_PREFIX/lib` and `$CMAKE_INSTALL_PREFIX/include/cugraph` respectively.
```

### C++ ABI issues

cuGraph builds with C++14 features.  By default, we build cuGraph with the latest ABI (the ABI changed with C++11).  The version of cuDF pointed to in the conda installation above is build with the new ABI.

If you see link errors indicating trouble finding functions that use C++ strings when trying to build cuGraph you may have an ABI incompatibility.

There are a couple of complications that may make this a problem:
* if you need to link in a library built with the old ABI, you may need to build the entire tool chain from source using the old ABI.
* if you build cudf from source (for whatever reason), the default behavior for cudf (at least through version 0.5.x) is to build using the old ABI.  You can build with the new ABI, but you need to follow the instructions in CUDF to explicitly turn that on.

If you must build cugraph with the old ABI, you can use the following command (instead of the cmake call above):

```bash
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CXX11_ABI=OFF
```
 

### Python package

Install the Python package to your Python path:

```bash
python setup.py install    # install cugraph python bindings
```

### Run tests
**C++ stand alone tests** 

From the build directory : `gtests/gdfgraph_test`


**Python tests with datasets** 

From cugraph's directory :
```bash
tar -zxvf src/tests/datasets.tar.gz -C /    # tests will look for this 'datasets' folder in '/'
pytest
```

### Documentation

Python API documentation can be generated from [docs](docs) directory.





------



## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aim to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<p align="center"><img src="img/rapids_arrow.png" width="80%"/></p>

### Apache Arrow on GPU

The GPU version of [Apache Arrow](https://arrow.apache.org/) is a common API that enables efficient interchange of tabular data between processes running on the GPU. End-to-end computation on the GPU avoids unnecessary copying and converting of data off the GPU, reducing compute time and cost for high-performance analytics common in artificial intelligence workloads. As the name implies, cuDF uses the Apache Arrow columnar data format on the GPU. Currently, a subset of the features in Apache Arrow are supported.
