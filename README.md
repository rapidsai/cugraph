# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuGraph - GPU Graph Analytics</div>

[![Build Status](http://18.191.94.64/buildStatus/icon?job=cugraph-master)](http://18.191.94.64/job/cugraph-master/)  [![Documentation Status](https://readthedocs.org/projects/cugraph/badge/?version=latest)](https://cugraph.readthedocs.io/en/latest/)

The [RAPIDS](https://rapids.ai) cuGraph library is a collection of graph analytics that process data found in GPU Dataframes - see [cuDF](https://github.com/rapidsai/cudf).  cuGraph aims to provide a NetworkX-like API that will be familiar to data scientists, so they can now build GPU-accelerated workflows more easily.

 For more project details, see [rapids.ai](https://rapids.ai/).

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/cudf/blob/master/README.md) ensure you are on the `master` branch.







## Getting cuGraph
### Intro
There are 4 ways to get cuGraph :
1. [Quick start with Docker Demo Repo](#quick)
1. [Conda Installation](#conda)
1. [Pip Installation](#pip)
1. [Build from Source](#source)



Building from source is currently the only viable option. Once version 0.6 is release, the other options will be available.  



## Quick Start {#quick}

Please see the [Demo Docker Repository](https://hub.docker.com/r/rapidsai/rapidsai/), choosing a tag based on the NVIDIA CUDA version you’re running. This provides a ready to run Docker container with example notebooks and data, showcasing how you can utilize all of the RAPIDS libraries: cuDF, cuML, and cuGraph.



### Conda{#conda}

It is easy to install cuGraph using conda. You can get a minimal conda installation with [Miniconda](https://conda.io/miniconda.html) or get the full installation with [Anaconda](https://www.anaconda.com/download).

Install and update cuGraph using the conda command:

```bash
# CUDA 9.2
conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cugraph

# CUDA 10.0
conda install -c nvidia/label/cuda10.0 -c rapidsai/label/cuda10.0 -c numba -c conda-forge -c defaults cugraph
```

Note: This conda installation only applies to Linux and Python versions 3.6/3.7.



### Pip {#pip}

It is easy to install cuGraph using pip. You must specify the CUDA version to ensure you install the right package.

```bash
# CUDA 9.2
pip install cugraph-cuda92

# CUDA 10.0.
pip install cugraph-cuda100
```





### Build from Source {#source}

The following instructions are for developers and contributors to cuGraph OSS development. These instructions are tested on Linux Ubuntu 16.04 & 18.04. Use these instructions to build cuGraph from source and contribute to its development.  Other operating systems may be compatible, but are not currently tested.

The cuGraph package include both a C/C++ CUDA portion and a python portion.  Both libraries need to be installed in order for cuGraph to operate correctly.  

The following instructions are tested on Linux systems.



#### Prerequisites

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



#### Build and Install the C/C++ CUDA components

To install cuGraph from source, ensure the dependencies are met and follow the steps below:

1) Clone the repository and submodules

  ```bash
  # Set the localtion to cuGraph in an environment variable CUGRAPH_HOME 
  export CUGRAPH_HOME=$(pwd)/cugraph

  # Download the cuGraph repo
  git clone https://github.com/rapidsai/cugraph.git $CUGRAPH_HOME

  # Next load all the submodules
  cd $CUGRAPH_HOME
  git submodule update --init --recursive
  ```



2) Create the conda development environment 

​	A)   Building the `master` branch uses the `cugraph_dev` environment

```bash
# create the conda environment (assuming in base `cugraph` directory)
# for CUDA 9.2
conda env create --name cugraph_dev --file conda/environments/cugraph_dev.yml

# for CUDA 10
conda env create --name cugraph_dev --file conda/environments/cugraph_dev_cuda10.yml

# activate the environment
conda activate cugraph_dev 

# to deactivate an environment
conda deactivate
```



​	B) Create the conda development environment `cugraph_nightly`  

If you are  on the latest development branch then you must use the `cugraph_nightly` environment.  The latest cuGraph code uses the latest cuDF features that might not yet be in the master branch.  To work off of the latest development branch, which could be unstable, use the nightly build environment.  

```bash
# create the conda environment (assuming in base `cugraph` directory)
conda env create --name cugraph_nightly --file conda/environments/cugraph_nightly.yml

# activate the environment
conda activate cugraph_nightly 

```




  - The environment can be updated as development includes/changes the dependencies. To do so, run:  



```bash
# for CUDA 9.2
conda env update --name cugraph_dev --file conda/environments/cugraph_dev.yml

# for CUDA 10
conda env update --name cugraph_dev --file conda/environments/cugraph_dev_cuda10.yml

conda activate cugraph_dev 
```





3) Build and install `libcugraph`. CMake depends on the `nvcc` executable being on your path or defined in `$CUDACXX`.

  This project uses cmake for building the C/C++ library. CMake will also automatically build and install nvGraph library (`$CUGRAPH_HOME/cpp/nvgraph`) which may take a few minutes. To configure cmake, run:

  ```bash
  # Set the localtion to cuGraph in an environment variable CUGRAPH_HOME 
  export CUGRAPH_HOME=$(pwd)/cugraph
  
  cd $CUGRAPH_HOME
  cd cpp	      		# enter cpp directory
  mkdir build   		# create build directory 
  cd build     		# enter the build directory
  cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX 

  # now build the code
  make -j				# "-j" starts multiple threads
  make install		# install the libraries 
  ```

The default installation  locations are `$CMAKE_INSTALL_PREFIX/lib` and `$CMAKE_INSTALL_PREFIX/include/cugraph` respectively.



#### Building and installing the Python package

5. Install the Python package to your Python path:

```bash
cd $CUGRAPH_HOME
cd python
python setup.py install    # install cugraph python bindings
```





#### Run tests

6. Run either the standalone tests or the Python tests with datasets
  - **C++ stand alone tests** 

    From the build directory : 

    ```bash
    # Run the cugraph tests
    cd $CUGRAPH_HOME
    cd cpp/build
    gtests/GDFGRAPH_TEST		# this is an executable file
    ```

  - **Python tests with datasets** 

    ```bash
    cd $CUGRAPH_HOME
    cd python
    pytest  
    ```


Note: This conda installation only applies to Linux and Python versions 3.6/3.7.



## Documentation

Python API documentation can be generated from [docs](docs) directory.



## C++ ABI issues

cuGraph builds with C++14 features.  By default, we build cuGraph with the latest ABI (the ABI changed with C++11).  The version of cuDF pointed to in the conda installation above is build with the new ABI.

If you see link errors indicating trouble finding functions that use C++ strings when trying to build cuGraph you may have an ABI incompatibility.

There are a couple of complications that may make this a problem:
* if you need to link in a library built with the old ABI, you may need to build the entire tool chain from source using the old ABI.
* if you build cudf from source (for whatever reason), the default behavior for cudf (at least through version 0.5.x) is to build using the old ABI.  You can build with the new ABI, but you need to follow the instructions in CUDF to explicitly turn that on.

If you must build cugraph with the old ABI, you can use the following command (instead of the cmake call above):

```bash
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CXX11_ABI=OFF
```



### (OPTIONAL) Set environment variable on activation

It is possible to configure the conda environment to set environmental variables on activation. Providing instructions to set PATH to include the CUDA toolkit bin directory and LD_LIBRARY_PATH to include the CUDA lib64 directory will be helpful.



```bash
cd  ~/anaconda3/envs/cugraph_dev

mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
```



Next the env_vars.sh file needs to be edited

```bash
vi ./etc/conda/activate.d/env_vars.sh

#!/bin/bash
export PATH=/usr/local/cuda-10.0/bin:$PATH # or cuda-9.2 if using CUDA 9.2
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH # or cuda-9.2 if using CUDA 9.2
```



```
vi ./etc/conda/deactivate.d/env_vars.sh

#!/bin/bash
unset PATH
unset LD_LIBRARY_PATH
```



## nvGraph

The nvGraph library is now open source and part of cuGraph. It can be build as a stand alone by following nvgraph's [readme](cpp/nvgraph/). 


------



## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aim to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<p align="center"><img src="img/rapids_arrow.png" width="80%"/></p>

### Apache Arrow on GPU

The GPU version of [Apache Arrow](https://arrow.apache.org/) is a common API that enables efficient interchange of tabular data between processes running on the GPU. End-to-end computation on the GPU avoids unnecessary copying and converting of data off the GPU, reducing compute time and cost for high-performance analytics common in artificial intelligence workloads. As the name implies, cuDF uses the Apache Arrow columnar data format on the GPU. Currently, a subset of the features in Apache Arrow are supported.