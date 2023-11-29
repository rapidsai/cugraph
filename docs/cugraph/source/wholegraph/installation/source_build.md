# Building from Source

The following instructions are for users wishing to build wholegraph from source code. These instructions are tested on supported distributions of Linux,CUDA,
and Python - See [RAPIDS Getting Started](https://rapids.ai/start.html) for a list of supported environments.
Other operating systems _might be_ compatible, but are not currently tested.

The wholegraph package includes both a C/C++ CUDA portion and a python portion. Both libraries need to be installed in order for cuGraph to operate correctly.
The C/C++ CUDA library is `libwholegraph` and the python library is `pylibwholegraph`.

## Prerequisites

__Compiler__:
* `gcc`         version 11.0+
* `nvcc`        version 11.0+
* `cmake`       version 3.26.4+

__CUDA__:
* CUDA 11.8+
* NVIDIA driver 450.80.02+
* Pascal architecture or better

You can obtain CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

__Other Packages__:
* ninja
* nccl
* cython
* setuputils3
* scikit-learn
* scikit-build
* nanobind>=0.2.0

## Building wholegraph
To install wholegraph from source, ensure the dependencies are met.

### Clone Repo and Configure Conda Environment
__GIT clone a version of the repository__

  ```bash
  # Set the location to wholegraph in an environment variable WHOLEGRAPH_HOME
  export WHOLEGRAPH_HOME=$(pwd)/wholegraph

  # Download the wholegraph repo - if you have a forked version, use that path here instead
  git clone https://github.com/rapidsai/wholegraph.git $WHOLEGRAPH_HOME

  cd $WHOLEGRAPH_HOME
  ```

__Create the conda development environment__

```bash
# create the conda environment (assuming in base `wholegraph` directory)

# for CUDA 11.x
conda env create --name wholegraph_dev --file conda/environments/all_cuda-118_arch-x86_64.yaml

# activate the environment
conda activate wholegraph_dev

# to deactivate an environment
conda deactivate
```

  - The environment can be updated as development includes/changes the dependencies. To do so, run:


```bash

# Where XXX is the CUDA version
conda env update --name wholegraph_dev --file conda/environments/all_cuda-XXX_arch-x86_64.yaml

conda activate wholegraph_dev
```


### Build and Install Using the `build.sh` Script
Using the `build.sh` script make compiling and installing wholegraph a
breeze. To build and install, simply do:

```bash
$ cd $WHOLEGRAPH_HOME
$ ./build.sh clean
$ ./build.sh libwholegraph
$ ./build.sh pylibwholegraph
```

There are several other options available on the build script for advanced users.
`build.sh` options:
```bash
build.sh [<target> ...] [<flag> ...]
 where <target> is:
   clean                    - remove all existing build artifacts and configuration (start over).
   uninstall                - uninstall libwholegraph and pylibwholegraph from a prior build/install (see also -n)
   libwholegraph            - build the libwholegraph C++ library.
   pylibwholegraph          - build the pylibwholegraph Python package.
   tests                    - build the C++ (OPG) tests.
   benchmarks               - build benchmarks.
   docs                     - build the docs
 and <flag> is:
   -v                          - verbose build mode
   -g                          - build for debug
   -n                          - no install step
   --allgpuarch               - build for all supported GPU architectures
   --cmake-args=\\\"<args>\\\" - add arbitrary CMake arguments to any cmake call
   --compile-cmd               - only output compile commands (invoke CMake without build)
   --clean                    - clean an individual target (note: to do a complete rebuild, use the clean target described above)
   -h | --h[elp]               - print this text

 default action (no args) is to build and install 'libwholegraph' then 'pylibwholegraph' targets

examples:
$ ./build.sh clean                        # remove prior build artifacts (start over)
$ ./build.sh

# make parallelism options can also be defined: Example build jobs using 4 threads (make -j4)
$ PARALLEL_LEVEL=4 ./build.sh libwholegraph

Note that the libraries will be installed to the location set in `$PREFIX` if set (i.e. `export PREFIX=/install/path`), otherwise to `$CONDA_PREFIX`.
```


## Building each section independently
### Build and Install the C++/CUDA `libwholegraph` Library
CMake depends on the `nvcc` executable being on your path or defined in `$CUDACXX`.

This project uses cmake for building the C/C++ library. To configure cmake, run:

  ```bash
  # Set the location to wholegraph in an environment variable WHOLEGRAPH_HOME
  export WHOLEGRAPH_HOME=$(pwd)/wholegraph

  cd $WHOLEGRAPH_HOME
  cd cpp                                        # enter cpp directory
  mkdir build                                   # create build directory
  cd build                                      # enter the build directory
  cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX

  # now build the code
  make -j                                       # "-j" starts multiple threads
  make install                                  # install the libraries
  ```
The default installation locations are `$CMAKE_INSTALL_PREFIX/lib` and `$CMAKE_INSTALL_PREFIX/include/wholegraph` respectively.

### Building and installing the Python package

Build and Install the Python packages to your Python path:

```bash
cd $WHOLEGRAPH_HOME
cd python
cd pylibwholegraph
python setup.py build_ext --inplace
python setup.py install    # install pylibwholegraph
```

## Run tests

Run either the C++ or the Python tests with datasets

  - **Python tests with datasets**

    ```bash
    cd $WHOLEGRAPH_HOME
    cd python
    pytest
    ```

  - **C++ stand alone tests**

    From the build directory :

    ```bash
    # Run the tests
    cd $WHOLEGRAPH_HOME
    cd cpp/build
    gtests/PARALLEL_UTILS_TESTS		# this is an executable file
    ```


Note: This conda installation only applies to Linux and Python versions 3.8/3.10.

## Creating documentation

Python API documentation can be generated from _./docs/wholegraph directory_. Or through using "./build.sh docs"

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
