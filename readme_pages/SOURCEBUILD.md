# Building from Source

The following instructions are for users wishing to build cuGraph from source code.  These instructions are tested on supported distributions of Linux, CUDA, and Python - See [RAPIDS Getting Started](https://rapids.ai/start.html) for list of supported environments.  Other operating systems _might be_ compatible, but are not currently tested.

The cuGraph package include both a C/C++ CUDA portion and a python portion.  Both libraries need to be installed in order for cuGraph to operate correctly.

## Prerequisites

__Compiler__:
* `gcc`         version 9.3+
* `nvcc`        version 11.0+
* `cmake`       version 3.20.1+

__CUDA:__
* CUDA 11.0+
* NVIDIA driver 450.80.02+
* Pascal architecture or better


You can obtain CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).



## Building cuGraph
To install cuGraph from source, ensure the dependencies are met.


### Clone Repo and Configure Conda Environment
__GIT clone a version of the repository__

  ```bash
  # Set the localtion to cuGraph in an environment variable CUGRAPH_HOME
  export CUGRAPH_HOME=$(pwd)/cugraph

  # Download the cuGraph repo - if you have a folked version, use that path here instead
  git clone https://github.com/rapidsai/cugraph.git $CUGRAPH_HOME

  cd $CUGRAPH_HOME
  ```

__Create the conda development environment__

```bash
# create the conda environment (assuming in base `cugraph` directory)

# for CUDA 11.5
conda env create --name cugraph_dev --file conda/environments/all_cuda-118_arch-x86_64.yaml

# activate the environment
conda activate cugraph_dev

# to deactivate an environment
conda deactivate
```

  - The environment can be updated as development includes/changes the dependencies. To do so, run:


```bash

# Where XXX is the CUDA 11 version
conda env update --name cugraph_dev --file conda/environments/cugraph_dev_cuda11.XXX.yml

conda activate cugraph_dev
```


### Build and Install Using the `build.sh` Script
Using the `build.sh` script make compiling and installing cuGraph a breeze.  To build and install, simply do:

```bash
$ cd $CUGRAPH_HOME
$ ./build.sh clean
$ ./build.sh libcugraph
$ ./build.sh cugraph
```

There are several other options available on the build script for advanced users.
`build.sh` options:
```bash
build.sh [<target> ...] [<flag> ...]
 where <target> is:
   clean                      - remove all existing build artifacts and configuration (start over)
   uninstall                  - uninstall libcugraph and cugraph from a prior build/install (see also -n)
   libcugraph                 - build libcugraph.so and SG test binaries
   libcugraph_etl             - build libcugraph_etl.so and SG test binaries
   pylibcugraph               - build the pylibcugraph Python package
   cugraph                    - build the cugraph Python package
   cugraph-service            - build the cugraph-service_client and cugraph-service_server Python package
   cpp-mgtests                - build libcugraph and libcugraph_etl MG tests. Builds MPI communicator, adding MPI as a dependency.
   cugraph-dgl                - build the cugraph-dgl extensions for DGL
   docs                       - build the docs
 and <flag> is:
   -v                         - verbose build mode
   -g                         - build for debug
   -n                         - do not install after a successful build
   --pydevelop                - use setup.py develop instead of install
   --allgpuarch               - build for all supported GPU architectures
   --skip_cpp_tests           - do not build the SG test binaries as part of the libcugraph and libcugraph_etl targets
   --without_cugraphops       - do not build algos that require cugraph-ops
   --cmake_default_generator  - use the default cmake generator instead of ninja
   --clean                    - clean an individual target (note: to do a complete rebuild, use the clean target described above)
   -h                         - print this text

 default action (no args) is to build and install 'libcugraph' then 'libcugraph_etl' then 'pylibcugraph' then 'cugraph' then 'cugraph-service' targets

examples:
$ ./build.sh clean                        # remove prior build artifacts (start over)
$ ./build.sh libcugraph -v                # compile and install libcugraph with verbose output
$ ./build.sh libcugraph -g                # compile and install libcugraph for debug
$ ./build.sh libcugraph -n                # compile libcugraph but do not install

# make parallelism options can also be defined: Example build jobs using 4 threads (make -j4)
$ PARALLEL_LEVEL=4 ./build.sh libcugraph

Note that the libraries will be installed to the location set in `$PREFIX` if set (i.e. `export PREFIX=/install/path`), otherwise to `$CONDA_PREFIX`.
```


## Building each section independently
#### Build and Install the C++/CUDA `libcugraph` Library
CMake depends on the `nvcc` executable being on your path or defined in `$CUDACXX`.

This project uses cmake for building the C/C++ library. To configure cmake, run:

  ```bash
  # Set the localtion to cuGraph in an environment variable CUGRAPH_HOME
  export CUGRAPH_HOME=$(pwd)/cugraph

  cd $CUGRAPH_HOME
  cd cpp                                        # enter cpp directory
  mkdir build                                   # create build directory
  cd build                                      # enter the build directory
  cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX

  # now build the code
  make -j                                       # "-j" starts multiple threads
  make install                                  # install the libraries
  ```
The default installation locations are `$CMAKE_INSTALL_PREFIX/lib` and `$CMAKE_INSTALL_PREFIX/include/cugraph` respectively.

#### Updating the RAFT branch

`libcugraph` uses the [RAFT](https://github.com/rapidsai/raft) library and there are times when it might be desirable to build against a different RAFT branch, such as when working on new features that might span both RAFT and cuGraph.

For local development, the `CPM_raft_SOURCE=<path/to/raft/source>` option can be passed to the `cmake` command to enable `libcugraph` to use the local RAFT branch.

To have CI test a `cugraph` pull request against a different RAFT branch, modify the bottom of the `cpp/cmake/thirdparty/get_raft.cmake` file as follows:

```cmake
# Change pinned tag and fork here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# RPM_raft_SOURCE=/path/to/local/raft
find_and_configure_raft(VERSION    ${CUGRAPH_MIN_VERSION_raft}
                        FORK       <your_git_fork>
                        PINNED_TAG <your_git_branch_or_tag>

                        # When PINNED_TAG above doesn't match cugraph,
                        # force local raft clone in build directory
                        # even if it's already installed.
                        CLONE_ON_PIN     ON
                        )
```

When the above change is pushed to a pull request, the continuous integration servers will use the specified RAFT branch to run the cuGraph tests. After the changes in the RAFT branch are merged to the release branch, remember to revert the `get_raft.cmake` file back to the original cuGraph branch.

### Building and installing the Python package

2) Install the Python packages to your Python path:

```bash
cd $CUGRAPH_HOME
cd python
cd pylibcugraph
python setup.py build_ext --inplace
python setup.py install    # install pylibcugraph
cd ../cugraph
python setup.py build_ext --inplace
python setup.py install    # install cugraph python bindings

```



## Run tests

If you already have the datasets:

   ```bash
   export RAPIDS_DATASET_ROOT_DIR=<path_to_ccp_test_and_reference_data>
   ```
   If you do not have the datasets:

   ```bash
   cd $CUGRAPH_HOME/datasets
   source get_test_data.sh #This takes about 10 minutes and downloads 1GB data (>5 GB uncompressed)
   ```

Run either the C++ or the Python tests with datasets

  - **Python tests with datasets**


    ```bash
    pip install python-louvain #some tests require this package to run
    cd $CUGRAPH_HOME
    cd python
    pytest
    ```
  - **C++ stand alone tests**

    From the build directory :

    ```bash
    # Run the cugraph tests
    cd $CUGRAPH_HOME
    cd cpp/build
    gtests/GDFGRAPH_TEST		# this is an executable file
    ```
 - **C++ tests with larger datasets**



   Run the C++ tests on large input:

   ```bash
   cd $CUGRAPH_HOME/cpp/build
   #test one particular analytics (eg. pagerank)
   gtests/PAGERANK_TEST
   #test everything
   make test
   ```

Note: This conda installation only applies to Linux and Python versions 3.8/3.10.

### Building and Testing on a gpuCI image locally

You can do a local build and test on your machine that mimics our gpuCI environment using the `ci/local/build.sh` script.
For detailed information on usage of this script, see [here](ci/local/README.md).

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
export PATH=/usr/local/cuda-11.0/bin:$PATH # or cuda-11.1 if using CUDA 11.1 and cuda-11.2 if using CUDA 11.2, respectively
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH # or cuda-11.1 if using CUDA 11.1 and cuda-11.2 if using CUDA 11.2, respectively
```

```
vi ./etc/conda/deactivate.d/env_vars.sh

#!/bin/bash
unset PATH
unset LD_LIBRARY_PATH
```

## Creating documentation

Python API documentation can be generated from [docs](docs) directory.

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
