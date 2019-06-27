# Contributing to cuGraph

If you are interested in contributing to cuGraph, your contributions will fall
into three categories:
1. You want to report a bug, feature request, or documentation issue
    - File an [issue](https://github.com/rapidsai/cugraph/issues/new/choose)
    describing what you encountered or what you want to see changed.
    - The RAPIDS team will evaluate the issues and triage them, scheduling
    them for a release. If you believe the issue needs priority attention
    comment on the issue to notify the team.
2. You want to propose a new Feature and implement it
    - Post about your intended feature, and we shall discuss the design and
    implementation.
    - Once we agree that the plan looks good, go ahead and implement it, using
    the [code contributions](#code-contributions) guide below.
3. You want to implement a feature or bug-fix for an outstanding issue
    - Follow the [code contributions](#code-contributions) guide below.
    - If you need more context on a particular issue, please ask and we shall
    provide.

## Code contributions

### Your first issue

1. Read the project's [README.md](https://github.com/rapidsai/cugraph/blob/master/README.md)
    to learn how to setup the development environment
2. Find an issue to work on. The best way is to look for the [good first issue](https://github.com/rapidsai/cugraph/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
    or [help wanted](https://github.com/rapidsai/cugraph/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels
3. Comment on the issue saying you are going to work on it
4. Fork the cuGraph repo and Code! Make sure to update unit tests!
5. When done, [create your pull request](https://github.com/rapidsai/cugraph/compare)
6. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/). Fix if needed
7. Wait for other developers to review your code and update code as needed
8. Once reviewed and approved, a RAPIDS developer will merge your pull request

Remember, if you are unsure about anything, don't hesitate to comment on issues
and ask for clarifications!

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues of our next release in our [project boards](https://github.com/rapidsai/cugraph/projects).

> **Pro Tip:** Always look at the release board with the highest number for
issues to work on. This is where RAPIDS developers also focus their efforts.

Look at the unassigned issues, and find an issue you are comfortable with
contributing to. Start with _Step 3_ from above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.

## Setting Up Your Build Environment

### Build from Source

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
  cd cpp                                        # enter cpp directory
  mkdir build                                   # create build directory
  cd build                                      # enter the build directory
  cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX

  # now build the code
  make -j                                       # "-j" starts multiple threads
  make install                                  # install the libraries
  ```
The default installation locations are `$CMAKE_INSTALL_PREFIX/lib` and `$CMAKE_INSTALL_PREFIX/include/cugraph` respectively.

As a convenience, a `build.sh` script is provided in `$CUGRAPH_HOME`. To execute the same build commands above, run the script as shown below.  Note that the libraries will be installed to the location set in `$PREFIX` if set (i.e. `export PREFIX=/install/path`), otherwise to `$CONDA_PREFIX`.
```bash
$ cd $CUGRAPH_HOME
$ ./build.sh libcugraph  # build the cuGraph libraries and install them to
                         # $PREFIX if set, otherwise $CONDA_PREFIX
```

#### Building and installing the Python package

5. Install the Python package to your Python path:

```bash
cd $CUGRAPH_HOME
cd python
python setup.py install    # install cugraph python bindings
```

Like the `libcugraph` build step above, `build.sh` can also be used to build the `cugraph` python package, as shown below:
```bash
$ cd $CUGRAPH_HOME
$ ./build.sh cugraph  # build the cuGraph python bindings and install them
                      # to $PREFIX if set, otherwise $CONDA_PREFIX
```

Note: other `build.sh` options include:
```bash
$ cd $CUGRAPH_HOME
$ ./build.sh clean                        # remove any prior build artifacts and configuration (start over)
$ ./build.sh libcugraph -v                # compile and install libcugraph with verbose output
$ ./build.sh libcugraph -g                # compile and install libcugraph for debug
$ PARALLEL_LEVEL=4 ./build.sh libcugraph  # compile and install libcugraph limiting parallel build jobs to 4 (make -j4)
$ ./build.sh libcugraph -n                # compile libcugraph but do not install
```

#### Run tests

6. Run either the C++ or the Python tests with datasets

  - **Python tests with datasets**

    ```bash
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

   If you already have the datasets:

   ```bash
   export RAPIDS_DATASET_ROOT_DIR=<path_to_ccp_test_and_reference_data>
   ```
   If you do not have the datasets:

   ```bash
   cd $CUGRAPH_HOME/datasets
   source get_test_data.sh #This takes about 10 minutes and download 1GB data (>5 GB uncompressed)
   ```

   Run the C++ tests on large input:

   ```bash
   cd $CUGRAPH_HOME/cpp/build
   #test one particular analytics (eg. pagerank)
   gtests/PAGERANK_TEST
   #test everything
   make test
   ```

Note: This conda installation only applies to Linux and Python versions 3.6/3.7.

### Building and Testing on a gpuCI image locally

Before submitting a pull request, you can do a local build and test on your machine that mimics our gpuCI environment using the `ci/local/build.sh` script.
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
export PATH=/usr/local/cuda-10.0/bin:$PATH # or cuda-9.2 if using CUDA 9.2
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH # or cuda-9.2 if using CUDA 9.2
```

```
vi ./etc/conda/deactivate.d/env_vars.sh

#!/bin/bash
unset PATH
unset LD_LIBRARY_PATH
```

## Creating documentation

Python API documentation can be generated from [docs](docs) directory.

## C++ ABI issues

cuGraph builds with C++14 features.  By default, we build cuGraph with the latest ABI (the ABI changed with C++11).  The version of cuDF pointed to in
the conda installation above is build with the new ABI.

If you see link errors indicating trouble finding functions that use C++ strings when trying to build cuGraph you may have an ABI incompatibility.

There are a couple of complications that may make this a problem:
* if you need to link in a library built with the old ABI, you may need to build the entire tool chain from source using the old ABI.
* if you build cudf from source (for whatever reason), the default behavior for cudf (at least through version 0.5.x) is to build using the old ABI.  You can build with the new ABI, but you need to follow the instructions in CUDF to explicitly turn that on.

If you must build cugraph with the old ABI, you can use the following command (instead of the cmake call above):

```bash
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CXX11_ABI=OFF
```

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
