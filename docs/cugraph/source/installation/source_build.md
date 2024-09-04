# Building from Source

These instructions are tested on supported versions/distributions of Linux,
CUDA, and Python - See [RAPIDS Getting Started](https://rapids.ai/start.html)
for the list of supported environments.  Other environments _might be_
compatible, but are not currently tested.

## Prerequisites

__Compilers:__
* `gcc`           version 9.3+
* `nvcc`          version 11.5+

__CUDA:__
* CUDA 11.8+
* NVIDIA GPU, Volta architecture or later, with [compute capability](https://developer.nvidia.com/cuda-gpus) 7.0+

Further details and download links for these prerequisites are available on the
[RAPIDS System Requirements page](https://docs.rapids.ai/install#system-req).

## Setting up the development environment

### Clone the repository:
```bash
CUGRAPH_HOME=$(pwd)/cugraph
git clone https://github.com/rapidsai/cugraph.git $CUGRAPH_HOME
cd $CUGRAPH_HOME
```

### Create the conda environment

Using conda is the easiest way to install both the build and runtime
dependencies for cugraph. While it is possible to build and run cugraph without
conda, the required packages occasionally change, making it difficult to
document here. The best way to see the current dependencies needed for a build
and run environment is to examine the list of packages in the [conda
environment YAML
files](https://github.com/rapidsai/cugraph/blob/main/conda/environments).

```bash
# for CUDA 11.x
conda env create --name cugraph_dev --file $CUGRAPH_HOME/conda/environments/all_cuda-118_arch-x86_64.yaml

# for CUDA 12.x
conda env create --name cugraph_dev --file $CUGRAPH_HOME/conda/environments/all_cuda-125_arch-x86_64.yaml


# activate the environment
conda activate cugraph_dev

# to deactivate an environment
conda deactivate
```

The environment can be updated as cugraph adds/removes/updates its dependencies. To do so, run:

```bash
# for CUDA 11.x
conda env update --name cugraph_dev --file $CUGRAPH_HOME/conda/environments/all_cuda-118_arch-x86_64.yaml
conda activate cugraph_dev

# for CUDA 12.x
conda env update --name cugraph_dev --file $CUGRAPH_HOME/conda/environments/all_cuda-125_arch-x86_64.yaml
conda activate cugraph_dev



```

### Build and Install

#### Build and install using `build.sh`
Using the `build.sh` script, located in the `$CUGRAPH_HOME` directory, is the
recommended way to build and install the cugraph libraries. By default,
`build.sh` will build and install a predefined set of targets
(packages/libraries), but can also accept a list of targets to build.

For example, to build only the cugraph C++ library (`libcugraph`) and the
high-level python library (`cugraph`) without building the C++ test binaries,
run this command:

```bash
$ cd $CUGRAPH_HOME
$ ./build.sh libcugraph pylibcugraph cugraph --skip_cpp_tests
```

There are several other options available on the build script for advanced
users. Refer to the output of `--help` for details.

Note that libraries will be installed to the location set in `$PREFIX` if set
(i.e. `export PREFIX=/install/path`), otherwise to `$CONDA_PREFIX`.

#### Updating the RAFT branch

`libcugraph` uses the [RAFT](https://github.com/rapidsai/raft) library and
there are times when it might be desirable to build against a different RAFT
branch, such as when working on new features that might span both RAFT and
cuGraph.

For local development, the `CPM_raft_SOURCE=<path/to/raft/source>` option can
be passed to the `cmake` command to enable `libcugraph` to use the local RAFT
branch. The `build.sh` script calls `cmake` to build the C/C++ targets, but
developers can call `cmake` directly in order to pass it options like those
described here. Refer to the `build.sh` script to see how to call `cmake` and
other commands directly.

To have CI test a `cugraph` pull request against a different RAFT branch,
modify the bottom of the `cpp/cmake/thirdparty/get_raft.cmake` file as follows:

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

When the above change is pushed to a pull request, the continuous integration
servers will use the specified RAFT branch to run the cuGraph tests. After the
changes in the RAFT branch are merged to the release branch, remember to revert
the `get_raft.cmake` file back to the original cuGraph branch.


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

Note: This conda installation only applies to Linux and Python versions 3.10, 3.11, and 3.12.

### (OPTIONAL) Set environment variable on activation

It is possible to configure the conda environment to set environment variables
on activation. Providing instructions to set PATH to include the CUDA toolkit
bin directory and LD_LIBRARY_PATH to include the CUDA lib64 directory will be
helpful.

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

Python API documentation can be generated from _./docs/cugraph directory_. Or
through using "./build.sh docs"

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
