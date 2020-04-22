# Building from Source

The following instructions are for users wishing to build cuGraph from source code.  These instructions are tested on supported distributions of Linux, CUDA, and Python - See [RAPIDS Getting Started](https://rapids.ai/start.html) for list of supported environments.  Other operating systems _might be_ compatible, but are not currently tested. 

The cuGraph package include both a C/C++ CUDA portion and a python portion.  Both libraries need to be installed in order for cuGraph to operate correctly.

### Prerequisites

Compiler requirement:

* `gcc`     version 5.4+
* `nvcc`    version 10.0+
* `cmake`   version 3.12

CUDA requirement:

* CUDA 10.0+
* NVIDIA driver 396.44+
* Pascal architecture or better

You can obtain CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

### Build and Install the C/C++ CUDA components

To install cuGraph from source, ensure the dependencies are met and follow the steps below:

1) A cloned version of the repository

  ```bash
  # Set the localtion to cuGraph in an environment variable CUGRAPH_HOME
  export CUGRAPH_HOME=$(pwd)/cugraph

  # Download the cuGraph repo - if you have a folked version, use that path here instead
  git clone https://github.com/rapidsai/cugraph.git $CUGRAPH_HOME

  cd $CUGRAPH_HOME
  ```

2) Create the conda development environment

```bash
# create the conda environment (assuming in base `cugraph` directory)

# for CUDA 10
conda env create --name cugraph_dev --file conda/environments/cugraph_dev_cuda10.0.yml

# for CUDA 10.1
conda env create --name cugraph_dev --file conda/environments/cugraph_dev_cuda10.1.yml

# for CUDA 10.2
conda env create --name cugraph_dev --file conda/environments/cugraph_dev_cuda10.2.yml

# activate the environment
conda activate cugraph_dev

# to deactivate an environment
conda deactivate
```

  - The environment can be updated as development includes/changes the dependencies. To do so, run:


```bash

# for CUDA 10
conda env update --name cugraph_dev --file conda/environments/cugraph_dev_cuda10.0.yml

# for CUDA 10.1
conda env update --name cugraph_dev --file conda/environments/cugraph_dev_cuda10.1.yml

# for CUDA 10.2
conda env update --name cugraph_dev --file conda/environments/cugraph_dev_cuda10.2.yml

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

### Building and installing the Python package

5. Install the Python package to your Python path:

```bash
cd $CUGRAPH_HOME
cd python
python setup.py build_ext --inplace
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

### Run tests

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
export PATH=/usr/local/cuda-10.0/bin:$PATH # or cuda-10.2 if using CUDA 10.2
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH # or cuda-10.2 if using CUDA 10.2
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