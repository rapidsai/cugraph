# <div align="left"><img src="../../img/rapids_logo.png" width="90px"/>&nbsp;nvgraph - NVIDIA graph library</div>

Data analytics is a growing application of high-performance computing. Many advanced data analytics problems can be couched as graph problems. In turn, many of the common graph problems today can be couched as sparse linear algebra. This is the motivation for nvGRAPH, which harnesses the power of GPUs for linear algebra to handle large graph analytics and big data analytics problems.

## Development Setup

### Conda{#conda}

It is easy to install nvGraph using conda. You can get a minimal conda installation with [Miniconda](https://conda.io/miniconda.html) or get the full installation with [Anaconda](https://www.anaconda.com/download).

Install and update nvGraph using the conda command:

```bash
# CUDA 9.2
conda install -c nvidia nvgraph

# CUDA 10.0
conda install -c nvidia/label/cuda10.0 nvgraph 

```

Note: This conda installation only applies to Linux and Python versions 3.6/3.7.

### Build from Source {#source}

The following instructions are for developers and contributors to nvGraph OSS development. These instructions are tested on Linux Ubuntu 16.04 & 18.04. Use these instructions to build nvGraph from source and contribute to its development.  Other operating systems may be compatible, but are not currently tested.

The nvGraph package is a C/C++ CUDA library. It needs to be installed in order for nvGraph to operate correctly.  

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
Compiler requirements:


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

2) Build and install `libnvgraph_rapids.so`. CMake depends on the `nvcc` executable being on your path or defined in `$CUDACXX`.

  This project uses cmake for building the C/C++ library. To configure cmake, run:

  ```bash
  cd $CUGRAPH_HOME
  cd cpp/nvgraph/cpp	# enter nvgraph's cpp directory
  mkdir build   		# create build directory 
  cd build     		# enter the build directory
  cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX 

  # now build the code
  make -j				# "-j" starts multiple threads
  make install		# install the libraries 
  ```

The default installation  locations are `$CMAKE_INSTALL_PREFIX/lib` and `$CMAKE_INSTALL_PREFIX/include/nvgraph` respectively.

#### C++ stand alone tests

```bash
# Run the cugraph tests
cd $CUGRAPH_HOME
cd cpp/nvgraph/cpp/build
gtests/NVGRAPH_TEST # this is an executable file
```
Other test executables require specific datasets and will result in failure if they are not present.
## Documentation

The C API documentation can be found in the [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/nvgraph/index.html).



