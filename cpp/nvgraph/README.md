# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;nvgraph - NVIDIA graph library</div>

## Development Setup

The following instructions are for developers and contributors to nvgraph OSS development.

### Get libcudf Dependencies

Compiler requirements:

* `gcc`     version 5.4
* `nvcc`    version 9.2
* `cmake`   version 3.12

CUDA/GPU requirements:

* CUDA 9.2+
* NVIDIA driver 396.44+
* Pascal architecture or better

You can obtain CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

### Build from Source

To install nvgraph from source, ensure the dependencies are met and follow the steps below:

- Clone the repository and submodules
```bash
git clone --recurse-submodules https://gitlab-master.nvidia.com/RAPIDS/nvgraph.git
cd nvgraph
```

- Build and install `libcudf`. CMake depends on the `nvcc` executable being on your path or defined in `$CUDACXX`.
```bash
$ cd cpp                                            # navigate to C/C++ CUDA source root directory
$ mkdir build                                       # make a build directory
$ cd build                                          # enter the build directory
$ cmake .. -DCMAKE_INSTALL_PREFIX=/install/path     # configure cmake
$ make -j                                           # compile the libraries ... '-j' will start a parallel job using the number of physical cores available on your system
$ make install                                      # install the libraries to '/install/path'
```

- To run tests (Optional):
```bash
$ make test
```
