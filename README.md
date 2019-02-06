# cuGraph: GPU accelerated graph analytics

cuGraph is a library implementing Graph Analytics functionalities based on GPU Data Frames. For more project details, see [rapids.ai](https://rapids.ai/).



## Install cuGraph

### Conda

It is easy to install cuGraph using conda. You can get a minimal conda installation with [Miniconda](https://conda.io/miniconda.html) or get the full installation with [Anaconda](https://www.anaconda.com/download).

Install and update cuGraph using the conda command:

```bash
# NOT YET WORKING #

# CUDA 9.2
conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cugraph

# CUDA 10.0
conda install -c nvidia/label/cuda10.0 -c rapidsai/label/cuda10.0 -c numba -c conda-forge -c defaults cugraph
```

Note: This conda installation only applies to Linux and Python versions 3.6/3.7.

### Pip

It is easy to install cuDF using pip. You must specify the CUDA version to ensure you install the right package.

```bash
# NOT YET WORKING #


# CUDA 9.2
pip install cugraph-cuda92

# CUDA 10.0.
pip install cugraph-cuda100
```






## Development Setup

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



## Install cuGraph

### Conda

You can get a minimal conda installation with [Miniconda](https://conda.io/miniconda.html) or get the full installation with [Anaconda](https://www.anaconda.com/download).

Note: This conda installation only applies to Linux and Python versions 3.5/3.6.

You can create and activate a development environment using the conda commands:

```bash
# create the conda environment (assuming in base `cugraph` directory)
conda env create --name cugraph_dev --file conda/environments/cugraph_dev.yml
# activate the environment
source activate 
```

The environment can be updated as development includes/changes the depedencies. To do so, run:

```bash
conda env update --name cugraph_dev --file conda/environments/cugraph_dev.yml
source activate 
```

This installs the required `cmake`, `cudf`, `pyarrow` and other
dependencies into the `cugraph_dev` conda environment and activates it.
### Configure and build

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

Install the Python package to your Python path:

```bash
python setup.py install    # install cudf python bindings
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
