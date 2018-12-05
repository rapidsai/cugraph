# cuGraph: GPU accelerated graph analytics

cuGraph is a library implementing Graph Analytics functionalities based on GPU Data Frames. For more project details, see [rapids.ai](https://rapids.ai/).

## Development Setup

The following instructions are tested on Linux systems.

Compiler requirement:

* `g++` 4.8 or 5.4
* `cmake` 3.12+

CUDA requirement:

* CUDA 9.0+

You can obtain CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

### Conda

You can get a minimal conda installation with [Miniconda](https://conda.io/miniconda.html) or get the full installation with [Anaconda](https://www.anaconda.com/download).

Note: This conda installation only applies to Linux and Python versions 3.5/3.6.

You can create and activate a development environment using the conda commands:

```bash
# create the conda environment (assuming in base `cugraph` directory)
conda env create --name cugraph_dev --file conda/environments/dev_py35.yml
# activate the environment
source activate 
```

The environment can be updated as development includes/changes the depedencies. To do so, run:

```bash
conda env update --name cugraph_dev --file conda/environments/dev_py35.yml
source activate 
```

This installs the required `cmake`, `cudf`, `pyarrow` and other
dependencies into the `cugraph_dev` conda environment and activates it.
### Configure and build

This project uses cmake for building the C/C++ library. To configure cmake,
run:

```bash
mkdir build   # create build directory for out-of-source build
cd build      # enter the build directory
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DNVG_PLUGIN=FALSE  # configure cmake ... use $CONDA_PREFIX if you're using Anaconda
```

Add `-DNVG_PLUGIN=TRUE` to configure cmake to build nvGraph plugin for cuGraph.

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
