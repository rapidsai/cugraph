# libcugraph C++ tests

## Prerequisites
### Datasets
```
/path/to/cuGraph> ./datasets/get_test_data.sh
/path/to/cuGraph> export RAPIDS_DATASET_ROOT_DIR=/path/to/cuGraph/datasets
```
### System Requirements
* MPI (multi-GPU tests only)
   ```
   conda install -c conda-forge openmpi
   ```

## Building
```
/path/to/cuGraph> ./build.sh libcugraph
```
To build the multi-GPU tests:
```
/path/to/cuGraph> ./build.sh libcugraph cpp-mgtests
```

## Running
```
<example here>
```
To run the multi-GPU tests (example using 2 GPUs):
```
/path/to/cuGraph> mpirun -n 2 ./cpp/build/gtests/MG_PAGERANK_TEST
```
