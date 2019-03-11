### Build dask-cugraph
- Get the latest version of `dask-cugraph` branch.

```bash
# Set the localtion to cuGraph in an environment variable CUGRAPH_HOME 
export CUGRAPH_HOME=$(pwd)/cugraph

# Download the cuGraph repo
git clone -b dask-cugraph https://github.com/rapidsai/cugraph.git $CUGRAPH_HOME

# Next load all the submodules
cd $CUGRAPH_HOME
git submodule update --init --remote --recursive
```

- Update or create `cugraph_dev` following the regular [README](README.md).
- Clean **all** previous build files
- Build cuGraph (C++ and Python) following the regular [README](README.md).

### Install dask-cugraph
- Go to `cugraph/dask` and run :
```bash
python setup.py install
```

### Run dask-cugraph test
Open 3 different terminals in `cugraph/dask` and run :
```bash
# dask scheduler in terminal 1 
dask-scheduler --scheduler-file cluster.json
```
```bash
# dask-mpi in terminal 2
mpirun -np 2 dask-mpi --no-nanny --nthreads 2 --no-scheduler --scheduler-file cluster.json
```
```bash
# dask-cugraph smoke test in terminal 3
python test_dask_cugraph.py
```

### Run multi-GPU Pagerank on large files using the C++ API
```bash
mpirun -np <NUM_PARTS> ./gtests/MULTI_PAGERANK_FILE_TEST path_to_part0
```
