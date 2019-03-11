### Build cugraph
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

- Update the `cugraph_dev` conda env
```bash
# for CUDA 9.2
conda env update --name cugraph_dev --file conda/environments/cugraph_dev.yml

conda activate cugraph_dev 
```

- Clean **all** previous build files
- Build cuGraph (C++ and Python) following the regular [README](README.md).

### Install dask-cugraph
- Go to `cugraph/dask` and run :
```bash
python setup.py install
```

### Run dask-cugraph test
Open 3 different terminals in `cugraph/dask` and run :
```
# dask scheduler in terminal 1 
dask-scheduler --scheduler-file cluster.json
```
```
# dask-mpi in terminal 2
mpirun -np 2 dask-mpi --no-nanny --nthreads 2 --no-scheduler --scheduler-file cluster.json
```
```
# dask-cugraph smoke test in terminal 3
python test_dask_cugraph.py
```
