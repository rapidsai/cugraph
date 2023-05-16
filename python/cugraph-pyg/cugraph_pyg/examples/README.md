This directory contains examples for running cugraph-pyg training.

For single-GPU (SG) scripts, no special configuration is required.

For multi-GPU (MG) scripts, dask must be started first in a separate process.
To do this, the `start_dask.sh` script has been provided.  This scripts starts
a dask scheduler and dask workers.  To select the GPUs and amount of memory
allocated to dask per GPU, the `CUDA_VISIBLE_DEVICES` and `WORKER_RMM_POOL_SIZE`
arguments in that script can be modified.
To connect to dask, the scheduler JSON file must be provided.  This can be done
using the `--dask_scheduler_file` argument in the mg python script being run.
