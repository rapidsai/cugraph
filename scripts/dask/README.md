# Dask scripts for multi-GPU environments

This directory contains tools for configuring environments for single-node or
multi-node, multi-gpu (SNMG or MNMG) Dask-based cugraph runs, currently
consisting of shell and python scripts.

Users should also consult the multi-GPU utilities in the
`python/cugraph/cugraph/testing/mg_utils.py` module, specifically the
`start_dask_client()` function, to see how to create `client` and `cluster`
instances in Python code to access the corresponding Dask processes created by
the tools here.


### run-dask-process.sh

 This script is used to start the Dask scheduler and workers as needed.

 To start a scheduler and workers on a node, run it like this:
 ```
 bash$ run-dask-process.sh scheduler workers
 ```
 Once a scheduler is running on a node in the cluster, workers can be started
 on other nodes in the cluster by running the script on each worker node like
 this:
 ```
 bash$ run-dask-process.sh workers
 ```
 The env var SCHEDULER_FILE must be set to the location where the scheduler
 will generate the scheduler JSON file. The same location is used by the
 workers to read the generated scheduler JSON file.

 The script will ensure the scheduler is started before the workers when both
 are specified.

### wait_for_workers.py

 This script can be used to ensure all workers that are expected to be present
 in the cluster are up and running. This is useful for automation that sets up
 the Dask cluster and cannot proceed until the Dask cluster is available
 to accept tasks.

 This example waits for 16 workers to be present:
 ```
 bash$ python wait_for_workers.py --scheduler-file-path=$SCHEDULER_FILE --num-expected-workers=16
 ```
