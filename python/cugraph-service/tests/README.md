# `cugraph_service` Tests

## Prerequisites
* The multi-GPU tests (`test_mg_*.py` files) require a `dask` scheduler and workers to be running on the test machine, with the path to the generated scheduler JSON file set in the env var `SCHEDULER_FILE`. These also assume the test machine has at least two GPUs, which can be accessed via device IDs 0 and 1.
  * When running on a multi-GPU machine with >2 GPUs, the `pytest` process can be limited to specific GPUs using the `CUDA_VISIBLE_DEVICES` env var.  For example, `export CUDA_VISIBLE_DEVICES=6,7` will limit the processes run in that environment to the two GPUs identified as 6 and 7, and within the process GPU 6 will be accessed as device `0`, GPU 7 will be device `1`.
  * The `dask` scheduler and workers can be run using the scripts in this repo: `<cugraph repo dir>/python/cugraph_service/scripts/run-dask-process.sh` (see `../README.md` for examples)

## End-to-end tests
* End-to-end (e2e) tests test code paths from the client to the server running in a separate process.
* e2e tests use pytest fixtures which automatically start a server subprocess in the background, and terminate it at the end of the test run(s). One challenge with this is STDOUT and STDERR is not currently redirected to the console running pytest, making debugging errors much harder.
* In order to debug in this situation, a user can start a server manually in the background prior to running pytest. If pytest detects a running server, it will use that instance instead of starting a new one. This allows the user to have access to the STDOUT and STDERR of the server process, as well as the ability to interactively debug it using `breakpoint()` calls if necessary.

## cugraph_handler tests
* cugraph_handler tests do not require a separate server process to be running, since these are tests which import the handler - just as the server script would - and run methods directly on it.  This tests the majorty of the code paths on the much larger server side, without the overhead of an e2e test.
* SG cugraph_handler tests are run in CI since 1) MG tests are not supported in CI, and 2) they provide the majority of the code coverage in a way that can be debugged without requiring access to separate processes (which would be difficult in the current CI system)

## client tests
* The client class is currently tested only through end-to-end tests, but in the future could be tested in isolation using a mock object to simulate interaction with a running server.
