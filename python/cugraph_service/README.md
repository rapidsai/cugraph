# cugraph_service

## Description
[RAPIDS](https://rapids.ai) cugraph_service provides an RPC interace to a remote [RAPIDS cuGraph](https://github.com/rapidsai/cugraph) session, allowing users to perform GPU accelerated graph analytics from a remote process. cugraph_service uses cuGraph, cuDF, and other libraries on the server to execute graph data prep and analysis on server-side GPUs. Multiple clients can connect to the server allowing different users and processes the ability to access large graph data that may not otherwise be possible using the client resources.

## <div align="center"><img src="img/cugraph_service_pict.png" width="400px"/></div>

-----

## Build & Install
Build and install the client first, then the server. This is necessary because the server depends on shared modules provided by the client.
```
$> cd cugraph_repo/python/cugraph_service/client
$> python setup.py install
$> cd ../server
$> python setup.py install
```

### Example
Starting a server for single-GPU-only cuGraph, using server extensions in `/my/cugraph_service/extensions`:
```
$> export PYTHONPATH=/rapids/cugraph/python/cugraph_service
$> python -m cugraph_service_server.server --graph-creation-extension-dir=/my/cugraph_service/extensions
```

Starting a server for multi-GPU cuGraph, same extensions:
```
$> export SCHEDULER_FILE=/tmp/scheduler.json
$> /rapids/cugraph/python/cugraph_service/scripts/run-dask-process.sh scheduler workers &
$> python -m cugraph_service_server.server --graph-creation-extension-dir=/my/cugraph_service/extensions --dask-scheduler-file=$SCHEDULER_FILE
```

### Example
Creating a client
```
>>> from cugraph_service_client import CugraphServiceClient
>>> client = CugraphServiceClient()
>>> client.load_csv_as_vertex_data(...)
```

### Debugging
#### UCX-Py related variables:
`UCX_TLS` - set the transports to use, in priority order. Example:
```
UCX_TLS=tcp,cuda_copy,cuda_ipc
```
`UCX_TCP_CM_REUSEADDR` - reuse addresses. This can be used to avoid "resource in use" errors during starting/restarting the service repeatedly.
```
UCX_TCP_CM_REUSEADDR=y
```
`UCX_LOG_LEVEL` - set the level for which UCX will output messages to the console. The example below will only output "ERROR" or higher. Set to "DEBUG" to see debug and higher messages.
```
UCX_LOG_LEVEL=ERROR
```

#### UCX performance checks:
Because cugraph-service uses UCX-Py for direct-to-client GPU data transfers when specified, it can be helpful to understand the various UCX performance chacks available to ensure cugraph-service is transfering results as efficiently as the system is capable of.
```
ucx_perftest -m cuda -t tag_bw -n 100 -s 16000 &
ucx_perftest -m cuda -t tag_bw -n 100 -s 16000 localhost
```
```
ucx_perftest -m cuda -t tag_bw -n 100 -s 1000000000 &
ucx_perftest -m cuda -t tag_bw -n 100 -s 1000000000 localhost
```
```
CUDA_VISIBLE_DEVICES=0,1 ucx_perftest -m cuda -t tag_bw -n 100 -s 16000 &
CUDA_VISIBLE_DEVICES=0,1 ucx_perftest -m cuda -t tag_bw -n 100 -s 16000 localhost
```
```
CUDA_VISIBLE_DEVICES=0,1 ucx_perftest -m cuda -t tag_bw -n 100 -s 1000000000 &
CUDA_VISIBLE_DEVICES=0,1 ucx_perftest -m cuda -t tag_bw -n 100 -s 1000000000 localhost
```
```
CUDA_VISIBLE_DEVICES=0,1 ucx_perftest -m cuda -t tag_bw -n 1000000 -s 1000000000 &
CUDA_VISIBLE_DEVICES=0,1 ucx_perftest -m cuda -t tag_bw -n 1000000 -s 1000000000 localhost
```

------

## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aims to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.
