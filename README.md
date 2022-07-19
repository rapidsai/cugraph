# GaaS - _Graph-as-a-Service_

## Description
There are numerous cases where sharing graph processing and analytics on the same GPU can be problematic. Especially as the size of the graph grows and uses more of the limited GPU memory space.  Or in situations where there is a single graph and multiple distributed analytic clients. Graph-as-a-Service, or GaaS, is our solution to the problem.

[RAPIDS](https://rapids.ai) GaaS is a lightweight wrapper around [RAPIDS cuGraph](https://github.com/rapidsai/cugraph) that provides access to graph functionality via an RPC API.  This allows graph processing to be on separate hardware from analysis.  GaaS uses cuGraph, cuDF, and other libraries on the server to execute graph data prep and analysis on server-side GPUs. Multiple clients can connect to the server allowing different users and processes the ability to access large graph data that may not otherwise be possible using the client resources.

## <div align="center"><img src="img/gaas-pict.png" width="400px"/></div>


-----

## Server
(description)
### Installing the `gaas-server` conda package

    TBD

### Example
Starting the server
```
/repos/GaaS$ PYTHONPATH=./python python ./python/gaas_server/server.py
```

## Client
(description)
### Installing the `gaas-client` conda package

    TBD


### Example
Creating a client
```
>>> import gaas_client
>>> client = gaas_client.GaasClient()
>>> client.load_csv_as_vertex_data(...)
```

------

## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aims to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

