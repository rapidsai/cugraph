# GaaS
_Graph-as-a-Service_

## Description
GaaS is a service with a corresponding lightweight client-side API that allows
for graph analysis on GPUs from outside processes.  GaaS uses cuGraph, cuDF, and
other libraries on the server to execute graph data prep and analysis on
server-side GPUs. Multiple clients can connect to the server allowing different
users and processes the ability to access large graph data that may not
otherwise be possible using the client resources.

## Server
(description)
### Installing the `gaas-server` conda package
### Example
Starting the server
```
/repos/GaaS$ PYTHONPATH=./python python ./python/gaas_server/server.py
```

## Client
(description)
### Installing the `gaas-client` conda package
### Example
Creating a client
```
>>> import gaas_client
>>> client = gaas_client.GaasClient()
>>> client.load_csv_as_vertex_data(...)
```
