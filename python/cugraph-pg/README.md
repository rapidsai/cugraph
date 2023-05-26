# cugraph_pg

## Description
[RAPIDS](https://rapids.ai) cugraph-pg is a new experimental implementation of Property Graphs
intended to work well with PyData and RAPIDS ecosystems. A Property Graph is a graph whose
vertices and edges can have many attributes, and there may be many "kinds" of vertices or edges
that have different sets of properties. A typical workflow involves loading DataFrames (from
`cudf`, `pandas`, `dask_cudf`, etc.) of vertex and edge data into a PropertyGraph, filtering, and
extracting the graph to be used in GNNs or converting the graph and running cuGraph algorithms.

`cugraph-pg` is a work in progress and is not ready to use.
