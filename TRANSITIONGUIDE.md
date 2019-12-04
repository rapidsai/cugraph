# 0.11 release

## Python API Transition Guide

### New graph types
The Python API now has `Graph` (undirected) and `DiGraph` (directed) types. This is closer to NetworkX's API.

In the past, directed graphs were stored using the `Graph` type. 
Starting in 0.11, `DiGraph` should be used for directed graphs instead. `Graph` only refers to undirected graphs now.

The `Multi(Di)Graph` types were added and more support this new structure will be added in the next releases (more details on that in issue #604).

### Undirected graphs in 0.11
cuGraph automatically "symmetrize" undirected inputs: each undirected edge (u,v) is stored as two directed edges (u,v) and (v,u). 

When viewing the graph or requesting the number of edges, cuGraph will currently return this symmetrized view. 
This is an implementation detail that will be hidden to the user in 0.12 (more details on that in issue #603).  

### Loading an edge list
Users are encouraged to use `from_cudf_edge_list` instead of `add_edge_list`.

This new API supports cuDF DataFrame. Users can now ask for an automatic renumbering of the edge list at the time it is loaded. 
In this case all analytics outputs are automatically un-renumbered before being returned.

## C++ API Transition Guide
...

## Directory Structure and File Naming
`c_` prefix of all `.pxd` files have been removed.
