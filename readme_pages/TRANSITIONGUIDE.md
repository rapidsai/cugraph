# 0.12

## Python API

### Loading an edge list
Renumbering is now enabled by default in `from_cudf_edgelist`. 
The renumbering feature allows us to hide the fact that we need vertices to be integers starting at 0. The auto-renumbering feature converts the data into the proper data type required by the underlying implementation. All algorithms accepting vertex identifiers (like the souce vertex for SSSP) now automatically accept user's notation too. On output, it will  transparently un-renumber results, basically convert the internal IDs back. 

## C++ API
Pagerank, BFS, and SSSP have dropped the `gdf_column` dependency in favor of basic types and templates  

Example : 
```
// 0.11 API 
cugraph::pagerank(cugraph::Graph, gdf_column *pagerank, ...) 
// 0.12 API 
cugraph::pagerank<int,float>(cugraph::Graph, float *pagerank ...) 
```

# 0.11

## Python API

This release introduces new concepts in the API and improves user experience through more automation and better DataFrame support. Python users are encouraged to review these changes and potentially upgrade their code before using 0.11 version.

### New graph types
The Python API now has `Graph` (undirected) and `DiGraph` (directed) types. This is closer to NetworkX's API.

In the past, directed graphs were stored using the `Graph` type. 
Starting in 0.11, `DiGraph` should be used for directed graphs instead. `Graph` only refers to undirected graphs now.

The `Multi(Di)Graph` types were added and more support for this new structure will be added in the next releases (more details in issue #604).

### Undirected graphs
cuGraph currently automatically "symmetrize" undirected inputs: each undirected edge (u,v) is stored as two directed edges (u,v) and (v,u). 

When viewing the graph or requesting the number of edges, cuGraph will currently return this symmetrized view. 
This is an implementation detail that will be hidden to the user in 0.12 (more details in issue #603).  

### Loading an edge list
Users are encouraged to use `from_cudf_edgelist` instead of `add_edge_list`.

This new API supports cuDF DataFrame. Users can now ask for an automatic renumbering of the edge list at the time it is loaded. 
In this case, all analytics outputs are automatically un-renumbered before being returned.

## C++ API

This release is the first step toward converting the former C-like API into a C++ API. Major changes have been made, C++ users should review these changes and upgrade their code before using 0.11 version.

- All C++ API functions are now encapsulated in namespace `cugraph`
- All functions are now processing errors through exceptions. As a result, `gdf_error` is no longer used as a return type and `void` is used instead.
- `gdf_graph` is now `Graph`. The content of the structure is still the same (will be upgraded in future releases).
- `gdf_` prefix has been removed from all C++ API functions.

Example :
```c
// < 0.11 API 
gdf_error err = gdf_pagerank(<gdf_graph>, ...) 
// >= 0.11 API 
cugraph::pagerank(<cugraph::Graph>, ...) 
```

The C++ API provides functions that efficiently convert between data formats and access to the efficient CUDA algorithms.  In 0.11, all automatic conversions and decision making were removed from the C++ layer.


## Directory Structure and File Naming
The `c_` prefix of all `.pxd` files has been removed.
