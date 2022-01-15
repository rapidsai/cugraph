# Suggested documentation changes for cugraph/python/cugraph/cugraph

This markdown file serves to suggest changes in the documentation for the python implementation of cugraph. These notes are split up by the organization of the modules in the repository.

Key functions should have a complete documentation, which includes:

- a specific description of the function's purpose
- a list of all arguments the function accepts with information about what types are allowed and how the function changes as the argument changes
- a description of the return value(s) with their corresponding type(s)
- at least 1 example, with an example output if possible Helper functions need not have a complete documentation, though some sort of description is still necessary

Currently, there are some tests that have been commented out because of various reasons. They are the following tests:

- 

## centrality

## comms

## community

### community/egonet.py

- `ego_graph`'s example uses variable seed, which is different from parameter n. For now, this example was made to be partially commented out

## components

## cores

## dask

## layout

## linear_assignment

### linear_assignment/lap.py
- `hungarian`'s example requires a bipartite graph, but current dataset lacks this, this is also shown in the `test_hungarian.py`. Current example has been commented out

## link_analysis

## link_prediction

## proto

## sampling

## structure

## structure/graph_implementation 

## traversal

## tree

## utilities - could NOT make all tests passing

### utils.py

- `import_optional`'s examples require not knowing networkx nor cudf...