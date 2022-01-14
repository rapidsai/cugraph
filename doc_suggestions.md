# Suggested documentation changes for cugraph/python/cugraph/cugraph

This markdown file serves to suggest changes in the documentation for the python implementation of cugraph. These notes are split up by the organization of the modules in the repository.

Key functions should have a complete documentation, which includes:

a specific description of the function's purpose
a list of all arguments the function accepts with information about what types are allowed and how the function changes as the argument changes
a description of the return value(s) with their corresponding type(s)
at least 1 example, with an example output if possible Helper functions need not have a complete documentation, though some sort of description is still necessary

## centrality

## comms

## community

## components

## cores

## dask

## generators

## internals

## layout

## linear_assignment

## link_analysis - DONE

## link_prediction - DONE

## proto

## sampling - DONE

## structure - DONE

## structure/graph_implementation - DONE

## traversal - DONE

## tree - DONE

## utilities - could NOT make all tests passing

### utils.py

- `import_optional`'s examples require not knowing networkx nor cudf...