import cugraph
from cugraph.structure.graph_implementation import simpleDistributedGraph, simpleGraph

def call_cugraph_algorithm(
    name,
    graph,
    *args,
    **kwargs
):
    # TODO check using graph property in a future PR
    if isinstance(graph._Impl, simpleDistributedGraph):
        import cugraph.dask
        getattr(cugraph.dask, name)(
            *args,
            **kwargs
        )

    # TODO check using graph property in a future PR
    elif isinstance(graph._Impl, simpleGraph):
        import cugraph
        getattr(cugraph, name)(
            *args,
            **kwargs
        )

    # TODO Properly dispatch for cugraph-service.
