from cugraph.structure.graph_implementation import (
    simpleDistributedGraphImpl,
    simpleGraphImpl,
)


def call_cugraph_algorithm(name, graph, *args, **kwargs):
    # TODO check using graph property in a future PR
    if isinstance(graph._Impl, simpleDistributedGraphImpl):
        import cugraph.dask

        return getattr(cugraph.dask, name)(graph, *args, **kwargs)

    # TODO check using graph property in a future PR
    elif isinstance(graph._Impl, simpleGraphImpl):
        import cugraph

        return getattr(cugraph, name)(graph, *args, **kwargs)

    # TODO Properly dispatch for cugraph-service.
