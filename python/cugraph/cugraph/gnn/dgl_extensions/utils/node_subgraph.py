import cugraph
import cudf


def node_subgraph(
    pg,
    nodes=None,
    create_using=cugraph.MultiGraph,
):
    """
    Return a subgraph induced on the given nodes.

    A node-induced subgraph is a graph with edges whose endpoints are both
    in the specified node set.

    Parameters
    ----------
    pg: Property Graph
        The graph to create subgraph from
    nodes : Tensor
        The nodes to form the subgraph.
    Returns
    -------
    cuGraph
        The sampled subgraph with the same node ID space with the original
        graph.
    """

    _g = pg.extract_subgraph(create_using=create_using, check_multi_edges=True)

    if nodes is None:
        return _g
    else:
        _n = cudf.Series(nodes)
        _subg = cugraph.subgraph(_g, _n)
        return _subg
