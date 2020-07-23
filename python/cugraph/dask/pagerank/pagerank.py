def pagerank(edge_list, alpha=0.85, max_iter=30):
    """
    Find the PageRank values for each vertex in a graph using multiple GPUs.
    cuGraph computes an approximation of the Pagerank using the power method.
    The input edge list should be provided in dask-cudf dataframe
    with one partition per GPU.

    Parameters
    ----------
    edge_list : dask_cudf.DataFrame
        Contain the connectivity information as an edge list.
        Source 'src' and destination 'dst' columns must be of type 'int32'.
        Edge weights are not used for this algorithm.
        Indices must be in the range [0, V-1], where V is the global number
        of vertices.
    alpha : float
        The damping factor alpha represents the probability to follow an
        outgoing edge, standard value is 0.85.
        Thus, 1.0-alpha is the probability to “teleport” to a random vertex.
        Alpha should be greater than 0.0 and strictly lower than 1.0.
    max_iter : int
        The maximum number of iterations before an answer is returned.
        If this value is lower or equal to 0 cuGraph will use the default
        value, which is 30.

    Returns
    -------
    PageRank : dask_cudf.DataFrame
        Dask GPU DataFrame containing two columns of size V: the vertex
        identifiers and the corresponding PageRank values.

    Examples
    --------
    >>> import dask_cugraph.pagerank as dcg
    >>> chunksize = dcg.get_chunksize(edge_list.csv)
    >>> ddf_edge_list = dask_cudf.read_csv(edge_list.csv,
    >>>                                    chunksize = chunksize,
    >>>                                    delimiter='\t',
    >>>                                    names=['src', 'dst'],
    >>>                                    dtype=['int32', 'int32'])
    >>> pr = dcg.pagerank(ddf_edge_list, alpha=0.85, max_iter=50)
    """

    raise Exception("mg_pagerank currently disabled... "
                    "new MG version coming soon")
