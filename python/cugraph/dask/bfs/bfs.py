def bfs(edge_list, start, return_distances=False):
    """
    Find the distances and predecessors for a breadth first traversal of a
    graph.
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
    start : Integer
        The index of the graph vertex from which the traversal begins

    return_distances : bool, optional, default=False
        Indicates if distances should be returned

    Returns
    -------
    df : cudf.DataFrame
        df['vertex'][i] gives the vertex id of the i'th vertex

        df['distance'][i] gives the path distance for the i'th vertex from the
        starting vertex

        df['predecessor'][i] gives for the i'th vertex the vertex it was
        reached from in the traversal

    Examples
    --------
    >>> import dask_cugraph.bfs as dcg
    >>> chunksize = dcg.get_chunksize(edge_list.csv)
    >>> ddf_edge_list = dask_cudf.read_csv(edge_list.csv,
    >>>                                    chunksize = chunksize,
    >>>                                    delimiter='\t',
    >>>                                    names=['src', 'dst'],
    >>>                                    dtype=['int32', 'int32'])
    >>> pr = dcg.bfs(ddf_edge_list, start=0, return_distances=False)
    """

    raise Exception("mg_bfs currently disabled... "
                    "new OPG version coming soon")
