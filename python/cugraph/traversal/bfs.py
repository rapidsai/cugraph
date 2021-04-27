# Copyright (c) 2019-2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cudf

from cugraph.traversal import bfs_wrapper
from cugraph.structure.graph_classes import Graph, DiGraph
from cugraph.utilities import (ensure_cugraph_obj,
                               is_matrix_type,
                               is_cp_matrix_type,
                               import_optional,
                               )

# optional dependencies used for handling different input types
nx = import_optional("networkx")

cp = import_optional("cupy")
cp_coo_matrix = import_optional("coo_matrix",
                                import_from="cupyx.scipy.sparse.coo")
cp_csr_matrix = import_optional("csr_matrix",
                                import_from="cupyx.scipy.sparse.csr")
cp_csc_matrix = import_optional("csc_matrix",
                                import_from="cupyx.scipy.sparse.csc")

sp = import_optional("scipy")
sp_coo_matrix = import_optional("coo_matrix",
                                import_from="scipy.sparse.coo")
sp_csr_matrix = import_optional("csr_matrix",
                                import_from="scipy.sparse.csr")
sp_csc_matrix = import_optional("csc_matrix",
                                import_from="scipy.sparse.csc")


def _ensure_args(G, start, i_start, directed):
    """
    Ensures the args passed in are usable for the API api_name and returns the
    args with proper defaults if not specified, or raises TypeError or
    ValueError if incorrectly specified.
    """
    # checks common to all input types
    if (start is not None) and (i_start is not None):
        raise TypeError("cannot specify both 'start' and 'i_start'")
    if (start is None) and (i_start is None):
        raise TypeError("must specify 'start' or 'i_start', but not both")

    G_type = type(G)
    # Check for Graph-type inputs
    if (G_type in [Graph, DiGraph]) or \
       ((nx is not None) and (G_type in [nx.Graph, nx.DiGraph])):
        if directed is not None:
            raise TypeError("'directed' cannot be specified for a "
                            "Graph-type input")

    start = start if start is not None else i_start
    if directed is None:
        directed = True

    return (start, directed)


def _convert_df_to_output_type(df, input_type):
    """
    Given a cudf.DataFrame df, convert it to a new type appropriate for the
    graph algos in this module, based on input_type.
    """
    if input_type in [Graph, DiGraph]:
        return df

    elif (nx is not None) and (input_type in [nx.Graph, nx.DiGraph]):
        return df.to_pandas()

    elif is_matrix_type(input_type):
        # A CuPy/SciPy input means the return value will be a 2-tuple of:
        #   distance: cupy.ndarray
        #   predecessor: cupy.ndarray
        sorted_df = df.sort_values("vertex")
        if is_cp_matrix_type(input_type):
            distances = cp.fromDlpack(sorted_df["distance"].to_dlpack())
            preds = cp.fromDlpack(sorted_df["predecessor"].to_dlpack())
            return (distances, preds)
        else:
            distances = sorted_df["distance"].to_array()
            preds = sorted_df["predecessor"].to_array()
            return (distances, preds)
    else:
        raise TypeError(f"input type {input_type} is not a supported type.")


def bfs(G,
        start=None,
        depth_limit=None,
        i_start=None,
        directed=None,
        return_predecessors=None):
    """
    Find the distances and predecessors for a breadth first traversal of a
    graph.

    Parameters
    ----------
    G : cugraph.Graph, networkx.Graph, CuPy or SciPy sparse matrix
        Graph or matrix object, which should contain the connectivity
        information. Edge weights, if present, should be single or double
        precision floating point values.

    start : Integer
        The index of the graph vertex from which the traversal begins

    i_start : Integer, optional
        Identical to start, added for API compatibility. Only start or i_start
        can be set, not both.

    depth_limit : Integer or None
        Limit the depth of the search

    directed : bool, optional
        NOTE
            For non-Graph-type (eg. sparse matrix) values of G only. Raises
            TypeError if used with a Graph object.

        If True (default), then convert the input matrix to a cugraph.DiGraph,
        otherwise a cugraph.Graph object will be used.

    Returns
    -------
    Return value type is based on the input type.  If G is a cugraph.Graph,
    returns:

       cudf.DataFrame
          df['vertex'] vertex IDs

          df['distance'] path distance for each vertex from the starting vertex

          df['predecessor'] for each i'th position in the column, the vertex ID
          immediately preceding the vertex at position i in the 'vertex' column

    If G is a networkx.Graph, returns:

       pandas.DataFrame with contents equivalent to the cudf.DataFrame
       described above.

    If G is a CuPy or SciPy matrix, returns:
       a 2-tuple of CuPy ndarrays (if CuPy matrix input) or Numpy ndarrays (if
       SciPy matrix input) representing:

       distance: cupy or numpy ndarray
          ndarray of shortest distances between source and vertex.

       predecessor: cupy or numpy ndarray
          ndarray of predecessors of a vertex on the path from source, which
          can be used to reconstruct the shortest paths.

       ...or if return_sp_counter is True, returns a 3-tuple with the above two
       arrays plus:

       sp_counter: cupy or numpy ndarray
          ndarray of number of shortest paths leading to each vertex.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(M, source='0', destination='1')
    >>> df = cugraph.bfs(G, 0)

    """
    (start, directed) = \
        _ensure_args(G, start, i_start, directed)

    # FIXME: allow nx_weight_attr to be specified
    (G, input_type) = ensure_cugraph_obj(
        G, nx_weight_attr="weight",
        matrix_graph_type=DiGraph if directed else Graph)

    if G.renumbered is True:
        if isinstance(start, cudf.DataFrame):
            start = G.lookup_internal_vertex_id(start, start.columns).iloc[0]
        else:
            start = G.lookup_internal_vertex_id(cudf.Series([start]))[0]

    df = bfs_wrapper.bfs(G, start, depth_limit)
    if G.renumbered:
        df = G.unrenumber(df, "vertex")
        df = G.unrenumber(df, "predecessor")
        df.fillna(-1, inplace=True)

    return _convert_df_to_output_type(df, input_type)


def bfs_edges(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    """
    Find the distances and predecessors for a breadth first traversal of a
    graph.

    Parameters
    ----------
    G : cugraph.Graph, networkx.Graph, CuPy or SciPy sparse matrix
        Graph or matrix object, which should contain the connectivity
        information. Edge weights, if present, should be single or double
        precision floating point values.

    source : Integer
        The starting vertex index

    reverse : boolean
        If a directed graph, then process edges in a reverse direction
        Currently not implemented

    depth_limit : Int or None
        Limit the depth of the search

    sort_neighbors : None or Function
        Currently not implemented

    Returns
    -------
    Return value type is based on the input type.  If G is a cugraph.Graph,
    returns:

       cudf.DataFrame
          df['vertex'] vertex IDs

          df['distance'] path distance for each vertex from the starting vertex

          df['predecessor'] for each i'th position in the column, the vertex ID
          immediately preceding the vertex at position i in the 'vertex' column

    If G is a networkx.Graph, returns:

       pandas.DataFrame with contents equivalent to the cudf.DataFrame
       described above.

    If G is a CuPy or SciPy matrix, returns:
       a 2-tuple of CuPy ndarrays (if CuPy matrix input) or Numpy ndarrays (if
       SciPy matrix input) representing:

       distance: cupy or numpy ndarray
          ndarray of shortest distances between source and vertex.

       predecessor: cupy or numpy ndarray
          ndarray of predecessors of a vertex on the path from source, which
          can be used to reconstruct the shortest paths.

       ...or if return_sp_counter is True, returns a 3-tuple with the above two
       arrays plus:

       sp_counter: cupy or numpy ndarray
          ndarray of number of shortest paths leading to each vertex.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(M, source='0', destination='1')
    >>> df = cugraph.bfs_edges(G, 0)
    """

    if reverse is True:
        raise NotImplementedError(
            "reverse processing of graph is currently not supported"
        )

    return bfs(G, source, depth_limit)
