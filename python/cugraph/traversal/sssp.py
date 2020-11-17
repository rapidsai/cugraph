# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import numpy as np

import cudf
from cugraph.utilities import ensure_cugraph_obj
from cugraph.structure import Graph, DiGraph
from cugraph.traversal import sssp_wrapper

# optional dependencies used for handling different input types
try:
    import cupy as cp
    from cupyx.scipy.sparse.coo import coo_matrix as cp_coo_matrix
except ModuleNotFoundError:
    cp = None
try:
    import networkx as nx
except ModuleNotFoundError:
    nx = None


def _convert_df_to_output_type(df, input_type):
    """
    Given a cudf.DataFrame df, convert it to a new type appropriate for the
    graph algos in this module, based on input_type.
    """
    if input_type in [Graph, DiGraph]:
        return df

    elif (nx is not None) and (input_type in [nx.Graph, nx.DiGraph]):
        return df.to_pandas()

    elif (cp is not None) and (input_type is cp_coo_matrix):
        # A CuPy/SciPy input means the return value will be a 2-tuple of:
        #   distance: cupy.ndarray
        #   predecessor: cupy.ndarray
        sorted_df = df.sort_values("vertex")
        return (cp.fromDlpack(sorted_df["distance"].to_dlpack()),
                cp.fromDlpack(sorted_df["predecessor"].to_dlpack()))

    else:
        raise TypeError(f"input type {input_type} is not a supported type.")


# FIXME: if G is a Nx type, the weight attribute is assumed to be "weight", if
# set. An additional optional parameter for the weight attr name when accepting
# Nx graphs may be needed.  From the Nx docs:
# |      Many NetworkX algorithms designed for weighted graphs use
# |      an edge attribute (by default `weight`) to hold a numerical value.
def sssp(G, source):
    """
    Compute the distance and predecessors for shortest paths from the specified
    source to all the vertices in the graph. The distances column will store
    the distance from the source to each vertex. The predecessors column will
    store each vertex's predecessor in the shortest path. Vertices that are
    unreachable will have a distance of infinity denoted by the maximum value
    of the data type and the predecessor set as -1. The source vertex's
    predecessor is also set to -1. Graphs with negative weight cycles are not
    supported.

    Parameters
    ----------
    graph : cuGraph.Graph, NetworkX.Graph, or CuPy sparse COO matrix
        cuGraph graph descriptor with connectivity information. Edge weights,
        if present, should be single or double precision floating point values.
    source : int
        Index of the source vertex.

    Returns
    -------
    Return value type is based on the input type.  If G is a cugraph.Graph,
    returns:

       cudf.DataFrame
          df['vertex']
              vertex id

          df['distance']
              gives the path distance from the starting vertex

          df['predecessor']
              the vertex it was reached from

    If G is a networkx.Graph, returns:

       pandas.DataFrame with contents equivalent to the cudf.DataFrame
       described above.

    If G is a CuPy sparse COO matrix, returns a 2-tuple of cupy.ndarray:

       distance: cupy.ndarray
          ndarray of shortest distances between source and vertex.

       predecessor: cupy.ndarray
          ndarray of predecessors of a vertex on the path from source, which
          can be used to reconstruct the shortest paths.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(M, source='0', destination='1')
    >>> distances = cugraph.sssp(G, 0)
    """
    # FIXME: allow nx_weight_attr to be specified
    (G, input_type) = ensure_cugraph_obj(G,
                                         nx_weight_attr="weight",
                                         matrix_graph_type=Graph)

    if G.renumbered:
        source = G.lookup_internal_vertex_id(cudf.Series([source]))[0]

    df = sssp_wrapper.sssp(G, source)

    if G.renumbered:
        df = G.unrenumber(df, "vertex")
        df = G.unrenumber(df, "predecessor")
        df["predecessor"].fillna(-1, inplace=True)

    return _convert_df_to_output_type(df, input_type)


def filter_unreachable(df):
    """
    Remove unreachable vertices from the result of SSSP or BFS

    Parameters
    ----------
    df : cudf.DataFrame
        cudf.DataFrame that is the output of SSSP or BFS

    Returns
    -------
    df : filtered cudf.DataFrame with only reachable vertices
        df['vertex'][i] gives the vertex id of the i'th vertex.
        df['distance'][i] gives the path distance for the i'th vertex from the
        starting vertex.
        df['predecessor'][i] gives the vertex that was reached before the i'th
        vertex in the traversal.
    """
    if "distance" not in df:
        raise KeyError("No distance column found in input data frame")
    if np.issubdtype(df["distance"].dtype, np.integer):
        max_val = np.iinfo(df["distance"].dtype).max
        return df[df.distance != max_val]
    elif np.issubdtype(df["distance"].dtype, np.inexact):
        max_val = np.finfo(df["distance"].dtype).max
        return df[df.distance != max_val]
    else:
        raise TypeError("distance type unsupported")


def shortest_path(G, source):
    """
    Alias for sssp(), provided for API compatibility with NetworkX. See sssp()
    for details.
    """
    return sssp(G, source)
