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


from cugraph.utilities import (check_nx_graph,
                               df_score_to_dictionary,
                               ensure_cugraph_obj,
                               )
from cugraph.structure import (Graph,
                               DiGraph,
                               )
from cugraph.components import connectivity_wrapper

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
        return df_score_to_dictionary(df, "labels", "vertex")

    elif (cp is not None) and (input_type is cp_coo_matrix):
        # Convert DF of 2 columns (labels, vertices) to the SciPy-style return
        # value:
        #   n_components: int
        #       The number of connected components (number of unique labels).
        #   labels: ndarray
        #       The length-N array of labels of the connected components.
        n_components = len(df["labels"].unique())
        sorted_df = df.sort_values("vertex")
        labels = cp.fromDlpack(sorted_df["labels"].to_dlpack())
        return (n_components, labels)

    else:
        raise TypeError(f"input type {input_type} is not a supported type.")


def weakly_connected_components(G):
    """
    Generate the Weakly Connected Components and attach a component label to
    each vertex.

    Parameters
    ----------
    G : cugraph.Graph or networkx.Graph or CuPy sparse COO matrix
        cuGraph graph descriptor, should contain the connectivity information
        as an edge list (edge weights are not used for this algorithm).
        Currently, the graph should be undirected where an undirected edge is
        represented by a directed edge in both directions. The adjacency list
        will be computed if not already present. The number of vertices should
        fit into a 32b int.

    Returns
    -------
    Return value type is based on the input type.  If G is a cugraph.Graph,
    returns:

       cudf.DataFrame
           GPU data frame containing two cudf.Series of size V: the vertex
           identifiers and the corresponding component identifier.

           df['vertex']
               Contains the vertex identifier
           df['labels']
               The component identifier

    If G is a networkx.Graph, returns:

       python dictionary, where keys are vertices and values are the component
       identifiers.

    If G is a CuPy sparse COO matrix, returns:

       CuPy ndarray of shape (<num vertices>, 2), where column 0 contains
       component identifiers and column 1 contains vertices.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv',
                          delimiter = ' ',
                          dtype=['int32', 'int32', 'float32'],
                          header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(M, source='0', destination='1', edge_attr=None)
    >>> df = cugraph.weakly_connected_components(G)
    """
    (G, input_type) = ensure_cugraph_obj(G, coo_graph_type=DiGraph)

    df = connectivity_wrapper.weakly_connected_components(G)

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    return _convert_df_to_output_type(df, input_type)


def strongly_connected_components(G):
    """
    Generate the Stronlgly Connected Components and attach a component label to
    each vertex.

    Parameters
    ----------

    G : cugraph.Graph or networkx.Graph or CuPy sparse COO matrix
        cuGraph graph descriptor, should contain the connectivity information as
        an edge list (edge weights are not used for this algorithm). The graph
        can be either directed or undirected where an undirected edge is
        represented by a directed edge in both directions.  The adjacency list
        will be computed if not already present.  The number of vertices should
        fit into a 32b int.

    Returns
    -------
    Return value type is based on the input type.  If G is a cugraph.Graph,
    returns:

       cudf.DataFrame
           GPU data frame containing two cudf.Series of size V: the vertex
           identifiers and the corresponding component identifier.

           df['vertex']
               Contains the vertex identifier
           df['labels']
               The component identifier

    If G is a networkx.Graph, returns:

       python dictionary, where keys are vertices and values are the component
       identifiers.

    If G is a CuPy sparse COO matrix, returns:

       CuPy ndarray of shape (<num vertices>, 2), where column 0 contains
       component identifiers and column 1 contains vertices.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv',
                          delimiter = ' ',
                          dtype=['int32', 'int32', 'float32'],
                          header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(M, source='0', destination='1', edge_attr=None)
    >>> df = cugraph.strongly_connected_components(G)
    """
    (G, input_type) = ensure_cugraph_obj(G, coo_graph_type=DiGraph)

    df = connectivity_wrapper.strongly_connected_components(G)

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    return _convert_df_to_output_type(df, input_type)
