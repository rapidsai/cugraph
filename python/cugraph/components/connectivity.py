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


from cugraph.utilities import (df_score_to_dictionary,
                               ensure_cugraph_obj,
                               is_matrix_type,
                               is_cp_matrix_type,
                               import_optional,
                               )
from cugraph.structure import Graph, DiGraph
from cugraph.components import connectivity_wrapper

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


def _ensure_args(api_name, G, directed, connection, return_labels):
    """
    Ensures the args passed in are usable for the API api_name and returns the
    args with proper defaults if not specified, or raises TypeError or
    ValueError if incorrectly specified.
    """
    G_type = type(G)
    # Check for Graph-type inputs and set defaults if unset
    if (G_type in [Graph, DiGraph]) or \
       ((nx is not None) and (G_type in [nx.Graph, nx.DiGraph])):
        exc_value = "'%s' cannot be specified for a Graph-type input"
        if directed is not None:
            raise TypeError(exc_value % "directed")
        if return_labels is not None:
            raise TypeError(exc_value % "return_labels")

        directed = True
        return_labels = True

    # Check for non-Graph-type inputs and set defaults if unset
    else:
        directed = True if (directed is None) else directed
        return_labels = True if (return_labels is None) else return_labels

    # Handle connection type, based on API being called
    if api_name == "strongly_connected_components":
        if (connection is not None) and (connection != "strong"):
            raise TypeError("'connection' must be 'strong' for "
                            f"{api_name}()")
        connection = "strong"
    elif api_name == "weakly_connected_components":
        if (connection is not None) and (connection != "weak"):
            raise TypeError("'connection' must be 'weak' for "
                            f"{api_name}()")
        connection = "weak"
    else:
        raise RuntimeError("invalid API name specified (internal): "
                           f"{api_name}")

    return (directed, connection, return_labels)


def _convert_df_to_output_type(df, input_type, return_labels):
    """
    Given a cudf.DataFrame df, convert it to a new type appropriate for the
    graph algos in this module, based on input_type.
    return_labels is only used for return values from cupy/scipy input types.
    """
    if input_type in [Graph, DiGraph]:
        return df

    elif (nx is not None) and (input_type in [nx.Graph, nx.DiGraph]):
        return df_score_to_dictionary(df, "labels", "vertex")

    elif is_matrix_type(input_type):
        # Convert DF of 2 columns (labels, vertices) to the SciPy-style return
        # value:
        #   n_components: int
        #       The number of connected components (number of unique labels).
        #   labels: ndarray
        #       The length-N array of labels of the connected components.
        n_components = len(df["labels"].unique())
        sorted_df = df.sort_values("vertex")
        if return_labels:
            if is_cp_matrix_type(input_type):
                labels = cp.fromDlpack(sorted_df["labels"].to_dlpack())
            else:
                labels = sorted_df["labels"].to_array()
            return (n_components, labels)
        else:
            return n_components

    else:
        raise TypeError(f"input type {input_type} is not a supported type.")


def weakly_connected_components(G,
                                directed=None,
                                connection=None,
                                return_labels=None):
    """
    Generate the Weakly Connected Components and attach a component label to
    each vertex.

    Parameters
    ----------
    G : cugraph.Graph, networkx.Graph, CuPy or SciPy sparse matrix

        Graph or matrix object, which should contain the connectivity
        information (edge weights are not used for this algorithm). If using a
        graph object, the graph can be either directed or undirected where an
        undirected edge is represented by a directed edge in both directions.
        The adjacency list will be computed if not already present.  The number
        of vertices should fit into a 32b int.

    directed : bool, optional

        NOTE: For non-Graph-type (eg. sparse matrix) values of G only. Raises
              TypeError if used with a Graph object.
        If True (default), then convert the input matrix to a cugraph.DiGraph
        and only move from point i to point j along paths csgraph[i, j]. If
        False, then find the shortest path on an undirected graph: the
        algorithm can progress from point i to j along csgraph[i, j] or
        csgraph[j, i].

    connection : str, optional

        Added for SciPy compatibility, can only be specified for non-Graph-type
        (eg. sparse matrix) values of G only (raises TypeError if used with a
        Graph object), and can only be set to "weak" for this API.

    return_labels : bool, optional

        NOTE: For non-Graph-type (eg. sparse matrix) values of G only. Raises
              TypeError if used with a Graph object.
        If True (default), then return the labels for each of the connected
        components.

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

    If G is a CuPy or SciPy matrix, returns:

       CuPy ndarray (if CuPy matrix input) or Numpy ndarray (if SciPy matrix
       input) of shape (<num vertices>, 2), where column 0 contains component
       identifiers and column 1 contains vertices.

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
    (directed, connection, return_labels) = _ensure_args(
        "weakly_connected_components", G, directed, connection, return_labels)

    # FIXME: allow nx_weight_attr to be specified
    (G, input_type) = ensure_cugraph_obj(
        G, nx_weight_attr="weight",
        matrix_graph_type=DiGraph if directed else Graph)

    df = connectivity_wrapper.weakly_connected_components(G)

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    return _convert_df_to_output_type(df, input_type, return_labels)


def strongly_connected_components(G,
                                  directed=None,
                                  connection=None,
                                  return_labels=None):
    """
    Generate the Strongly Connected Components and attach a component label to
    each vertex.

    Parameters
    ----------
    G : cugraph.Graph, networkx.Graph, CuPy or SciPy sparse matrix

        Graph or matrix object, which should contain the connectivity
        information (edge weights are not used for this algorithm). If using a
        graph object, the graph can be either directed or undirected where an
        undirected edge is represented by a directed edge in both directions.
        The adjacency list will be computed if not already present.  The number
        of vertices should fit into a 32b int.

    directed : bool, optional

        NOTE: For non-Graph-type (eg. sparse matrix) values of G only. Raises
              TypeError if used with a Graph object.
        If True (default), then convert the input matrix to a cugraph.DiGraph
        and only move from point i to point j along paths csgraph[i, j]. If
        False, then find the shortest path on an undirected graph: the
        algorithm can progress from point i to j along csgraph[i, j] or
        csgraph[j, i].

    connection : str, optional

        Added for SciPy compatibility, can only be specified for non-Graph-type
        (eg. sparse matrix) values of G only (raises TypeError if used with a
        Graph object), and can only be set to "strong" for this API.

    return_labels : bool, optional

        NOTE: For non-Graph-type (eg. sparse matrix) values of G only. Raises
              TypeError if used with a Graph object.
        If True (default), then return the labels for each of the connected
        components.

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

    If G is a CuPy or SciPy matrix, returns:

       CuPy ndarray (if CuPy matrix input) or Numpy ndarray (if SciPy matrix
       input) of shape (<num vertices>, 2), where column 0 contains component
       identifiers and column 1 contains vertices.

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
    (directed, connection, return_labels) = _ensure_args(
        "strongly_connected_components", G, directed,
        connection, return_labels)

    # FIXME: allow nx_weight_attr to be specified
    (G, input_type) = ensure_cugraph_obj(
        G, nx_weight_attr="weight",
        matrix_graph_type=DiGraph if directed else Graph)

    df = connectivity_wrapper.strongly_connected_components(G)

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    return _convert_df_to_output_type(df, input_type, return_labels)


def connected_components(G,
                         directed=None,
                         connection="weak",
                         return_labels=None):
    """
    Generate either the stronlgly or weakly connected components and attach a
    component label to each vertex.

    Parameters
    ----------
    G : cugraph.Graph, networkx.Graph, CuPy or SciPy sparse matrix

        Graph or matrix object, which should contain the connectivity
        information (edge weights are not used for this algorithm). If using a
        graph object, the graph can be either directed or undirected where an
        undirected edge is represented by a directed edge in both directions.
        The adjacency list will be computed if not already present.  The number
        of vertices should fit into a 32b int.

    directed : bool, optional

        NOTE: For non-Graph-type (eg. sparse matrix) values of G only. Raises
              TypeError if used with a Graph object.
        If True (default), then convert the input matrix to a cugraph.DiGraph
        and only move from point i to point j along paths csgraph[i, j]. If
        False, then find the shortest path on an undirected graph: the
        algorithm can progress from point i to j along csgraph[i, j] or
        csgraph[j, i].

    connection : str, optional

        [‘weak’|’strong’]. Return either weakly or strongly connected
        components.

    return_labels : bool, optional

        NOTE: For non-Graph-type (eg. sparse matrix) values of G only. Raises
              TypeError if used with a Graph object.
        If True (default), then return the labels for each of the connected
        components.

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

    If G is a CuPy or SciPy matrix, returns:

       CuPy ndarray (if CuPy matrix input) or Numpy ndarray (if SciPy matrix
       input) of shape (<num vertices>, 2), where column 0 contains component
       identifiers and column 1 contains vertices.

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
    if connection == "weak":
        return weakly_connected_components(G, directed,
                                           connection, return_labels)
    elif connection == "strong":
        return strongly_connected_components(G, directed,
                                             connection, return_labels)
    else:
        raise ValueError(f"invalid connection type: {connection}, "
                         "must be either 'strong' or 'weak'")
