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

# this file is pure python and no need to be a cython file. Once cugraph's
# issue #146 is addressed, this file's extension should be changed from .pyx to
# .py and should be located outside the python/cugraph/bindings directory.

import cudf
import dask_cudf

from cugraph.structure.graph import DiGraph, Graph

# optional dependencies used for handling different input types
try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


def from_edgelist(df, source='source', destination='destination',
                  edge_attr=None, create_using=Graph, renumber=True):
    """
    Return a new graph created from the edge list representaion.

    Parameters
    ----------
    df : cudf.DataFrame, pandas.DataFrame, dask_cudf.core.DataFrame
        This DataFrame contains columns storing edge source vertices,
        destination (or target following NetworkX's terminology) vertices, and
        (optional) weights.
    source : string or integer
        This is used to index the source column.
    destination : string or integer
        This is used to index the destination (or target following NetworkX's
        terminology) column.
    edge_attr : string or integer, optional
        This pointer can be ``None``. If not, this is used to index the weight
        column.
    create_using : cuGraph.Graph
        Specify the type of Graph to create.  Default is cugraph.Graph
    renumber : bool
        If source and destination indices are not in range 0 to V where V
        is number of vertices, renumber argument should be True.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G = cugraph.from_edgelist(M, source='0', destination='1',
                                  edge_attr='2')
    """
    df_type = type(df)

    if df_type is cudf.DataFrame:
        return from_cudf_edgelist(df, source, destination,
                                  edge_attr, create_using, renumber)

    elif (pd is not None) and (df_type is pd.DataFrame):
        return from_pandas_edgelist(df, source, destination,
                                    edge_attr, create_using, renumber)

    elif df_type is dask_cudf.core.DataFrame:
        if create_using in [Graph, DiGraph]:
            G = create_using()
        else:
            raise TypeError(f"'create_using' is type {create_using}, must be "
                            "either a cugraph.Graph or cugraph.DiGraph")
        G.from_dask_cudf_edgelist(df, source, destination, edge_attr, renumber)
        return G

    else:
        raise TypeError(f"obj of type {df_type} is not supported.")


def from_adjlist(offsets, indices, values=None, create_using=Graph):
    """
    Initializes the graph from cuDF or Pandas Series representing adjacency
    matrix CSR data and returns a new cugraph.Graph object if 'create_using' is
    set to cugraph.Graph (the default), or cugraph.DiGraph if 'create_using' is
    set to cugraph.DiGraph.

    Parameters
    ----------
    offsets : cudf.Series, pandas.Series
        The offsets of a CSR adjacency matrix.
    indices : cudf.Series, pandas.Series
        The indices of a CSR adjacency matrix.
    values : cudf.Series, pandas.Series, or None (default), optional
        The values in a CSR adjacency matrix, which represent edge weights in a
        graph. If not provided, the resulting graph is considered unweighted.
    create_using : cuGraph.Graph
        Specify the type of Graph to create.  Default is cugraph.Graph

    Examples
    --------
    >>> pdf = pd.read_csv('datasets/karate.csv', delimiter=' ',
    ...                   dtype={0:'int32', 1:'int32', 2:'float32'},
    ...                   header=None)
    >>> M = scipy.sparse.coo_matrix((pdf[2],(pdf[0],pdf[1])))
    >>> M = M.tocsr()
    >>> offsets = pd.Series(M.indptr)
    >>> indices = pd.Series(M.indices)
    >>> G = cugraph.from_adjlist(offsets, indices, None)
    """
    offsets_type = type(offsets)
    indices_type = type(indices)
    if offsets_type != indices_type:
        raise TypeError(f"'offsets' type {offsets_type} != 'indices' "
                        f"type {indices_type}")
    if values is not None:
        values_type = type(values)
        if values_type != offsets_type:
            raise TypeError(f"'values' type {values_type} != 'offsets' "
                            f"type {offsets_type}")

    if create_using in [Graph, DiGraph]:
        G = create_using()
    else:
        raise TypeError(f"'create_using' is type {create_using}, must be "
                        "either a cugraph.Graph or cugraph.DiGraph")

    if offsets_type is cudf.Series:
        G.from_cudf_adjlist(offsets, indices, values)

    elif (pd is not None) and (offsets_type is pd.Series):
        G.from_cudf_adjlist(cudf.Series(offsets), cudf.Series(indices),
                            None if values is None else cudf.Series(values))

    else:
        raise TypeError(f"obj of type {offsets_type} is not supported.")

    return G


def from_cudf_edgelist(df, source='source', destination='destination',
                       edge_attr=None, create_using=Graph, renumber=True):
    """
    Return a new graph created from the edge list representaion. This function
    is added for NetworkX compatibility (this function is a RAPIDS version of
    NetworkX's from_pandas_edge_list()).  This function does not support
    multiple source or destination columns.  But does support renumbering

    Parameters
    ----------
    df : cudf.DataFrame
        This cudf.DataFrame contains columns storing edge source vertices,
        destination (or target following NetworkX's terminology) vertices, and
        (optional) weights.
    source : string or integer
        This is used to index the source column.
    destination : string or integer
        This is used to index the destination (or target following NetworkX's
        terminology) column.
    edge_attr : string or integer, optional
        This pointer can be ``None``. If not, this is used to index the weight
        column.
    create_using : cuGraph.Graph
        Specify the type of Graph to create.  Default is cugraph.Graph
    renumber : bool
        If source and destination indices are not in range 0 to V where V
        is number of vertices, renumber argument should be True.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G = cugraph.from_cudf_edgelist(M, source='0', target='1', weight='2')
    """
    if create_using is Graph:
        G = Graph()
    elif create_using is DiGraph:
        G = DiGraph()
    else:
        raise Exception("create_using supports Graph and DiGraph")

    G.from_cudf_edgelist(df, source=source, destination=destination,
                         edge_attr=edge_attr, renumber=renumber)

    return G


def from_pandas_edgelist(df,
                         source="source",
                         destination="destination",
                         edge_attr=None,
                         create_using=Graph,
                         renumber=True):
    """
    Initialize a graph from the edge list. It is an error to call this
    method on an initialized Graph object. Source argument is source
    column name and destination argument is destination column name.

    By default, renumbering is enabled to map the source and destination
    vertices into an index in the range [0, V) where V is the number
    of vertices.  If the input vertices are a single column of integers
    in the range [0, V), renumbering can be disabled and the original
    external vertex ids will be used.

    If weights are present, edge_attr argument is the weights column name.

    Parameters
    ----------
    input_df : pandas.DataFrame
        A DataFrame that contains edge information
    source : str or array-like
        source column name or array of column names
    destination : str or array-like
        destination column name or array of column names
    edge_attr : str or None
        the weights column name. Default is None
    renumber : bool
        Indicate whether or not to renumber the source and destination
        vertex IDs. Default is True.
    create_using: cugraph.DiGraph or cugraph.Graph
        Indicate whether to create a directed or undirected graph

    Returns
    -------
    G : cugraph.DiGraph or cugraph.Graph
        graph containing edges from the pandas edgelist

    Examples
    --------
    >>> df = pandas.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_pandas_edgelist(df, source='0', destination='1',
                               edge_attr='2', renumber=False)
    """
    if create_using is Graph:
        G = Graph()
    elif create_using is DiGraph:
        G = DiGraph()
    else:
        raise Exception("create_using supports Graph and DiGraph")

    G.from_pandas_edgelist(df, source=source, destination=destination,
                           edge_attr=edge_attr, renumber=renumber)
    return G


def to_pandas_edgelist(G, source='source', destination='destination'):
    """
    Returns the graph edge list as a Pandas DataFrame.

    Parameters
    ----------
    G : cugraph.Graph or cugraph.DiGraph
        Graph containg the edgelist.
    source : str or array-like
        source column name or array of column names
    destination : str or array-like
        destination column name or array of column names

    Returns
    ------
    df : pandas.DataFrame
        pandas dataframe containing the edgelist as source and
        destination columns.
    """
    pdf = G.to_pandas_edgelist(source=source, destination=destination)
    return pdf


def from_pandas_adjacency(df, create_using=Graph):
    """
    Initializes the graph from pandas adjacency matrix.
    Set create_using to cugraph.DiGraph for directed graph and
    cugraph.Graph for undirected Graph.
    """
    if create_using is Graph:
        G = Graph()
    elif create_using is DiGraph:
        G = DiGraph()
    else:
        raise Exception("create_using supports Graph and DiGraph")

    G.from_pandas_adjacency(df)
    return G


def to_pandas_adjacency(G):
    """
    Returns the graph adjacency matrix as a Pandas DataFrame.
    The row indices denote source and column names denote destination.
    """
    pdf = G.to_pandas_adjacency()
    return pdf


def from_numpy_array(A, create_using=Graph):
    """
    Initializes the graph from numpy array containing adjacency matrix.
    Set create_using to cugraph.DiGraph for directed graph and
    cugraph.Graph for undirected Graph.
    """
    if create_using is Graph:
        G = Graph()
    elif create_using is DiGraph:
        G = DiGraph()
    else:
        raise Exception("create_using supports Graph and DiGraph")

    G.from_numpy_array(A)
    return G


def to_numpy_array(G):
    """
    Returns the graph adjacency matrix as a NumPy array.
    """
    A = G.to_numpy_array()
    return A


def from_numpy_matrix(A, create_using=Graph):
    """
    Initializes the graph from numpy matrix containing adjacency matrix.
    Set create_using to cugraph.DiGraph for directed graph and
    cugraph.Graph for undirected Graph.
    """
    if create_using is Graph:
        G = Graph()
    elif create_using is DiGraph:
        G = DiGraph()
    else:
        raise Exception("create_using supports Graph and DiGraph")
    G.from_numpy_matrix(A)
    return G


def to_numpy_matrix(G):
    """
    Returns the graph adjacency matrix as a NumPy matrix.
    """
    A = G.to_numpy_matrix()
    return A
