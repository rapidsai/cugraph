# Copyright (c) 2021, NVIDIA CORPORATION.
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
from .graph_implementation import (simpleGraphImpl,
                                   simpleDistributedGraphImpl,
                                   npartiteGraphImpl)
import cudf
import warnings


# TODO: Move to utilities
def null_check(col):
    if col.null_count != 0:
        raise ValueError("Series contains NULL values")


class Graph:
    class Properties:
        def __init__(self, directed):
            self.directed = directed
            self.weights = False

    def __init__(self, m_graph=None, directed=False):
        self._Impl = None
        self.graph_properties = Graph.Properties(directed)
        if m_graph is not None:
            if m_graph.is_multigraph():
                elist = m_graph.view_edge_list()
                if m_graph.is_weighted():
                    weights = "weights"
                else:
                    weights = None
                self.from_cudf_edgelist(elist,
                                        source="src",
                                        destination="dst",
                                        edge_attr=weights)
            else:
                msg = (
                    "Graph can only be initialized using MultiGraph "
                    "or MultiDiGraph"
                )
                raise Exception(msg)

    def __getattr__(self, name):
        if self._Impl is None:
            raise AttributeError(name)
        if hasattr(self._Impl, name):
            return getattr(self._Impl, name)
        # FIXME: Remove access to Impl properties
        elif hasattr(self._Impl.properties, name):
            return getattr(self._Impl.properties, name)
        else:
            raise AttributeError(name)

    def __dir__(self):
        return dir(self._Impl)

    def from_cudf_edgelist(
        self,
        input_df,
        source="source",
        destination="destination",
        edge_attr=None,
        renumber=True
    ):
        """
        Initialize a graph from the edge list. It is an error to call this
        method on an initialized Graph object. The passed input_df argument
        wraps gdf_column objects that represent a graph using the edge list
        format. source argument is source column name and destination argument
        is destination column name.
        By default, renumbering is enabled to map the source and destination
        vertices into an index in the range [0, V) where V is the number
        of vertices.  If the input vertices are a single column of integers
        in the range [0, V), renumbering can be disabled and the original
        external vertex ids will be used.
        If weights are present, edge_attr argument is the weights column name.
        Parameters
        ----------
        input_df : cudf.DataFrame or dask_cudf.DataFrame
            A DataFrame that contains edge information
            If a dask_cudf.DataFrame is passed it will be reinterpreted as
            a cudf.DataFrame. For the distributed path please use
            from_dask_cudf_edgelist.
        source : str or array-like
            source column name or array of column names
        destination : str or array-like
            destination column name or array of column names
        edge_attr : str or None
            the weights column name. Default is None
        renumber : bool
            Indicate whether or not to renumber the source and destination
            vertex IDs. Default is True.
        Examples
        --------
        >>> df = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(df, source='0', destination='1',
                                 edge_attr='2', renumber=False)
        """
        if self._Impl is None:
            self._Impl = simpleGraphImpl(self.graph_properties)
        elif type(self._Impl) is not simpleGraphImpl:
            raise Exception("Graph is already initialized")
        elif (self._Impl.edgelist is not None or
              self._Impl.adjlist is not None):
            raise Exception("Graph already has values")
        self._Impl._simpleGraphImpl__from_edgelist(input_df,
                                                   source=source,
                                                   destination=destination,
                                                   edge_attr=edge_attr,
                                                   renumber=renumber)

    def from_cudf_adjlist(self, offset_col, index_col, value_col=None):
        """
        Initialize a graph from the adjacency list. It is an error to call this
        method on an initialized Graph object. The passed offset_col and
        index_col arguments wrap gdf_column objects that represent a graph
        using the adjacency list format.
        If value_col is None, an unweighted graph is created. If value_col is
        not None, a weighted graph is created.
        Undirected edges must be stored as directed edges in both directions.
        Parameters
        ----------
        offset_col : cudf.Series
            This cudf.Series wraps a gdf_column of size V + 1 (V: number of
            vertices).
            The gdf column contains the offsets for the vertices in this graph.
            Offsets must be in the range [0, E] (E: number of edges).
        index_col : cudf.Series
            This cudf.Series wraps a gdf_column of size E (E: number of edges).
            The gdf column contains the destination index for each edge.
            Destination indices must be in the range [0, V) (V: number of
            vertices).
        value_col : cudf.Series, optional
            This pointer can be ``None``.
            If not, this cudf.Series wraps a gdf_column of size E (E: number of
            edges).
            The gdf column contains the weight value for each edge.
            The expected type of the gdf_column element is floating point
            number.
        Examples
        --------
        >>> gdf = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> M = gdf.to_pandas()
        >>> M = scipy.sparse.coo_matrix((M['2'],(M['0'],M['1'])))
        >>> M = M.tocsr()
        >>> offsets = cudf.Series(M.indptr)
        >>> indices = cudf.Series(M.indices)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_adjlist(offsets, indices, None)
        """
        if self._Impl is None:
            self._Impl = simpleGraphImpl(self.graph_properties)
        elif type(self._Impl) is not simpleGraphImpl:
            raise Exception("Graph is already initialized")
        elif (self._Impl.edgelist is not None or
              self._Impl.adjlist is not None):
            raise Exception("Graph already has values")
        self._Impl._simpleGraphImpl__from_adjlist(offset_col,
                                                  index_col,
                                                  value_col)

    def from_dask_cudf_edgelist(
        self,
        input_ddf,
        source="source",
        destination="destination",
        edge_attr=None,
        renumber=True,
    ):
        """
        Initializes the distributed graph from the dask_cudf.DataFrame
        edgelist. Undirected Graphs are not currently supported.
        By default, renumbering is enabled to map the source and destination
        vertices into an index in the range [0, V) where V is the number
        of vertices.  If the input vertices are a single column of integers
        in the range [0, V), renumbering can be disabled and the original
        external vertex ids will be used.
        Note that the graph object will store a reference to the
        dask_cudf.DataFrame provided.
        Parameters
        ----------
        input_ddf : dask_cudf.DataFrame
            The edgelist as a dask_cudf.DataFrame
        source : str or array-like
            source column name or array of column names
        destination : str
            destination column name or array of column names
        edge_attr : str
            weights column name.
        renumber : bool
            If source and destination indices are not in range 0 to V where V
            is number of vertices, renumber argument should be True.
        """
        if self._Impl is None:
            self._Impl = simpleDistributedGraphImpl(self.graph_properties)
        elif type(self._Impl) is not simpleDistributedGraphImpl:
            raise Exception("Graph is already initialized")
        elif (self._Impl.edgelist is not None):
            raise Exception("Graph already has values")
        self._Impl._simpleDistributedGraphImpl__from_edgelist(input_ddf,
                                                              source,
                                                              destination,
                                                              edge_attr,
                                                              renumber)

    # Move to Compat Module
    def from_pandas_edgelist(
        self,
        pdf,
        source="source",
        destination="destination",
        edge_attr=None,
        renumber=True,
    ):
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
        Examples
        --------
        >>> df = pandas.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_pandas_edgelist(df, source='0', destination='1',
                                 edge_attr='2', renumber=False)
        """
        gdf = cudf.DataFrame.from_pandas(pdf)
        self.from_cudf_edgelist(gdf, source=source, destination=destination,
                                edge_attr=edge_attr, renumber=renumber)

    def from_pandas_adjacency(self, pdf):
        """
        Initializes the graph from pandas adjacency matrix
        """
        np_array = pdf.to_numpy()
        columns = pdf.columns
        self.from_numpy_array(np_array, columns)

    def from_numpy_array(self, np_array, nodes=None):
        """
        Initializes the graph from numpy array containing adjacency matrix.
        """
        src, dst = np_array.nonzero()
        weight = np_array[src, dst]
        df = cudf.DataFrame()
        if nodes is not None:
            df['src'] = nodes[src]
            df['dst'] = nodes[dst]
        else:
            df['src'] = src
            df['dst'] = dst
        df['weight'] = weight
        self.from_cudf_edgelist(df, 'src', 'dst', edge_attr='weight')

    def from_numpy_matrix(self, np_matrix):
        """
        Initializes the graph from numpy matrix containing adjacency matrix.
        """
        np_array = np.asarray(np_matrix)
        self.from_numpy_array(np_array)

    def unrenumber(self, df, column_name, preserve_order=False):
        """
        Given a DataFrame containing internal vertex ids in the identified
        column, replace this with external vertex ids.  If the renumbering
        is from a single column, the output dataframe will use the same
        name for the external vertex identifiers.  If the renumbering is from
        a multi-column input, the output columns will be labeled 0 through
        n-1 with a suffix of _column_name.
        Note that this function does not guarantee order in single GPU mode,
        and does not guarantee order or partitioning in multi-GPU mode.  If you
        wish to preserve ordering, add an index column to df and sort the
        return by that index column.
        Parameters
        ----------
        df: cudf.DataFrame or dask_cudf.DataFrame
            A DataFrame containing internal vertex identifiers that will be
            converted into external vertex identifiers.
        column_name: string
            Name of the column containing the internal vertex id.
        preserve_order: (optional) bool
            If True, preserve the order of the rows in the output
            DataFrame to match the input DataFrame
        Returns
        ---------
        df : cudf.DataFrame or dask_cudf.DataFrame
            The original DataFrame columns exist unmodified.  The external
            vertex identifiers are added to the DataFrame, the internal
            vertex identifier column is removed from the dataframe.
        """
        return self.renumber_map.unrenumber(df, column_name, preserve_order)

    def lookup_internal_vertex_id(self, df, column_name=None):
        """
        Given a DataFrame containing external vertex ids in the identified
        columns, or a Series containing external vertex ids, return a
        Series with the internal vertex ids.
        Note that this function does not guarantee order in single GPU mode,
        and does not guarantee order or partitioning in multi-GPU mode.
        Parameters
        ----------
        df: cudf.DataFrame, cudf.Series, dask_cudf.DataFrame, dask_cudf.Series
            A DataFrame containing external vertex identifiers that will be
            converted into internal vertex identifiers.
        column_name: (optional) string
            Name of the column containing the external vertex ids
        Returns
        ---------
        series : cudf.Series or dask_cudf.Series
            The internal vertex identifiers
        """
        return self.renumber_map.to_internal_vertex_id(df, column_name)

    def add_internal_vertex_id(
        self,
        df,
        internal_column_name,
        external_column_name,
        drop=True,
        preserve_order=False,
    ):
        """
        Given a DataFrame containing external vertex ids in the identified
        columns, return a DataFrame containing the internal vertex ids as the
        specified column name.  Optionally drop the external vertex id columns.
        Optionally preserve the order of the original DataFrame.
        Parameters
        ----------
        df: cudf.DataFrame or dask_cudf.DataFrame
            A DataFrame containing external vertex identifiers that will be
            converted into internal vertex identifiers.
        internal_column_name: string
            Name of column to contain the internal vertex id
        external_column_name: string or list of strings
            Name of the column(s) containing the external vertex ids
        drop: (optional) bool, defaults to True
            Drop the external columns from the returned DataFrame
        preserve_order: (optional) bool, defaults to False
            Preserve the order of the data frame (requires an extra sort)
        Returns
        ---------
        df : cudf.DataFrame or dask_cudf.DataFrame
            Original DataFrame with new column containing internal vertex
            id
        """
        return self.renumber_map.add_internal_vertex_id(
            df,
            internal_column_name,
            external_column_name,
            drop,
            preserve_order,
        )

    def clear(self):
        """
        Empty the graph.
        """
        self._Impl = None

    def is_bipartite(self):
        """
        Checks if Graph is bipartite. This solely relies on the user call of
        add_nodes_from with the bipartite parameter. This does not parse the
        graph to check if it is bipartite.
        """
        # TO DO: Call coloring algorithm
        return False

    def is_multipartite(self):
        """
        Checks if Graph is multipartite. This solely relies on the user call
        of add_nodes_from with the partition parameter. This does not parse
        the graph to check if it is multipartite.
        """
        # TO DO: Call coloring algorithm
        return False

    def is_multigraph(self):
        """
        Returns True if the graph is a multigraph. Else returns False.
        """
        # TO DO: Call coloring algorithm
        return False

    def is_directed(self):
        """
        Returns True if the graph is a directed graph.
        Returns False if the graph is an undirected graph.
        """
        return self.graph_properties.directed

    def is_renumbered(self):
        """
        Returns True if the graph is renumbered.
        """
        return self.properties.renumbered

    def is_weighted(self):
        """
        Returns True if the graph has edge weights.
        """
        return self.properties.weighted

    def has_isolated_vertices(self):
        """
        Returns True if the graph has isolated vertices.
        """
        return self.properties.isolated_vertices

    def to_directed(self):
        """
        Return a directed representation of the graph.
        This function sets the type of graph as DiGraph() and returns the
        directed view.
        Returns
        -------
        G : DiGraph
            A directed graph with the same nodes, and each edge (u,v,weights)
            replaced by two directed edges (u,v,weights) and (v,u,weights).
        Examples
        --------
        >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(M, '0', '1')
        >>> DiG = G.to_directed()
        """
        directed_graph = type(self)()
        directed_graph.graph_properties.directed = True
        directed_graph._Impl = type(self._Impl)(directed_graph.
                                                graph_properties)
        self._Impl.to_directed(directed_graph._Impl)
        return directed_graph

    def to_undirected(self):
        """
        Return an undirected copy of the graph.
        Returns
        -------
        G : Graph
            A undirected graph with the same nodes, and each directed edge
            (u,v,weights) replaced by an undirected edge (u,v,weights).
        Examples
        --------
        >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> DiG = cugraph.DiGraph()
        >>> DiG.from_cudf_edgelist(M, '0', '1')
        >>> G = DiG.to_undirected()
        """

        if self.graph_properties.directed is False:
            undirected_graph = type(self)()
        elif self.__class__.__bases__[0] == object:
            undirected_graph = type(self)()
        else:
            undirected_graph = self.__class__.__bases__[0]()
        undirected_graph._Impl = type(self._Impl)(undirected_graph.
                                                  graph_properties)
        self._Impl.to_undirected(undirected_graph._Impl)
        return undirected_graph

    def add_nodes_from(self, nodes):
        """
        Add nodes information to the Graph.
        Parameters
        ----------
        nodes : list or cudf.Series
            The nodes of the graph to be stored.
        """
        self._Impl._nodes["all_nodes"] = cudf.Series(nodes)

    # TODO: Add function
    # def properties():


class DiGraph(Graph):
    def __init__(self, m_graph=None):
        warnings.warn(
            "DiGraph is deprecated, use Graph(directed=True) instead",
            DeprecationWarning
        )
        super(DiGraph, self).__init__(m_graph, directed=True)


class MultiGraph(Graph):
    def __init__(self, directed=False):
        super(MultiGraph, self).__init__(directed=directed)
        self.graph_properties.multi_edge = True

    def is_multigraph(self):
        """
        Returns True if the graph is a multigraph. Else returns False.
        """
        # TO DO: Call coloring algorithm
        return True


class MultiDiGraph(MultiGraph):
    def __init__(self):
        warnings.warn(
            "MultiDiGraph is deprecated,\
 use MultiGraph(directed=True) instead",
            DeprecationWarning
        )
        super(MultiDiGraph, self).__init__(directed=True)


class Tree(Graph):
    def __init__(self, directed=False):
        super(Tree, self).__init__(directed=directed)
        self.graph_properties.tree = True


class NPartiteGraph(Graph):
    def __init__(self, bipartite=False, directed=False):
        super(NPartiteGraph, self).__init__(directed=directed)
        self.graph_properties.bipartite = bipartite
        self.graph_properties.multipartite = True

    def from_cudf_edgelist(
        self,
        input_df,
        source="source",
        destination="destination",
        edge_attr=None,
        renumber=True
    ):
        """
        Initialize a graph from the edge list. It is an error to call this
        method on an initialized Graph object. The passed input_df argument
        wraps gdf_column objects that represent a graph using the edge list
        format. source argument is source column name and destination argument
        is destination column name.
        By default, renumbering is enabled to map the source and destination
        vertices into an index in the range [0, V) where V is the number
        of vertices.  If the input vertices are a single column of integers
        in the range [0, V), renumbering can be disabled and the original
        external vertex ids will be used.
        If weights are present, edge_attr argument is the weights column name.
        Parameters
        ----------
        input_df : cudf.DataFrame or dask_cudf.DataFrame
            A DataFrame that contains edge information
            If a dask_cudf.DataFrame is passed it will be reinterpreted as
            a cudf.DataFrame. For the distributed path please use
            from_dask_cudf_edgelist.
        source : str or array-like
            source column name or array of column names
        destination : str or array-like
            destination column name or array of column names
        edge_attr : str or None
            the weights column name. Default is None
        renumber : bool
            Indicate whether or not to renumber the source and destination
            vertex IDs. Default is True.
        Examples
        --------
        >>> df = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.BiPartiteGraph()
        >>> G.from_cudf_edgelist(df, source='0', destination='1',
                                 edge_attr='2', renumber=False)
        """
        if self._Impl is None:
            self._Impl = npartiteGraphImpl(self.graph_properties)
        # API may change in future
        self._Impl._npartiteGraphImpl__from_edgelist(input_df,
                                                     source=source,
                                                     destination=destination,
                                                     edge_attr=edge_attr,
                                                     renumber=renumber)

    def from_dask_cudf_edgelist(
        self,
        input_ddf,
        source="source",
        destination="destination",
        edge_attr=None,
        renumber=True,
    ):
        """
        Initializes the distributed graph from the dask_cudf.DataFrame
        edgelist. Undirected Graphs are not currently supported.
        By default, renumbering is enabled to map the source and destination
        vertices into an index in the range [0, V) where V is the number
        of vertices.  If the input vertices are a single column of integers
        in the range [0, V), renumbering can be disabled and the original
        external vertex ids will be used.
        Note that the graph object will store a reference to the
        dask_cudf.DataFrame provided.
        Parameters
        ----------
        input_ddf : dask_cudf.DataFrame
            The edgelist as a dask_cudf.DataFrame
        source : str or array-like
            source column name or array of column names
        destination : str
            destination column name or array of column names
        edge_attr : str
            weights column name.
        renumber : bool
            If source and destination indices are not in range 0 to V where V
            is number of vertices, renumber argument should be True.
        """
        raise Exception("Distributed N-partite graph not supported")

    def add_nodes_from(self, nodes, bipartite=None, multipartite=None):
        """
        Add nodes information to the Graph.
        Parameters
        ----------
        nodes : list or cudf.Series
            The nodes of the graph to be stored. If bipartite and multipartite
            arguments are not passed, the nodes are considered to be a list of
            all the nodes present in the Graph.
        bipartite : str
            Sets the Graph as bipartite. The nodes are stored as a set of nodes
            of the partition named as bipartite argument.
        multipartite : str
            Sets the Graph as multipartite. The nodes are stored as a set of
            nodes of the partition named as multipartite argument.
        """
        if self._Impl is None:
            self._Impl = npartiteGraphImpl(self.graph_properties)
        if bipartite is None and multipartite is None:
            self._Impl._nodes["all_nodes"] = cudf.Series(nodes)
        else:
            self._Impl.add_nodes_from(nodes, bipartite=bipartite,
                                      multipartite=multipartite)

    def is_multipartite(self):
        """
        Checks if Graph is multipartite. This solely relies on the user call
        of add_nodes_from with the partition parameter and the Graph created.
        This does not parse the graph to check if it is multipartite.
        """
        return True


class BiPartiteGraph(NPartiteGraph):
    def __init__(self, directed=False):
        super(BiPartiteGraph, self).__init__(directed=directed, bipartite=True)

    def is_bipartite(self):
        """
        Checks if Graph is bipartite. This solely relies on the user call of
        add_nodes_from with the bipartite parameter and the Graph created.
        This does not parse the graph to check if it is bipartite.
        """
        return True


class BiPartiteDiGraph(BiPartiteGraph):
    def __init__(self):
        warnings.warn(
            "BiPartiteDiGraph is deprecated,\
 use BiPartiteGraph(directed=True) instead",
            DeprecationWarning
        )
        super(BiPartiteDiGraph, self).__init__(directed=True)


class NPartiteDiGraph(NPartiteGraph):
    def __init__(self):
        warnings.warn(
            "NPartiteDiGraph is deprecated,\
 use NPartiteGraph(directed=True) instead",
            DeprecationWarning
        )
        super(NPartiteGraph, self).__init__(directed=True)


def is_directed(G):
    """
    Returns True if the graph is a directed graph.
    Returns False if the graph is an undirected graph.
    """
    return G.is_directed()


def is_multigraph(G):
    """
    Returns True if the graph is a multigraph. Else returns False.
    """
    return G.is_multigraph()


def is_multipartite(G):
    """
    Checks if Graph is multipartite. This solely relies on the Graph
    type. This does not parse the graph to check if it is multipartite.
    """
    return G.is_multipatite()


def is_bipartite(G):
    """
    Checks if Graph is bipartite. This solely relies on the Graph type.
    This does not parse the graph to check if it is bipartite.
    """
    return G.is_bipartite()


def is_weighted(G):
    """
    Returns True if the graph has edge weights.
    """
    return G.is_weighted()
