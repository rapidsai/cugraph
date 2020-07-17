# Copyright (c) 2019, NVIDIA CORPORATION.
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

from cugraph.structure import graph_new_wrapper
from cugraph.structure.symmetrize import symmetrize
from cugraph.structure.renumber import renumber as rnb
from cugraph.structure.renumber import renumber_from_cudf as multi_rnb
from cugraph.dask.common.input_utils import get_local_data
import cudf
import dask_cudf
import dask
import numpy as np
import warnings


def null_check(col):
    if col.null_count != 0:
        raise ValueError('Series contains NULL values')


class Graph:

    class EdgeList:
        def __init__(self, *args):
            if len(args) <= 2:
                self.__from_dask_cudf(*args)
            else:
                self.__from_cudf(*args)

        def __from_cudf(self, source, destination, edge_attr=None,
                        renumber_map=None):
            self.renumber_map = renumber_map
            self.edgelist_df = cudf.DataFrame()
            self.edgelist_df['src'] = source
            self.edgelist_df['dst'] = destination
            self.weights = False
            if edge_attr is not None:
                self.weights = True
                if type(edge_attr) is dict:
                    for k in edge_attr.keys():
                        self.edgelist_df[k] = edge_attr[k]
                else:
                    self.edgelist_df['weights'] = edge_attr

        def __from_dask_cudf(self, ddf, renumber_map=None):
            self.renumber_map = renumber_map
            self.edgelist_df = ddf
            self.weights = False
            # FIXME: Edge Attribute not handled

    class AdjList:
        def __init__(self, offsets, indices, value=None):
            self.offsets = offsets
            self.indices = indices
            self.weights = value  # Should be a dataframe for multiple weights

    class transposedAdjList:
        def __init__(self, offsets, indices, value=None):
            Graph.AdjList.__init__(self, offsets, indices, value)
    """
    cuGraph graph class containing basic graph creation and transformation
    operations.
    """
    def __init__(self, m_graph=None, edge_attr=None, symmetrized=False,
                 bipartite=False, multi=False, dynamic=False):
        """
        Returns
        -------
        G : cuGraph.Graph.

        Examples
        --------
        >>> import cuGraph
        >>> G = cuGraph.Graph()

        """
        self.symmetrized = symmetrized
        self.renumbered = False
        self.bipartite = bipartite
        self.multi = multi
        self.distributed = False
        self.replicatable = False
        self.dynamic = dynamic
        self.edgelist = None
        self.adjlist = None
        self.transposedadjlist = None
        self.edge_count = None
        self.node_count = None
        if m_graph is not None:
            if ((type(self) is Graph and type(m_graph) is MultiGraph)
               or (type(self) is DiGraph and type(m_graph) is MultiDiGraph)):
                self.from_cudf_edgelist(m_graph.edgelist.edgelist_df,
                                        source='src',
                                        destination='dst',
                                        edge_attr=edge_attr)
                self.renumbered = m_graph.renumbered
                self.edgelist.renumber_map = m_graph.edgelist.renumber_map
            else:
                msg = "Graph can be initialized using MultiGraph\
 and DiGraph can be initialized using MultiDiGraph"
                raise Exception(msg)
        # self.number_of_vertices = None

    def clear(self):
        """
        Empty this graph. This function is added for NetworkX compatibility.
        """
        self.edgelist = None
        self.adjlist = None
        self.transposedadjlist = None

    def from_cudf_edgelist(self, input_df, source='source',
                           destination='destination',
                           edge_attr=None, renumber=True):
        """
        Initialize a graph from the edge list. It is an error to call this
        method on an initialized Graph object. The passed input_df argument
        wraps gdf_column objects that represent a graph using the edge list
        format. source argument is source column name and destination argument
        is destination column name.
        Source and destination indices must be in the range [0, V) where V is
        the number of vertices. If renumbering needs to be done, renumber
        argument should be passed as True.
        If weights are present, edge_attr argument is the weights column name.

        Parameters
        ----------
        input_df : cudf.DataFrame or dask_cudf.DataFrame
            This cudf.DataFrame wraps source, destination and weight
            gdf_column of size E (E: number of edges)
            The 'src' column contains the source index for each edge.
            Source indices are in the range [0, V) (V: number of vertices).
            The 'dst' column contains the destination index for each edge.
            Destination indices are in the range [0, V) (V: number of
            vertices).
            If renumbering needs to be done, renumber
            argument should be passed as True.
            For weighted graphs, dataframe contains 'weight' column
            containing the weight value for each edge.
            If a dask_cudf.DataFrame is passed it will be reinterpreted as
            a cudf.DataFrame. For the distributed path please use
            from_dask_cudf_edgelist.
        source : str
            source argument is source column name
        destination : str
            destination argument is destination column name.
        edge_attr : str
            edge_attr argument is the weights column name.
        renumber : bool
            If source and destination indices are not in range 0 to V where V
            is number of vertices, renumber argument should be True.

        Examples
        --------
        >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(M, source='0', destination='1', edge_attr='2',
                                 renumber=False)

        """
        if self.edgelist is not None or self.adjlist is not None:
            raise Exception('Graph already has values')

        # Consolidation
        if isinstance(input_df, cudf.DataFrame):
            if len(input_df[source]) > 2147483100:
                raise Exception('cudf dataFrame edge list is too big \
                                 to fit in a single GPU')
            elist = input_df
        elif isinstance(input_df, dask_cudf.DataFrame):
            if len(input_df[source]) > 2147483100:
                raise Exception('dask_cudf dataFrame edge list is too big \
                                 to fit in a single GPU')
            elist = input_df.compute().reset_index(drop=True)
        else:
            raise Exception('input should be a cudf.DataFrame or \
                              a dask_cudf dataFrame')

        if self.multi:
            if type(edge_attr) is not list:
                raise Exception('edge_attr should be a list of column names')
            value_col = {}
            for col_name in edge_attr:
                value_col[col_name] = elist[col_name]
        elif edge_attr is not None:
            value_col = elist[edge_attr]
        else:
            value_col = None
        renumber_map = None
        if renumber:
            if type(source) is list and type(destination) is list:
                source_col, dest_col, renumber_map = multi_rnb(elist,
                                                               source,
                                                               destination)
            else:
                source_col, dest_col, renumber_map = rnb(elist[source],
                                                         elist[destination])
            self.renumbered = True
        else:
            if type(source) is list and type(destination) is list:
                raise Exception('set renumber to True for multi column ids')
            else:
                source_col = elist[source]
                dest_col = elist[destination]
        if not self.symmetrized and not self.multi:
            if value_col is not None:
                source_col, dest_col, value_col = symmetrize(source_col,
                                                             dest_col,
                                                             value_col)
            else:
                source_col, dest_col = symmetrize(source_col, dest_col)

        self.edgelist = Graph.EdgeList(source_col, dest_col, value_col,
                                       renumber_map)

    def add_edge_list(self, source, destination, value=None):
        warnings.warn('add_edge_list will be deprecated in next release.\
 Use from_cudf_edgelist instead')
        input_df = cudf.DataFrame()
        input_df['source'] = source
        input_df['destination'] = destination
        if value is not None:
            input_df['weights'] = value
            self.from_cudf_edgelist(input_df, edge_attr='weights')
        else:
            self.from_cudf_edgelist(input_df)

    def from_dask_cudf_edgelist(self, input_ddf):
        """
        Initializes the distributed graph from the dask_cudf.DataFrame
        edgelist. Renumbering and undirected Graphs are not currently
        supported.
        Parameters
        ----------
        input_ddf : dask_cudf.DataFrame
            The edgelist as a dask_cudf.DataFrame
        """
        if self.edgelist is not None or self.adjlist is not None:
            raise Exception('Graph already has values')
        if not isinstance(input_ddf, dask_cudf.DataFrame):
            raise Exception('input should be a dask_cudf dataFrame')
        self.distributed = True
        self.replicatable = (input_ddf.npartitions == 1)
        self.local_data = None

        if not self.replicatable:  # MG Distributed
            if type(self) is Graph:
                raise Exception('Undirected distributed graph not supported')
            self.edgelist = self.EdgeList(input_ddf)
        else:  # MG Batch
            renumber = True  # FIXME: Handle option
            edge_attr = None  # FIXME: Handle weights attributes
            source = 'src'
            destination = 'dst'
            if self.multi:
                if type(edge_attr) is not list:
                    raise Exception('edge_attr should be a list of column'
                                    'names')
                value_col = {}
                for col_name in edge_attr:
                    value_col[col_name] = input_ddf[col_name]
            elif edge_attr is not None:
                value_col = input_ddf[edge_attr]
            else:
                value_col = None
            renumber_map = None
            if renumber:
                if type(source) is list and type(destination) is list:
                    source_col, dest_col, renumber_map = multi_rnb(input_ddf,
                                                                   source,
                                                                   destination)
                else:
                    source_col, dest_col, renumber_map = rnb(
                        input_ddf[source].compute(),
                        input_ddf[destination].compute())
                self.renumbered = True
            else:
                if type(source) is list and type(destination) is list:
                    raise Exception('set renumber to True for multi column'
                                    'ids')
                else:
                    source_col = input_ddf[source]
                    dest_col = input_ddf[destination]
            if not self.symmetrized and not self.multi:
                if value_col is not None:
                    source_col, dest_col, value_col = symmetrize(source_col,
                                                                 dest_col,
                                                                 value_col)
                else:
                    source_col, dest_col = symmetrize(source_col, dest_col)

            df = dask.delayed(cudf.DataFrame)({'src': source_col,
                                               'dst': dest_col},
                                              dtype=np.int32)
            new_ddf = dask_cudf.from_cudf(df.compute(), npartitions=1)
            new_ddf = new_ddf.persist()
            self.edgelist = Graph.EdgeList(new_ddf, renumber_map)

    def compute_local_data(self, by, load_balance=True):
        """
        Compute the local edges, vertices and offsets for a distributed
        graph stored as a dask-cudf dataframe and initialize the
        communicator. Performs global sorting and load_balancing.

        Parameters
        ----------
        by : str
            by argument is the column by which we want to sort and
            partition. It should be the source column name for generating
            CSR format and destination column name for generating CSC
            format.
        load_balance : bool
            Set as True to perform load_balancing after global sorting of
            dask-cudf DataFrame. This ensures that the data is uniformly
            distributed among multiple GPUs to avoid over-loading.
        """
        if self.distributed:
            data = get_local_data(self, by, load_balance)
            self.local_data = {}
            self.local_data['data'] = data
            self.local_data['by'] = by
        else:
            raise Exception('Graph should be a distributed graph')

    def view_edge_list(self):
        """
        Display the edge list. Compute it if needed.

        NOTE: If the graph is of type Graph() then the displayed undirected
        edges are the same as displayed by networkx Graph(), but the direction
        could be different i.e. an edge displayed by cugraph as (src, dst)
        could be displayed as (dst, src) by networkx.

        cugraph.Graph stores symmetrized edgelist internally. For displaying
        undirected edgelist for a Graph the upper trianglar matrix of the
        symmetrized edgelist is returned.

        networkx.Graph renumbers the input and stores the upper triangle of
        this renumbered input. Since the internal renumbering of networx and
        cugraph is different, the upper triangular matrix of networkx
        renumbered input may not be the same as cugraph's upper trianglar
        matrix of the symmetrized edgelist. Hence the displayed source and
        destination pairs in both will represent the same edge but node values
        could be swapped.

        Returns
        -------
        edgelist_df : cudf.DataFrame
            This cudf.DataFrame wraps source, destination and weight
            gdf_column of size E (E: number of edges)
            The 'src' column contains the source index for each edge.
            Source indices are in the range [0, V) (V: number of vertices).
            The 'dst' column contains the destination index for each edge.
            Destination indices are in the range [0, V) (V: number of
            vertices).
            For weighted graphs, dataframe contains 'weight' column
            containing the weight value for each edge.
        """
        if self.distributed:
            if self.edgelist is None:
                raise Exception("Graph has no Edgelist.")
            return self.edgelist.edgelist_df
        if self.edgelist is None:
            src, dst, weights = graph_new_wrapper.view_edge_list(self)
            self.edgelist = self.EdgeList(src, dst, weights)
        if type(self) is Graph:
            edgelist_df = self.edgelist.edgelist_df[self.edgelist.edgelist_df[
                'src'] <= self.edgelist.edgelist_df['dst']].\
                reset_index(drop=True)
            self.edge_count = len(edgelist_df)
        else:
            edgelist_df = self.edgelist.edgelist_df

        if self.renumbered:
            if isinstance(self.edgelist.renumber_map, cudf.DataFrame):
                df = cudf.DataFrame()
                ncols = len(edgelist_df.columns) - 2
                unrnb_df = edgelist_df.merge(
                    self.edgelist.renumber_map,
                    left_on='src', right_on='id', how='left'
                ).drop(['id', 'src']).rename(columns={'0': 'src'}, copy=False)
                unrnb_df = unrnb_df.merge(
                    self.edgelist.renumber_map,
                    left_on='dst', right_on='id', how='left'
                ).drop(['id', 'dst']).rename(columns={'0': 'dst'}, copy=False)
                cols = unrnb_df.columns.to_list()
                df = unrnb_df[cols[ncols:]+cols[0:ncols]]
            else:
                df = cudf.DataFrame()
                for c in edgelist_df.columns:
                    if c in ['src', 'dst']:
                        df[c] = self.edgelist.renumber_map.\
                            iloc[edgelist_df[c]].reset_index(drop=True)
                    else:
                        df[c] = edgelist_df[c]
            return df
        else:
            return edgelist_df

    def delete_edge_list(self):
        """
        Delete the edge list.
        """
        # decrease reference count to free memory if the referenced objects are
        # no longer used.
        self.edgelist = None

    def from_cudf_adjlist(self, offset_col, index_col, value_col=None):
        """
        Initialize a graph from the adjacency list. It is an error to call this
        method on an initialized Graph object. The passed offset_col and
        index_col arguments wrap gdf_column objects that represent a graph
        using the adjacency list format.
        If value_col is None, an unweighted graph is created. If value_col is
        not None, a weighted graph is created.
        If copy is False, this function stores references to the passed objects
        pointed by offset_col and index_col. If copy is True, this funcion
        stores references to the deep-copies of the passed objects pointed by
        offset_col and index_col.
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
        >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> M = M.to_pandas()
        >>> M = scipy.sparse.coo_matrix((M['2'],(M['0'],M['1'])))
        >>> M = M.tocsr()
        >>> offsets = cudf.Series(M.indptr)
        >>> indices = cudf.Series(M.indices)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_adjlist(offsets, indices, None)

        """
        if self.edgelist is not None or self.adjlist is not None:
            raise Exception('Graph already has values')
        self.adjlist = Graph.AdjList(offset_col, index_col, value_col)

    def add_adj_list(self, offset_col, index_col, value_col=None):
        warnings.warn('add_adj_list will be deprecated in next release.\
 Use from_cudf_adjlist instead')
        self.from_cudf_adjlist(offset_col, index_col, value_col)

    def view_adj_list(self):
        """
        Display the adjacency list. Compute it if needed.

        Returns
        -------
        offset_col : cudf.Series
            This cudf.Series wraps a gdf_column of size V + 1 (V: number of
            vertices).
            The gdf column contains the offsets for the vertices in this graph.
            Offsets are in the range [0, E] (E: number of edges).
        index_col : cudf.Series
            This cudf.Series wraps a gdf_column of size E (E: number of edges).
            The gdf column contains the destination index for each edge.
            Destination indices are in the range [0, V) (V: number of
            vertices).
        value_col : cudf.Series or ``None``
            This pointer is ``None`` for unweighted graphs.
            For weighted graphs, this cudf.Series wraps a gdf_column of size E
            (E: number of edges).
            The gdf column contains the weight value for each edge.
            The expected type of the gdf_column element is floating point
            number.
        """
        if self.distributed:
            raise Exception("Not supported for distributed graph")
        if self.adjlist is None:
            if self.transposedadjlist is not None and type(self) is Graph:
                off, ind, vals = (self.transposedadjlist.offsets,
                                  self.transposedadjlist.indices,
                                  self.transposedadjlist.weights)
            else:
                off, ind, vals = graph_new_wrapper.view_adj_list(self)
            self.adjlist = self.AdjList(off, ind, vals)
        return self.adjlist.offsets, self.adjlist.indices, self.adjlist.weights

    def view_transposed_adj_list(self):
        """
        Display the transposed adjacency list. Compute it if needed.

        Returns
        -------
        offset_col : cudf.Series
            This cudf.Series wraps a gdf_column of size V + 1 (V: number of
            vertices).
            The gdf column contains the offsets for the vertices in this graph.
            Offsets are in the range [0, E] (E: number of edges).
        index_col : cudf.Series
            This cudf.Series wraps a gdf_column of size E (E: number of edges).
            The gdf column contains the destination index for each edge.
            Destination indices are in the range [0, V) (V: number of
            vertices).
        value_col : cudf.Series or ``None``
            This pointer is ``None`` for unweighted graphs.
            For weighted graphs, this cudf.Series wraps a gdf_column of size E
            (E: number of edges).
            The gdf column contains the weight value for each edge.
            The expected type of the gdf_column element is floating point
            number.

        """
        if self.distributed:
            raise Exception("Not supported for distributed graph")
        if self.transposedadjlist is None:
            if self.adjlist is not None and type(self) is Graph:
                off, ind, vals = (self.adjlist.offsets, self.adjlist.indices,
                                  self.adjlist.weights)
            else:
                off, ind, vals = graph_new_wrapper.\
                    view_transposed_adj_list(self)
            self.transposedadjlist = self.transposedAdjList(off, ind, vals)
        return (self.transposedadjlist.offsets,
                self.transposedadjlist.indices,
                self.transposedadjlist.weights)

    def delete_adj_list(self):
        """
        Delete the adjacency list.
        """
        self.adjlist = None

    def get_two_hop_neighbors(self):
        """
        Compute vertex pairs that are two hops apart. The resulting pairs are
        sorted before returning.

        Returns
        -------
        df : cudf.DataFrame
            df['first'] : cudf.Series
                the first vertex id of a pair.
            df['second'] : cudf.Series
                the second vertex id of a pair.

        """
        if self.distributed:
            raise Exception("Not supported for distributed graph")
        df = graph_new_wrapper.get_two_hop_neighbors(self)
        if self.renumbered is True:
            if isinstance(self.edgelist.renumber_map, cudf.DataFrame):
                n_cols = len(self.edgelist.renumber_map.columns) - 1
                unrenumbered_df_ = df.merge(self.edgelist.renumber_map,
                                            left_on='first', right_on='id',
                                            how='left').\
                    drop(['id', 'first'])
                unrenumbered_df = unrenumbered_df_.merge(self.edgelist.
                                                         renumber_map,
                                                         left_on='second',
                                                         right_on='id',
                                                         how='left').\
                    drop(['id', 'second'])
                unrenumbered_df.columns = ['first_' + str(i)
                                           for i in range(n_cols)]\
                    + ['second_' + str(i) for i in range(n_cols)]
                df = unrenumbered_df
            else:
                df['first'] = self.edgelist.renumber_map.\
                    iloc[df['first']].reset_index(drop=True)
                df['second'] = self.edgelist.renumber_map.\
                    iloc[df['second']].reset_index(drop=True)
        return df

    def number_of_vertices(self):
        """
        Get the number of nodes in the graph.

        """
        if self.node_count is None:
            if self.distributed:
                if self.edgelist is not None:
                    ddf = self.edgelist.edgelist_df[['src', 'dst']]
                    self.node_count = ddf.max().max().compute() + 1
                else:
                    raise Exception("Graph is Empty")
            elif self.adjlist is not None:
                self.node_count = len(self.adjlist.offsets)-1
            elif self.transposedadjlist is not None:
                self.node_count = len(self.transposedadjlist.offsets)-1
            elif self.edgelist is not None:
                df = self.edgelist.edgelist_df[['src', 'dst']]
                self.node_count = df.max().max() + 1
            else:
                raise Exception("Graph is Empty")
        return self.node_count

    def number_of_nodes(self):
        """
        An alias of number_of_vertices(). This function is added for NetworkX
        compatibility.

        """
        return self.number_of_vertices()

    def number_of_edges(self, directed_edges=False):
        """
        Get the number of edges in the graph.

        """
        if self.distributed:
            if self.edgelist is not None:
                return len(self.edgelist.edgelist_df)
            else:
                raise ValueError('Graph is Empty')
        if directed_edges and self.edgelist is not None:
            return len(self.edgelist.edgelist_df)
        if self.edge_count is None:
            if self.edgelist is not None:
                if type(self) is Graph:
                    self.edge_count = len(self.edgelist.edgelist_df[
                                          self.edgelist.edgelist_df['src']
                                          >= self.edgelist.edgelist_df['dst']]
                                          )
                else:
                    self.edge_count = len(self.edgelist.edgelist_df)
            elif self.adjlist is not None:
                self.edge_count = len(self.adjlist.indices)
            elif self.transposedadjlist is not None:
                self.edge_count = len(self.transposedadjlist.indices)
            else:
                raise ValueError('Graph is Empty')
        return self.edge_count

    def in_degree(self, vertex_subset=None):
        """
        Compute vertex in-degree. Vertex in-degree is the number of edges
        pointing into the vertex. By default, this method computes vertex
        degrees for the entire set of vertices. If vertex_subset is provided,
        this method optionally filters out all but those listed in
        vertex_subset.

        Parameters
        ----------
        vertex_subset : cudf.Series or iterable container, optional
            A container of vertices for displaying corresponding in-degree.
            If not set, degrees are computed for the entire set of vertices.

        Returns
        -------
        df : cudf.DataFrame
            GPU data frame of size N (the default) or the size of the given
            vertices (vertex_subset) containing the in_degree. The ordering is
            relative to the adjacency list, or that given by the specified
            vertex_subset.
            df['vertex'] : cudf.Series
                The vertex IDs (will be identical to vertex_subset if
                specified).
            df['degree'] : cudf.Series
                The computed in-degree of the corresponding vertex.

        Examples
        --------
        >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> sources = cudf.Series(M['0'])
        >>> destinations = cudf.Series(M['1'])
        >>> G = cugraph.Graph()
        >>> G.add_edge_list(sources, destinations, None)
        >>> df = G.in_degree([0,9,12])

        """
        return self._degree(vertex_subset, x=1)

    def out_degree(self, vertex_subset=None):
        """
        Compute vertex out-degree. Vertex out-degree is the number of edges
        pointing out from the vertex. By default, this method computes vertex
        degrees for the entire set of vertices. If vertex_subset is provided,
        this method optionally filters out all but those listed in
        vertex_subset.

        Parameters
        ----------
        vertex_subset : cudf.Series or iterable container, optional
            A container of vertices for displaying corresponding out-degree.
            If not set, degrees are computed for the entire set of vertices.

        Returns
        -------
        df : cudf.DataFrame
            GPU data frame of size N (the default) or the size of the given
            vertices (vertex_subset) containing the out_degree. The ordering is
            relative to the adjacency list, or that given by the specified
            vertex_subset.
            df['vertex'] : cudf.Series
                The vertex IDs (will be identical to vertex_subset if
                specified).
            df['degree'] : cudf.Series
                The computed out-degree of the corresponding vertex.

        Examples
        --------
        >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> sources = cudf.Series(M['0'])
        >>> destinations = cudf.Series(M['1'])
        >>> G = cugraph.Graph()
        >>> G.add_edge_list(sources, destinations, None)
        >>> df = G.out_degree([0,9,12])

        """
        if self.distributed:
            raise Exception("Not supported for distributed graph")
        return self._degree(vertex_subset, x=2)

    def degree(self, vertex_subset=None):
        """
        Compute vertex degree. By default, this method computes vertex
        degrees for the entire set of vertices. If vertex_subset is provided,
        this method optionally filters out all but those listed in
        vertex_subset.

        Parameters
        ----------
        vertex_subset : cudf.Series or iterable container, optional
            A container of vertices for displaying corresponding degree. If not
            set, degrees are computed for the entire set of vertices.

        Returns
        -------
        df : cudf.DataFrame
            GPU data frame of size N (the default) or the size of the given
            vertices (vertex_subset) containing the degree. The ordering is
            relative to the adjacency list, or that given by the specified
            vertex_subset.
            df['vertex'] : cudf.Series
                The vertex IDs (will be identical to vertex_subset if
                specified).
            df['degree'] : cudf.Series
                The computed degree of the corresponding vertex.

        Examples
        --------
        >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> sources = cudf.Series(M['0'])
        >>> destinations = cudf.Series(M['1'])
        >>> G = cugraph.Graph()
        >>> G.add_edge_list(sources, destinations, None)
        >>> df = G.degree([0,9,12])

        """
        if self.distributed:
            raise Exception("Not supported for distributed graph")
        return self._degree(vertex_subset)

    def degrees(self, vertex_subset=None):
        """
        Compute vertex in-degree and out-degree. By default, this method
        computes vertex degrees for the entire set of vertices. If
        vertex_subset is provided, this method optionally filters out all but
        those listed in vertex_subset.

        Parameters
        ----------
        vertex_subset : cudf.Series or iterable container, optional
            A container of vertices for displaying corresponding degree. If not
            set, degrees are computed for the entire set of vertices.

        Returns
        -------
        df : cudf.DataFrame
            df['vertex'] : cudf.Series
                The vertex IDs (will be identical to vertex_subset if
                specified).
            df['in_degree'] : cudf.Series
                The in-degree of the vertex.
            df['out_degree'] : cudf.Series
                The out-degree of the vertex.

        Examples
        --------
        >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> sources = cudf.Series(M['0'])
        >>> destinations = cudf.Series(M['1'])
        >>> G = cugraph.Graph()
        >>> G.add_edge_list(sources, destinations, None)
        >>> df = G.degrees([0,9,12])

        """
        if self.distributed:
            raise Exception("Not supported for distributed graph")
        vertex_col, in_degree_col, out_degree_col = graph_new_wrapper._degrees(
            self)

        df = cudf.DataFrame()
        if vertex_subset is None:
            if self.renumbered is True:
                df['vertex'] = self.edgelist.renumber_map.iloc[vertex_col]
            else:
                df['vertex'] = vertex_col
            df['in_degree'] = in_degree_col
            df['out_degree'] = out_degree_col
        else:
            df['vertex'] = cudf.Series(
                np.asarray(vertex_subset, dtype=np.int32))
            if self.renumbered is True:
                renumber_series = cudf.Series(self.edgelist.renumber_map.index,
                                              index=self.edgelist.renumber_map)
                vertices_renumbered = renumber_series.loc[vertex_subset]

                df['in_degree'] = cudf.Series(
                    np.asarray([in_degree_col[i] for i in vertices_renumbered],
                               dtype=np.int32))
                df['out_degree'] = cudf.Series(np.asarray([out_degree_col[i]
                                               for i in vertices_renumbered],
                                               dtype=np.int32))
            else:
                df['in_degree'] = cudf.Series(
                    np.asarray([in_degree_col[i] for i in vertex_subset],
                               dtype=np.int32))
                df['out_degree'] = cudf.Series(
                    np.asarray([out_degree_col[i] for i in vertex_subset],
                               dtype=np.int32))

        return df

    def _degree(self, vertex_subset, x=0):
        vertex_col, degree_col = graph_new_wrapper._degree(self, x)
        df = cudf.DataFrame()
        if vertex_subset is None:
            if self.renumbered is True:
                df['vertex'] = self.edgelist.renumber_map.iloc[vertex_col]
            else:
                df['vertex'] = vertex_col
            df['degree'] = degree_col
        else:
            df['vertex'] = cudf.Series(np.asarray(
                vertex_subset, dtype=np.int32
            ))
            if self.renumbered is True:
                renumber_series = cudf.Series(self.edgelist.renumber_map.index,
                                              index=self.edgelist.renumber_map)
                vertices_renumbered = renumber_series.loc[vertex_subset]
                df['degree'] = cudf.Series(np.asarray(
                    [degree_col[i] for i in vertices_renumbered],
                    dtype=np.int32
                ))
            else:
                df['degree'] = cudf.Series(np.asarray(
                    [degree_col[i] for i in vertex_subset], dtype=np.int32
                ))

        return df

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
        if self.distributed:
            raise Exception("Not supported for distributed graph")
        if type(self) is DiGraph:
            return self
        if type(self) is Graph:
            DiG = DiGraph()
            DiG.renumbered = self.renumbered
            DiG.edgelist = self.edgelist
            DiG.adjlist = self.adjlist
            DiG.transposedadjlist = self.transposedadjlist
            return DiG

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
        if self.distributed:
            raise Exception("Not supported for distributed graph")
        if type(self) is Graph:
            return self
        if type(self) is DiGraph:
            G = Graph()
            df = self.edgelist.edgelist_df
            G.renumbered = self.renumbered
            if self.edgelist.weights:
                source_col, dest_col, value_col = symmetrize(df['src'],
                                                             df['dst'],
                                                             df['weights'])
            else:
                source_col, dest_col = symmetrize(df['src'],
                                                  df['dst'])
                value_col = None
            G.edgelist = Graph.EdgeList(source_col, dest_col, value_col,
                                        self.edgelist.renumber_map)

            return G

    def is_directed(self):
        if type(self) is DiGraph:
            return True
        else:
            return False

    def has_node(self, n):
        """
        Returns True if the graph contains the node n.
        """
        if self.edgelist is None:
            raise Exception("Graph has no Edgelist.")
        if self.distributed:
            ddf = self.edgelist.edgelist_df[['src', 'dst']]
            return (ddf == n).any().any().compute()
        if self.renumbered:
            return (self.edgelist.renumber_map == n).any()
        else:
            df = self.edgelist.edgelist_df[['src', 'dst']]
            return (df == n).any().any()

    def has_edge(self, u, v):
        """
        Returns True if the graph contains the edge (u,v).
        """
        if self.edgelist is None:
            raise Exception("Graph has no Edgelist.")
        if self.renumbered:
            src = self.edgelist.renumber_map.index[self.edgelist.
                                                   renumber_map == u]
            dst = self.edgelist.renumber_map.index[self.edgelist.
                                                   renumber_map == v]
            if (len(src) and len(dst)) == 0:
                return False
            else:
                u = src[0]
                v = dst[0]
        df = self.edgelist.edgelist_df
        if self.distributed:
            return ((df['src'] == u) & (df['dst'] == v)).any().compute()
        return ((df['src'] == u) & (df['dst'] == v)).any()

    def edges(self):
        """
        Returns all the edges in the graph as a cudf.DataFrame containing
        sources and destinations. It does not return the edge weights.
        For viewing edges with weights use view_edge_list()
        """
        return self.view_edge_list()[['src', 'dst']]

    def nodes(self):
        """
        Returns all the nodes in the graph as a cudf.Series
        """
        if self.distributed:
            raise Exception("Not supported for distributed graph")
        if self.edgelist is None:
            raise Exception("Graph has no Edgelist.")
        df = self.edgelist.edgelist_df
        n = cudf.concat([df['src'], df['dst']]).unique()
        if self.renumbered:
            return self.edgelist.renumber_map.iloc[n]
        else:
            return n

    def neighbors(self, n):
        if self.edgelist is None:
            raise Exception("Graph has no Edgelist.")
        if self.distributed:
            ddf = self.edgelist.edgelist_df
            return ddf[ddf['src'] == n]['dst'].reset_index(drop=True)
        if self.renumbered:
            node = self.edgelist.renumber_map.index[self.edgelist.
                                                    renumber_map == n]
            if len(node) == 0:
                return cudf.Series(dtype='int')
            n = node[0]

        df = self.edgelist.edgelist_df
        neighbors = df[df['src'] == n]['dst'].reset_index(drop=True)
        if self.renumbered:
            return self.edgelist.renumber_map.iloc[neighbors]
        else:
            return neighbors


class DiGraph(Graph):
    def __init__(self, m_graph=None, edge_attr=None):
        super().__init__(m_graph=m_graph, edge_attr=edge_attr,
                         symmetrized=True)


class MultiGraph(Graph):
    def __init__(self, renumbered=True):
        super().__init__(multi=True)


class MultiDiGraph(Graph):
    def __init__(self, renumbered=True):
        super().__init__(symmetrized=True, multi=True)
