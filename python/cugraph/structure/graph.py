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
from cugraph.structure.number_map import NumberMap
import cudf
import warnings


def null_check(col):
    if col.null_count != 0:
        raise ValueError("Series contains NULL values")


class Graph:
    class EdgeList:
        def __init__(self, *args):
            if len(args) == 1:
                self.__from_dask_cudf(*args)
            else:
                self.__from_cudf(*args)

        def __from_cudf(
            self, source, destination, edge_attr=None, renumber_map=None
        ):
            self.renumber_map = renumber_map
            self.edgelist_df = cudf.DataFrame()
            self.edgelist_df["src"] = source
            self.edgelist_df["dst"] = destination
            self.weights = False
            if edge_attr is not None:
                self.weights = True
                if type(edge_attr) is dict:
                    for k in edge_attr.keys():
                        self.edgelist_df[k] = edge_attr[k]
                else:
                    self.edgelist_df["weights"] = edge_attr

        def __from_dask_cudf(self, ddf):
            self.renumber_map = None
            self.edgelist_df = ddf
            self.weights = False

    class AdjList:
        def __init__(self, offsets, indices, value=None):
            self.offsets = offsets
            self.indices = indices
            self.weights = value  # Should de a daftaframe for multiple weights

    class transposedAdjList:
        def __init__(self, offsets, indices, value=None):
            Graph.AdjList.__init__(self, offsets, indices, value)

    """
    cuGraph graph class containing basic graph creation and transformation
    operations.
    """

    def __init__(
        self,
        m_graph=None,
        edge_attr=None,
        symmetrized=False,
        bipartite=False,
        multi=False,
        dynamic=False,
    ):
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
        self.dynamic = dynamic
        self.edgelist = None
        self.adjlist = None
        self.transposedadjlist = None
        self.edge_count = None
        self.node_count = None
        if m_graph is not None:
            if (type(self) is Graph and type(m_graph) is MultiGraph) or (
                type(self) is DiGraph and type(m_graph) is MultiDiGraph
            ):
                self.from_cudf_edgelist(
                    m_graph.edgelist.edgelist_df,
                    source="src",
                    destination="dst",
                    edge_attr=edge_attr,
                )
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

    def from_cudf_edgelist(
        self,
        input_df,
        source="source",
        destination="destination",
        edge_attr=None,
        renumber=True,
    ):
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
        input_df : cudf.DataFrame
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
            raise Exception("Graph already has values")

        source_col = None
        dest_col = None
        value_col = None
        renumber_map = None

        if self.multi:
            if type(edge_attr) is not list:
                raise Exception("edge_attr should be a list of column names")
            value_col = {}
            for col_name in edge_attr:
                value_col[col_name] = input_df[col_name]
        elif edge_attr is not None:
            value_col = input_df[edge_attr]
        else:
            value_col = None

        if renumber:
            renumber_map = NumberMap()
            if type(source) is list:
                renumber_map.from_dataframe(input_df, source, destination)
                source_col = renumber_map.to_vertex_id(input_df, source)
                dest_col = renumber_map.to_vertex_id(input_df, destination)
            else:
                renumber_map.from_dataframe(input_df, [source], [destination])
                source_col = renumber_map.to_vertex_id(input_df, [source])
                dest_col = renumber_map.to_vertex_id(input_df, [destination])

            self.renumbered = True
        else:
            if type(source) is list and type(destination) is list:
                raise Exception("set renumber to True for multi column ids")
            else:
                source_col = input_df[source]
                dest_col = input_df[destination]

        if not self.symmetrized and not self.multi:
            if value_col is not None:
                source_col, dest_col, value_col = symmetrize(
                    source_col, dest_col, value_col
                )
            else:
                source_col, dest_col = symmetrize(source_col, dest_col)

        self.edgelist = Graph.EdgeList(
            source_col, dest_col, value_col, renumber_map
        )

    def add_edge_list(self, source, destination, value=None):
        warnings.warn(
            "add_edge_list will be deprecated in next release.\
 Use from_cudf_edgelist instead"
        )
        input_df = cudf.DataFrame()
        input_df["source"] = source
        input_df["destination"] = destination
        if value is not None:
            input_df["weights"] = value
            self.from_cudf_edgelist(input_df, edge_attr="weights")
        else:
            self.from_cudf_edgelist(input_df)

    def from_dask_cudf_edgelist(self, input_ddf):
        self.edgelist = self.EdgeList(input_ddf)

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
        if self.edgelist is None:
            src, dst, weights = graph_new_wrapper.view_edge_list(self)
            self.edgelist = self.EdgeList(src, dst, weights)

        edgelist_df = self.edgelist.edgelist_df

        if self.renumbered:
            src_name_mapping = {"0": "src"}
            dst_name_mapping = {"0": "dst"}
            if len(self.edgelist.renumber_map.column_names()) > 1:
                for name in self.edgelist.renumber_map.column_names():
                    src_name_mapping[name] = name + "_src"
                    dst_name_mapping[name] = name + "_dst"

            edgelist_df = self.edgelist.renumber_map.from_vertex_id(
                edgelist_df, "src", drop=True
            ).rename(src_name_mapping)
            edgelist_df = self.edgelist.renumber_map.from_vertex_id(
                edgelist_df, "dst", drop=True
            ).rename(dst_name_mapping)

        if type(self) is Graph:
            edgelist_df = edgelist_df[edgelist_df["src"] <= edgelist_df["dst"]]
            edgelist_df = edgelist_df.reset_index(drop=True)
            self.edge_count = len(edgelist_df)

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
            raise Exception("Graph already has values")
        self.adjlist = Graph.AdjList(offset_col, index_col, value_col)

    def add_adj_list(self, offset_col, index_col, value_col=None):
        warnings.warn(
            "add_adj_list will be deprecated in next release.\
 Use from_cudf_adjlist instead"
        )
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
        if self.adjlist is None:
            if self.transposedadjlist is not None and type(self) is Graph:
                off, ind, vals = (
                    self.transposedadjlist.offsets,
                    self.transposedadjlist.indices,
                    self.transposedadjlist.weights,
                )
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
        if self.transposedadjlist is None:
            if self.adjlist is not None and type(self) is Graph:
                off, ind, vals = (
                    self.adjlist.offsets,
                    self.adjlist.indices,
                    self.adjlist.weights,
                )
            else:
                off, ind, vals = graph_new_wrapper.view_transposed_adj_list(
                    self
                )
            self.transposedadjlist = self.transposedAdjList(off, ind, vals)
        return (
            self.transposedadjlist.offsets,
            self.transposedadjlist.indices,
            self.transposedadjlist.weights,
        )

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
                the first vertex id of a pair, if an external vertex id
                is defined by only one column
            df['second'] : cudf.Series
                the second vertex id of a pair, if an external vertex id
                is defined by only one column
            df['*_first'] : cudf.Series
                the first vertex id of a pair, column 0 of the external
                vertex id will be represented as '0_first', column 1 as
                '1_first', etc.
            df['*_second'] : cudf.Series
                the second vertex id of a pair, column 0 of the external
                vertex id will be represented as '0_first', column 1 as
                '1_first', etc.
        """
        df = graph_new_wrapper.get_two_hop_neighbors(self)
        if self.renumbered is True:
            first_name_mapping = {"0": "first"}
            second_name_mapping = {"0": "second"}
            if len(self.edgelist.renumber_map.column_names()) > 1:
                for name in self.edgelist.renumber_map.column_names():
                    first_name_mapping[name] = name + "_first"
                    second_name_mapping[name] = name + "_second"

            df = (
                self.edgelist.renumber_map.from_vertex_id(df, "first")
                .drop("first")
                .rename(first_name_mapping)
            )
            df = (
                self.edgelist.renumber_map.from_vertex_id(df, "second")
                .drop("second")
                .rename(second_name_mapping)
            )

        return df

    def number_of_vertices(self):
        """
        Get the number of nodes in the graph.

        """
        if self.node_count is None:
            if self.adjlist is not None:
                self.node_count = len(self.adjlist.offsets) - 1
            elif self.transposedadjlist is not None:
                self.node_count = len(self.transposedadjlist.offsets) - 1
            elif self.edgelist is not None:
                df = self.edgelist.edgelist_df[["src", "dst"]]
                self.node_count = df.max().max() + 1
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
        if directed_edges and self.edgelist is not None:
            return len(self.edgelist.edgelist_df)
        if self.edge_count is None:
            if self.edgelist is not None:
                if type(self) is Graph:
                    self.edge_count = len(
                        self.edgelist.edgelist_df[
                            self.edgelist.edgelist_df["src"]
                            >= self.edgelist.edgelist_df["dst"]
                        ]
                    )
                else:
                    self.edge_count = len(self.edgelist.edgelist_df)
            elif self.adjlist is not None:
                self.edge_count = len(self.adjlist.indices)
            elif self.transposedadjlist is not None:
                self.edge_count = len(self.transposedadjlist.indices)
            else:
                raise ValueError("Graph is Empty")
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
        return self._degree(vertex_subset)

    # FIXME:  vertex_subset could be a data frame for multi-column vertices
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
        vertex_col, in_degree_col, out_degree_col = graph_new_wrapper._degrees(
            self
        )

        df = cudf.DataFrame()
        df["vertex"] = vertex_col
        df["in_degree"] = in_degree_col
        df["out_degree"] = out_degree_col

        if self.renumbered is True:
            name_mapping = {"0": "vertex"}
            if len(self.edgelist.renumber_map.column_names()) > 1:
                for name in self.edgelist.renumber_map.column_names():
                    name_mapping[name] = name + "_vertex"

            df = (
                self.edgelist.renumber_map.from_vertex_id(df, "vertex")
                .drop("vertex")
                .rename(name_mapping)
            )

        if vertex_subset is not None:
            df = df.query("`vertex` in @vertex_subset")

        return df

    def _degree(self, vertex_subset, x=0):
        vertex_col, degree_col = graph_new_wrapper._degree(self, x)

        df = cudf.DataFrame()
        df["vertex"] = vertex_col
        df["degree"] = degree_col

        if self.renumbered is True:
            name_mapping = {"0": "vertex"}
            if len(self.edgelist.renumber_map.column_names()) > 1:
                for name in self.edgelist.renumber_map.column_names():
                    name_mapping[name] = name + "_vertex"

            df = (
                self.edgelist.renumber_map.from_vertex_id(df, "vertex")
                .drop("vertex")
                .rename(name_mapping)
            )

        if vertex_subset is not None:
            df = df.query("`vertex` in @vertex_subset")

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

        if type(self) is Graph:
            return self
        if type(self) is DiGraph:
            G = Graph()
            df = self.edgelist.edgelist_df
            G.renumbered = self.renumbered
            if self.edgelist.weights:
                source_col, dest_col, value_col = symmetrize(
                    df["src"], df["dst"], df["weights"]
                )
            else:
                source_col, dest_col = symmetrize(df["src"], df["dst"])
                value_col = None
            G.edgelist = Graph.EdgeList(
                source_col, dest_col, value_col, self.edgelist.renumber_map
            )

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

        if self.renumbered:
            tmp = self.edgelist.renumber_map.to_vertex_id(cudf.Series([n]))
            return tmp[0] >= 0
        else:
            df = self.edgelist.edgelist_df[["src", "dst"]]
            return (df == n).any().any()

    def has_edge(self, u, v):
        """
        Returns True if the graph contains the edge (u,v).
        """

        if self.renumbered:
            tmp = self.edgelist.renumber_map.to_vertex_id(cudf.Series([u, v]))

            u = tmp[0]
            v = tmp[1]

        df = self.edgelist.edgelist_df
        return ((df["src"] == u) & (df["dst"] == v)).any()

    def edges(self):
        """
        Returns all the edges in the graph as a cudf.DataFrame containing
        sources and destinations. It does not return the edge weights.
        For viewing edges with weights use view_edge_list()
        """
        return self.view_edge_list()[["src", "dst"]]

    def nodes(self):
        """
        Returns all the nodes in the graph as a cudf.Series
        """
        df = self.edgelist.edgelist_df
        if self.renumbered:
            # FIXME: If vertices are multicolumn
            #        this needs to return a dataframe
            # FIXME: This relies un current implementation
            #        of NumberMap, should not really expose
            #        this, perhaps add a method to NumberMap
            return self.edgelist.renumber_map.df["0"]
        else:
            return cudf.concat([df["src"], df["dst"]]).unique()

    def neighbors(self, n):

        if self.renumbered:
            node = self.edgelist.renumber_map.to_vertex_id(cudf.Series([n]))
            if len(node) == 0:
                return cudf.Series(dtype="int")
            n = node[0]

        df = self.edgelist.edgelist_df
        neighbors = df[df["src"] == n]["dst"].reset_index(drop=True)
        if self.renumbered:
            # FIXME:  Multi-column vertices
            return self.edgelist.renumber_map.from_vertex_id(neighbors)["0"]
        else:
            return neighbors


class DiGraph(Graph):
    def __init__(self, m_graph=None, edge_attr=None):
        super().__init__(
            m_graph=m_graph, edge_attr=edge_attr, symmetrized=True
        )


class MultiGraph(Graph):
    def __init__(self, renumbered=True):
        super().__init__(multi=True)


class MultiDiGraph(Graph):
    def __init__(self, renumbered=True):
        super().__init__(symmetrized=True, multi=True)
