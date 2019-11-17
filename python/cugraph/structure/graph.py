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

from cugraph.structure import graph_wrapper
from cugraph.structure.symmetrize import symmetrize
from cugraph.structure.renumber import renumber as rnb
import cudf
import numpy as np


def null_check(col):
    if col.null_count != 0:
        raise ValueError('Series contains NULL values')


class Graph:

    class EdgeList:
        def __init__(self, source, dest, value=None, renumber_map=None):
            self.renumber_map = renumber_map
            df = cudf.DataFrame()
            df['src'] = source
            df['dst'] = dest
            self.weights = False
            if value is not None:
                self.weights = True
                df['weights'] = value
            self.edgelist_df = df

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
    def __init__(self, symmetrized=False, bipartite=False, multi=False,
                 dynamic=False):
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
        # self.number_of_vertices = None

    def clear(self):
        """
        Empty this graph. This function is added for NetworkX compatibility.
        """
        self.edgelist = None
        self.adjlist = None
        self.transposedadjlist = None

    def add_edge_list(self, input_df, source='source', target='target',
                      edge_attr=None, renumber=False):
        """
        Initialize a graph from the edge list. It is an error to call this
        method on an initialized Graph object. The passed source_col and
        dest_col arguments wrap gdf_column objects that represent a graph
        using the edge list format.
        Source and destination indices must be in the range [0, V) where V is
        the number of vertices. They must be 32 bit integers. Please refer to
        cuGraph's renumbering feature if your input does not match these
        requierments. When using cudf.read_csv to load a CSV edge list,
        make sure to set dtype to int32 for the source and destination
        columns.
        If value_col is None, an unweighted graph is created. If value_col is
        not None, a weighted graph is created.
        If copy is False, this function stores references to the passed objects
        pointed by source_col and dest_col. If copy is True, this funcion
        stores references to the deep-copies of the passed objects pointed by
        source_col and dest_col.
        Undirected edges must be stored as directed edges in both directions.

        Parameters
        ----------
        source_col : cudf.Series
            This cudf.Series wraps a gdf_column of size E (E: number of edges).
            The gdf column contains the source index for each edge.
            Source indices must be in the range [0, V) (V: number of vertices).
            Source indices must be 32 bit integers.
        dest_col : cudf.Series
            This cudf.Series wraps a gdf_column of size E (E: number of edges).
            The gdf column contains the destination index for each edge.
            Destination indices must be in the range [0, V) (V: number of
            vertices).
            Destination indices must be 32 bit integers.
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
        >>> sources = cudf.Series(M['0'])
        >>> destinations = cudf.Series(M['1'])
        >>> G = cugraph.Graph()
        >>> G.add_edge_list(sources, destinations, None)
        """

        if self.edgelist is not None or self.adjlist is not None:
            raise Exception('Graph already has values')
        source_col = input_df[input_df.columns[0]]
        dest_col = input_df[input_df.columns[1]]
        if len(input_df.columns) > 2:
            value_col = input_df[input_df.columns[2]]
        else:
            value_col = None
        renumber_map = None
        if renumber:
            source_col, dest_col,
            renumber_map = rnb(input_df[input_df.columns[0]],
                               input_df[input_df.columns[1]])
            self.renumbered = True
        if not self.symmetrized:
            if value_col is not None:
                source_col, dest_col,
                value_col = symmetrize(source_col,
                                       dest_col,
                                       input_df[input_df.columns[2]])
            else:
                source_col, dest_col = symmetrize(source_col, dest_col)

        self.edgelist = Graph.EdgeList(source_col, dest_col, value_col,
                                       renumber_map)

    def view_edge_list(self):
        """
        Display the edge list. Compute it if needed.

        Returns
        -------
        source_col : cudf.Series
            This cudf.Series wraps a gdf_column of size E (E: number of edges).
            The gdf column contains the source index for each edge.
            Source indices are in the range [0, V) (V: number of vertices).
            Source indices must be 32 bit integers.
        dest_col : cudf.Series
            This cudf.Series wraps a gdf_column of size E (E: number of edges).
            The gdf column contains the destination index for each edge.
            Destination indices are in the range [0, V) (V: number of
            vertices).
            Destination indices must be 32 bit integers.
        value_col : cudf.Series or ``None``
            This pointer is ``None`` for unweighted graphs.
            For weighted graphs, this cudf.Series wraps a gdf_column of size E
            (E: number of edges).
            The gdf column contains the weight value for each edge.
            The expected type of the gdf_column element is floating point
            number.
        """
        if self.edgelist is None:
            graph_wrapper.view_edge_list(self)
        return self.edgelist.edgelist_df

    def delete_edge_list(self):
        """
        Delete the edge list.
        """
        # decrease reference count to free memory if the referenced objects are
        # no longer used.
        self.edgelist = None

    def add_adj_list(self, offset_col, index_col, value_col=None):
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
        >>> G.add_adj_list(offsets, indices, None)
        """
        if self.edgelist is not None or self.adjlist is not None:
            raise Exception('Graph already has values')
        self.adjlist = Graph.AdjList(offset_col, index_col, value_col)

    def view_adj_list(self):
        if self.adjlist is None:
            graph_wrapper.view_adj_list(self)
        return self.adjlist.offsets, self.adjlist.indices, self.adjlist.weights

    def delete_adj_list(self):
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
        df = graph_wrapper.get_two_hop_neighbors(self)

        return df

    def number_of_vertices(self):
        if self.adjlist is not None:
            num_vertices = len(self.adjlist.offsets)-1
        elif self.transposedadjlist is not None:
            num_vertices = len(self.transposedadjlist.offsets)-1
        else:
            num_vertices = graph_wrapper.number_of_vertices(self)
        return num_vertices

    def number_of_nodes(self):
        """
        An alias of number_of_vertices(). This function is added for NetworkX
        compatibility.
        """
        return self.number_of_vertices()

    def number_of_edges(self):
        """
        Get the number of edges in the graph.
        """
        if self.edgelist is not None:
            return len(self.edgelist.edgelist_df)
        elif self.adjlist is not None:
            return len(self.adjlist.indices)
        elif self.transposedadjlist is not None:
            return len(self.transposedadjlist.indices)
        else:
            raise ValueError('Graph is Empty')

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
        vertex_col, in_degree_col, out_degree_col = graph_wrapper._degrees(
                                                        self)

        df = cudf.DataFrame()
        if vertex_subset is None:
            df['vertex'] = vertex_col
            df['in_degree'] = in_degree_col
            df['out_degree'] = out_degree_col
        else:
            df['vertex'] = cudf.Series(
                np.asarray(vertex_subset, dtype=np.int32))
            df['in_degree'] = cudf.Series(
                np.asarray([in_degree_col[i] for i in vertex_subset],
                           dtype=np.int32))
            df['out_degree'] = cudf.Series(
                np.asarray([out_degree_col[i] for i in vertex_subset],
                           dtype=np.int32))
            # is this necessary???
            del vertex_col
            del in_degree_col
            del out_degree_col

        return df

    def _degree(self, vertex_subset, x=0):
        vertex_col, degree_col = graph_wrapper._degree(self, x)

        df = cudf.DataFrame()
        if vertex_subset is None:
            df['vertex'] = vertex_col
            df['degree'] = degree_col
        else:
            df['vertex'] = cudf.Series(np.asarray(
                vertex_subset, dtype=np.int32
            ))
            df['degree'] = cudf.Series(np.asarray(
                [degree_col[i] for i in vertex_subset], dtype=np.int32
            ))
            # is this necessary???
            del vertex_col
            del degree_col

        return df


class DiGraph(Graph):
    def __init__(self):
        super().__init__(symmetrized=True)


class MultiGraph(Graph):
    def __init__(self, renumbered=True):
        super().__init__(multi=True)


class DiMultiGraph(Graph):
    def __init__(self, renumbered=True):
        super().__init__(symmetrized=True, multi=True)
