# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

from cugraph.structure import graph_primtypes_wrapper
from cugraph.structure.graph_primtypes_wrapper import Direction
from cugraph.structure.number_map import NumberMap
import cudf
import dask_cudf


class simpleDistributedGraphImpl:
    class EdgeList:
        def __init__(self, ddf):
            self.edgelist_df = ddf
            self.weights = False
            # FIXME: Edge Attribute not handled

    # class AdjList:
    # Not Supported

    # class transposedAdjList:
    # Not Supported

    class Properties:
        def __init__(self, properties):
            self.multi_edge = getattr(properties, 'multi_edge', False)
            self.directed = properties.directed
            self.renumbered = False
            self.store_transposed = False
            self.self_loop = None
            self.isolated_vertices = None
            self.node_count = None
            self.edge_count = None
            self.weighted = False

    def __init__(self, properties):
        # Structure
        self.edgelist = None
        self.renumber_map = None
        self.aggregate_segment_offsets = None
        self.properties = simpleDistributedGraphImpl.Properties(properties)
        self.source_columns = None
        self.destination_columns = None

    # Functions
    def __from_edgelist(
        self,
        input_ddf,
        source="source",
        destination="destination",
        edge_attr=None,
        renumber=True,
        store_transposed=False,
    ):
        if not isinstance(input_ddf, dask_cudf.DataFrame):
            raise TypeError("input should be a dask_cudf dataFrame")
        if self.properties.directed is False:
            raise TypeError("Undirected distributed graph not supported")

        s_col = source
        d_col = destination
        if not isinstance(s_col, list):
            s_col = [s_col]
        if not isinstance(d_col, list):
            d_col = [d_col]
        if not (
            set(s_col).issubset(set(input_ddf.columns))
            and set(d_col).issubset(set(input_ddf.columns))
        ):
            raise ValueError(
                "source column names and/or destination column "
                "names not found in input. Recheck the source "
                "and destination parameters"
            )
        ddf_columns = s_col + d_col
        if edge_attr is not None:
            if not (set([edge_attr]).issubset(set(input_ddf.columns))):
                raise ValueError(
                    "edge_attr column name not found in input."
                    "Recheck the edge_attr parameter")
            self.properties.weighted = True
            ddf_columns = ddf_columns + [edge_attr]
        input_ddf = input_ddf[ddf_columns]

        if edge_attr is not None:
            input_ddf = input_ddf.rename(columns={edge_attr: 'value'})

        #
        # Keep all of the original parameters so we can lazily
        # evaluate this function
        #

        # FIXME: Edge Attribute not handled
        self.properties.renumbered = renumber
        self.input_df = input_ddf
        self.source_columns = source
        self.destination_columns = destination

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
        df : cudf.DataFrame
            This cudf.DataFrame wraps source, destination and weight
            df[src] : cudf.Series
                contains the source index for each edge
            df[dst] : cudf.Series
                contains the destination index for each edge
            df[weight] : cusd.Series
                Column is only present for weighted Graph,
                then containing the weight value for each edge
        """
        if self.edgelist is None:
            raise RuntimeError("Graph has no Edgelist.")
        return self.edgelist.edgelist_df

    def delete_edge_list(self):
        """
        Delete the edge list.
        """
        self.edgelist = None

    def clear(self):
        """
        Empty this graph.
        """
        self.edgelist = None

    def number_of_vertices(self):
        """
        Get the number of nodes in the graph.
        """
        if self.properties.node_count is None:
            if self.edgelist is not None:
                ddf = self.edgelist.edgelist_df[["src", "dst"]]
                self.properties.node_count = ddf.max().max().compute() + 1
            else:
                raise RuntimeError("Graph is Empty")
        return self.properties.node_count

    def number_of_nodes(self):
        """
        An alias of number_of_vertices().
        """
        return self.number_of_vertices()

    def number_of_edges(self, directed_edges=False):
        """
        Get the number of edges in the graph.
        """
        if self.edgelist is not None:
            return len(self.edgelist.edgelist_df)
        else:
            raise RuntimeError("Graph is Empty")

    def in_degree(self, vertex_subset=None):
        """
        Compute vertex in-degree. Vertex in-degree is the number of edges
        pointing into the vertex. By default, this method computes vertex
        degrees for the entire set of vertices. If vertex_subset is provided,
        this method optionally filters out all but those listed in
        vertex_subset.

        Parameters
        ----------
        vertex_subset : cudf.Series or iterable container, opt. (default=None)
            A container of vertices for displaying corresponding in-degree.
            If not set, degrees are computed for the entire set of vertices.

        Returns
        -------
        df : cudf.DataFrame
            GPU DataFrame of size N (the default) or the size of the given
            vertices (vertex_subset) containing the in_degree. The ordering is
            relative to the adjacency list, or that given by the specified
            vertex_subset.
            df[vertex] : cudf.Series
                The vertex IDs (will be identical to vertex_subset if
                specified).
            df[degree] : cudf.Series
                The computed in-degree of the corresponding vertex.
        Examples
        --------
        >>> M = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
        ...                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(M, '0', '1')
        >>> df = G.in_degree([0,9,12])

        """
        return self._degree(vertex_subset, direction=Direction.IN)

    def out_degree(self, vertex_subset=None):
        """
        Compute vertex out-degree. Vertex out-degree is the number of edges
        pointing out from the vertex. By default, this method computes vertex
        degrees for the entire set of vertices. If vertex_subset is provided,
        this method optionally filters out all but those listed in
        vertex_subset.

        Parameters
        ----------
        vertex_subset : cudf.Series or iterable container, opt. (default=None)
            A container of vertices for displaying corresponding out-degree.
            If not set, degrees are computed for the entire set of vertices.

        Returns
        -------
        df : cudf.DataFrame
            GPU DataFrame of size N (the default) or the size of the given
            vertices (vertex_subset) containing the out_degree. The ordering is
            relative to the adjacency list, or that given by the specified
            vertex_subset.
            df[vertex] : cudf.Series
                The vertex IDs (will be identical to vertex_subset if
                specified).
            df[degree] : cudf.Series
                The computed out-degree of the corresponding vertex.
        Examples
        --------
        >>> M = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
        ...                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(M, '0', '1')
        >>> df = G.out_degree([0,9,12])

        """
        return self._degree(vertex_subset, direction=Direction.OUT)

    def degree(self, vertex_subset=None):
        """
        Compute vertex degree, which is the total number of edges incident
        to a vertex (both in and out edges). By default, this method computes
        degrees for the entire set of vertices. If vertex_subset is provided,
        then this method optionally filters out all but those listed in
        vertex_subset.
        Parameters
        ----------
        vertex_subset : cudf.Series or iterable container, optional
            a container of vertices for displaying corresponding degree. If not
            set, degrees are computed for the entire set of vertices.
        Returns
        -------
        df : cudf.DataFrame
            GPU DataFrame of size N (the default) or the size of the given
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
        >>> M = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
        ...                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(M, '0', '1')
        >>> all_df = G.degree()
        >>> subset_df = G.degree([0,9,12])

        """
        raise NotImplementedError("Not supported for distributed graph")

    # FIXME:  vertex_subset could be a DataFrame for multi-column vertices
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
            GPU DataFrame of size N (the default) or the size of the given
            vertices (vertex_subset) containing the degrees. The ordering is
            relative to the adjacency list, or that given by the specified
            vertex_subset.
            df['vertex'] : cudf.Series
                The vertex IDs (will be identical to vertex_subset if
                specified).
            df['in_degree'] : cudf.Series
                The in-degree of the vertex.
            df['out_degree'] : cudf.Series
                The out-degree of the vertex.

        Examples
        --------
        >>> M = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
        ...                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(M, '0', '1')
        >>> df = G.degrees([0,9,12])

        """
        raise NotImplementedError("Not supported for distributed graph")

    def _degree(self, vertex_subset, direction=Direction.ALL):
        vertex_col, degree_col = graph_primtypes_wrapper._mg_degree(self,
                                                                    direction)
        df = cudf.DataFrame()
        df["vertex"] = vertex_col
        df["degree"] = degree_col

        if self.properties.renumbered is True:
            df = self.renumber_map.unrenumber(df, "vertex")

        if vertex_subset is not None:
            df = df[df['vertex'].isin(vertex_subset)]

        return df

    def to_directed(self, DiG):
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
        >>> M = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
        ...                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(M, '0', '1')
        >>> DiG = G.to_directed()

        """
        # TODO: Add support
        raise NotImplementedError("Not supported for distributed graph")

    def to_undirected(self, G):
        """
        Return an undirected copy of the graph.

        Returns
        -------
        G : Graph
            A undirected graph with the same nodes, and each directed edge
            (u,v,weights) replaced by an undirected edge (u,v,weights).

        Examples
        --------
        >>> M = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
        ...         dtype=['int32', 'int32', 'float32'], header=None)
        >>> DiG = cugraph.Graph(directed=True)
        >>> DiG.from_cudf_edgelist(M, '0', '1')
        >>> G = DiG.to_undirected()
        """

        # TODO: Add support
        raise NotImplementedError("Not supported for distributed graph")

    def has_node(self, n):
        """
        Returns True if the graph contains the node n.
        """
        if self.edgelist is None:
            raise RuntimeError("Graph has no Edgelist.")
        # FIXME: Check renumber map
        ddf = self.edgelist.edgelist_df[["src", "dst"]]
        return (ddf == n).any().any().compute()

    def has_edge(self, u, v):
        """
        Returns True if the graph contains the edge (u,v).
        """
        # TODO: Verify Correctness
        if self.properties.renumbered:
            src_col_name = self.renumber_map.renumbered_src_col_name

            tmp = cudf.DataFrame({src_col_name: [u, v]})
            tmp = tmp.astype({src_col_name: "int"})
            tmp = self.add_internal_vertex_id(
                tmp, "id", src_col_name, preserve_order=True
            )

            u = tmp["id"][0]
            v = tmp["id"][1]

        df = self.edgelist.edgelist_df
        return ((df["src"] == u) & (df["dst"] == v)).any().compute()

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
        # FIXME: Return renumber map nodes
        raise NotImplementedError("Not supported for distributed graph")

    def neighbors(self, n):
        if self.edgelist is None:
            raise RuntimeError("Graph has no Edgelist.")
        # FIXME: Add renumbering of node n
        ddf = self.edgelist.edgelist_df
        return ddf[ddf["src"] == n]["dst"].reset_index(drop=True)

    def compute_renumber_edge_list(self, transposed=False):
        """
        Compute a renumbered edge list
        This function works in the MNMG pipeline and will transform
        the input dask_cudf.DataFrame into a renumbered edge list
        in the prescribed direction.
        This function will be called by the algorithms to ensure
        that the graph is renumbered properly.  The graph object will
        cache the most recent renumbering attempt.  For benchmarking
        purposes, this function can be called prior to calling a
        graph algorithm so we can measure the cost of computing
        the renumbering separately from the cost of executing the
        algorithm.
        When creating a CSR-like structure, set transposed to False.
        When creating a CSC-like structure, set transposed to True.

        Parameters
        ----------
        transposed : (optional) bool
            If True, renumber with the intent to make a CSC-like
            structure.  If False, renumber with the intent to make
            a CSR-like structure.  Defaults to False.
        """
        # FIXME:  What to do about edge_attr???
        #         currently ignored for MNMG

        # FIXME: this is confusing - in the code below,
        # self.properties.renumbered needs to be interpreted as "needs to be
        # renumbered", everywhere else it means "has been renumbered".
        if not self.properties.renumbered:
            self.edgelist = self.EdgeList(self.input_df)
            self.renumber_map = None
        else:
            if self.edgelist is not None:
                if self.properties.directed is False:
                    return

                if self.properties.store_transposed == transposed:
                    return

                del self.edgelist

            renumbered_ddf, number_map, aggregate_segment_offsets = \
                NumberMap.renumber_and_segment(self.input_df,
                                               self.source_columns,
                                               self.destination_columns,
                                               store_transposed=transposed)
            self.edgelist = self.EdgeList(renumbered_ddf)
            self.renumber_map = number_map
            self.aggregate_segment_offsets = aggregate_segment_offsets
            self.properties.store_transposed = transposed

    def vertex_column_size(self):
        if self.properties.renumbered:
            return self.renumber_map.vertex_column_size()
        else:
            return 1
