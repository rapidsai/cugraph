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
from cugraph.structure.symmetrize import symmetrize
from cugraph.structure.number_map import NumberMap
import cugraph.dask.common.mg_utils as mg_utils
import cudf
import dask_cudf
import cugraph.comms.comms as Comms
import pandas as pd
import numpy as np
from cugraph.dask.structure import replication


# FIXME: Change to consistent camel case naming
class simpleGraphImpl:

    class EdgeList:
        def __init__(self, source, destination, edge_attr=None):
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

    class AdjList:
        def __init__(self, offsets, indices, value=None):
            self.offsets = offsets
            self.indices = indices
            self.weights = value  # Should be a dataframe for multiple weights

    class transposedAdjList:
        def __init__(self, offsets, indices, value=None):
            simpleGraphImpl.AdjList.__init__(self, offsets, indices, value)

    class Properties:
        def __init__(self, properties):
            self.multi_edge = getattr(properties, 'multi_edge', False)
            self.directed = properties.directed
            self.renumbered = False
            self.self_loop = None
            self.isolated_vertices = None
            self.node_count = None
            self.edge_count = None
            self.weighted = False

    def __init__(self, properties):
        # Structure
        self.edgelist = None
        self.adjlist = None
        self.transposedadjlist = None
        self.renumber_map = None
        self.properties = simpleGraphImpl.Properties(properties)
        self._nodes = {}

        # TODO: Move to new batch class
        # MG - Batch
        self.batch_enabled = False
        self.batch_edgelists = None
        self.batch_adjlists = None
        self.batch_transposed_adjlists = None

    # Functions
    # FIXME: Change to public function
    # FIXME: Make function more modular
    def __from_edgelist(
        self,
        input_df,
        source="source",
        destination="destination",
        edge_attr=None,
        renumber=True,
    ):

        # Verify column names present in input DataFrame
        s_col = source
        d_col = destination
        if not isinstance(s_col, list):
            s_col = [s_col]
        if not isinstance(d_col, list):
            d_col = [d_col]
        if not (
            set(s_col).issubset(set(input_df.columns))
            and set(d_col).issubset(set(input_df.columns))
        ):
            raise ValueError(
                "source column names and/or destination column "
                "names not found in input. Recheck the source and "
                "destination parameters"
            )

        # FIXME: check if the consolidated graph fits on the
        # device before gathering all the edge lists

        # Consolidation
        if isinstance(input_df, cudf.DataFrame):
            if len(input_df[source]) > 2147483100:
                raise ValueError(
                    "cudf dataFrame edge list is too big "
                    "to fit in a single GPU"
                )
            elist = input_df
        elif isinstance(input_df, dask_cudf.DataFrame):
            if len(input_df[source]) > 2147483100:
                raise ValueError(
                    "dask_cudf dataFrame edge list is too big "
                    "to fit in a single GPU"
                )
            elist = input_df.compute().reset_index(drop=True)
        else:
            raise TypeError(
                "input should be a cudf.DataFrame or "
                "a dask_cudf dataFrame"
            )

        # Renumbering
        self.renumber_map = None
        if renumber:
            # FIXME: Should SG do lazy evaluation like MG?
            elist, renumber_map = NumberMap.renumber(
                elist, source, destination, store_transposed=False
            )
            source = renumber_map.renumbered_src_col_name
            destination = renumber_map.renumbered_dst_col_name
            self.properties.renumbered = True
            self.renumber_map = renumber_map
        else:
            if type(source) is list and type(destination) is list:
                raise ValueError("set renumber to True for multi column ids")

        # Populate graph edgelist
        source_col = elist[source]
        dest_col = elist[destination]

        if edge_attr is not None:
            self.properties.weighted = True
            value_col = elist[edge_attr]
        else:
            value_col = None

        # TODO: Update Symmetrize to work on Graph and/or DataFrame
        if value_col is not None:
            source_col, dest_col, value_col = symmetrize(
                source_col, dest_col, value_col,
                multi=self.properties.multi_edge,
                symmetrize=not self.properties.directed)
            if isinstance(value_col, cudf.DataFrame):
                value_dict = {}
                for i in value_col.columns:
                    value_dict[i] = value_col[i]
                value_col = value_dict
        else:
            source_col, dest_col = symmetrize(
                source_col, dest_col, multi=self.properties.multi_edge,
                symmetrize=not self.properties.directed)

        self.edgelist = simpleGraphImpl.EdgeList(source_col, dest_col,
                                                 value_col)

        if self.batch_enabled:
            self._replicate_edgelist()

    def to_pandas_edgelist(self, source='source', destination='destination'):
        """
        Returns the graph edge list as a Pandas DataFrame.

        Parameters
        ----------
        source : str or array-like, optional (default='source')
            source column name or array of column names
        destination : str or array-like, optional (default='destination')
            destination column name or array of column names

        Returns
        -------
        df : pandas.DataFrame
        """

        gdf = self.view_edge_list()
        return gdf.to_pandas()

    def to_pandas_adjacency(self):
        """
        Returns the graph adjacency matrix as a Pandas DataFrame.
        """

        np_array_data = self.to_numpy_array()
        pdf = pd.DataFrame(np_array_data)
        if self.properties.renumbered:
            nodes = self.renumber_map.implementation.df['0'].\
                    values_host.tolist()
        pdf.columns = nodes
        pdf.index = nodes
        return pdf

    def to_numpy_array(self):
        """
        Returns the graph adjacency matrix as a NumPy array.
        """

        nlen = self.number_of_nodes()
        elen = self.number_of_edges()
        df = self.edgelist.edgelist_df
        np_array = np.full((nlen, nlen), 0.0)
        for i in range(0, elen):
            np_array[df['src'].iloc[i], df['dst'].iloc[i]] = df['weights'].\
                                                             iloc[i]
        return np_array

    def to_numpy_matrix(self):
        """
        Returns the graph adjacency matrix as a NumPy matrix.
        """
        np_array = self.to_numpy_array()
        return np.asmatrix(np_array)

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
            src, dst, weights = graph_primtypes_wrapper.view_edge_list(self)
            self.edgelist = self.EdgeList(src, dst, weights)

        edgelist_df = self.edgelist.edgelist_df

        if self.properties.renumbered:
            edgelist_df = self.renumber_map.unrenumber(edgelist_df, "src")
            edgelist_df = self.renumber_map.unrenumber(edgelist_df, "dst")

        if not self.properties.directed:
            edgelist_df = edgelist_df[edgelist_df["src"] <= edgelist_df["dst"]]
            edgelist_df = edgelist_df.reset_index(drop=True)
            self.properties.edge_count = len(edgelist_df)

        return edgelist_df

    def delete_edge_list(self):
        """
        Delete the edge list.
        """
        # decrease reference count to free memory if the referenced objects are
        # no longer used.
        self.edgelist = None

    def __from_adjlist(self, offset_col, index_col, value_col=None):
        self.adjlist = simpleGraphImpl.AdjList(offset_col, index_col,
                                               value_col)

        if self.batch_enabled:
            self._replicate_adjlist()

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
            if self.transposedadjlist is not None and\
             self.properties.directed is False:
                off, ind, vals = (
                    self.transposedadjlist.offsets,
                    self.transposedadjlist.indices,
                    self.transposedadjlist.weights,
                )
            else:
                off, ind, vals = graph_primtypes_wrapper.view_adj_list(self)
            self.adjlist = self.AdjList(off, ind, vals)

            if self.batch_enabled:
                self._replicate_adjlist()

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
            if self.adjlist is not None and self.properties.directed is False:
                off, ind, vals = (
                    self.adjlist.offsets,
                    self.adjlist.indices,
                    self.adjlist.weights,
                )
            else:
                (
                    off,
                    ind,
                    vals,
                ) = graph_primtypes_wrapper.view_transposed_adj_list(self)
            self.transposedadjlist = self.transposedAdjList(off, ind, vals)

            if self.batch_enabled:
                self._replicate_transposed_adjlist()

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

    # FIXME: Update batch workflow and refactor to suitable file
    def enable_batch(self):
        client = mg_utils.get_client()
        comms = Comms.get_comms()

        if client is None or comms is None:
            raise RuntimeError("MG Batch needs a Dask Client and the "
                               "Communicator needs to be initialized.")

        self.batch_enabled = True

        if self.edgelist is not None:
            if self.batch_edgelists is None:
                self._replicate_edgelist()

        if self.adjlist is not None:
            if self.batch_adjlists is None:
                self._replicate_adjlist()

        if self.transposedadjlist is not None:
            if self.batch_transposed_adjlists is None:
                self._replicate_transposed_adjlist()

    def _replicate_edgelist(self):
        client = mg_utils.get_client()
        comms = Comms.get_comms()

        # FIXME: There  might be a better way to control it
        if client is None:
            return
        work_futures = replication.replicate_cudf_dataframe(
            self.edgelist.edgelist_df, client=client, comms=comms
        )

        self.batch_edgelists = work_futures

    def _replicate_adjlist(self):
        client = mg_utils.get_client()
        comms = Comms.get_comms()

        # FIXME: There  might be a better way to control it
        if client is None:
            return

        weights = None
        offsets_futures = replication.replicate_cudf_series(
            self.adjlist.offsets, client=client, comms=comms
        )
        indices_futures = replication.replicate_cudf_series(
            self.adjlist.indices, client=client, comms=comms
        )

        if self.adjlist.weights is not None:
            weights = replication.replicate_cudf_series(self.adjlist.weights)
        else:
            weights = {worker: None for worker in offsets_futures}

        merged_futures = {
            worker: [
                offsets_futures[worker],
                indices_futures[worker],
                weights[worker],
            ]
            for worker in offsets_futures
        }
        self.batch_adjlists = merged_futures

    # FIXME: Not implemented yet
    def _replicate_transposed_adjlist(self):
        self.batch_transposed_adjlists = True

    def get_two_hop_neighbors(self):
        """
        Compute vertex pairs that are two hops apart. The resulting pairs are
        sorted before returning.

        Returns
        -------
        df : cudf.DataFrame
            df[first] : cudf.Series
                the first vertex id of a pair, if an external vertex id
                is defined by only one column
            df[second] : cudf.Series
                the second vertex id of a pair, if an external vertex id
                is defined by only one column
        """

        df = graph_primtypes_wrapper.get_two_hop_neighbors(self)

        if self.properties.renumbered is True:
            df = self.renumber_map.unrenumber(df, "first")
            df = self.renumber_map.unrenumber(df, "second")

        return df

    def number_of_vertices(self):
        """
        Get the number of nodes in the graph.
        """
        if self.properties.node_count is None:
            if self.adjlist is not None:
                self.properties.node_count = len(self.adjlist.offsets) - 1
            elif self.transposedadjlist is not None:
                self.properties.node_count = len(
                    self.transposedadjlist.offsets) - 1
            elif self.edgelist is not None:
                df = self.edgelist.edgelist_df[["src", "dst"]]
                self.properties.node_count = df.max().max() + 1
            else:
                raise RuntimeError("Graph is Empty")
        return self.properties.node_count

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
        # TODO: Move to Outer graphs?
        if directed_edges and self.edgelist is not None:
            return len(self.edgelist.edgelist_df)
        if self.properties.edge_count is None:
            if self.edgelist is not None:
                if self.properties.directed is False:
                    self.properties.edge_count = len(
                        self.edgelist.edgelist_df[
                            self.edgelist.edgelist_df["src"]
                            >= self.edgelist.edgelist_df["dst"]
                        ]
                    )
                else:
                    self.properties.edge_count = len(self.edgelist.edgelist_df)
            elif self.adjlist is not None:
                self.properties.edge_count = len(self.adjlist.indices)
            elif self.transposedadjlist is not None:
                self.properties.edge_count = len(
                    self.transposedadjlist.indices)
            else:
                raise ValueError("Graph is Empty")
        return self.properties.edge_count

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
        vertex_subset : cudf.Series or iterable container, optional
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
        return self._degree(vertex_subset)

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
        (
            vertex_col,
            in_degree_col,
            out_degree_col,
        ) = graph_primtypes_wrapper._degrees(self)

        df = cudf.DataFrame()
        df["vertex"] = vertex_col
        df["in_degree"] = in_degree_col
        df["out_degree"] = out_degree_col

        if self.properties.renumbered is True:
            df = self.renumber_map.unrenumber(df, "vertex")

        if vertex_subset is not None:
            df = df[df['vertex'].isin(vertex_subset)]

        return df

    def _degree(self, vertex_subset, direction=Direction.ALL):
        vertex_col, degree_col = graph_primtypes_wrapper._degree(self,
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
        Return a directed representation of the graph Implementation.
        This function copies the internal structures and returns the
        directed view.
        """
        DiG.properties.renumbered = self.properties.renumbered
        DiG.renumber_map = self.renumber_map
        DiG.edgelist = self.edgelist
        DiG.adjlist = self.adjlist
        DiG.transposedadjlist = self.transposedadjlist

    def to_undirected(self, G):
        """
        Return an undirected copy of the graph.
        """
        G.properties.renumbered = self.properties.renumbered
        G.renumber_map = self.renumber_map
        if self.properties.directed is False:
            G.edgelist = self.edgelist
            G.adjlist = self.adjlist
            G.transposedadjlist = self.transposedadjlist
        else:
            df = self.edgelist.edgelist_df
            if self.edgelist.weights:
                source_col, dest_col, value_col = symmetrize(
                    df["src"], df["dst"], df["weights"]
                )
            else:
                source_col, dest_col = symmetrize(df["src"], df["dst"])
                value_col = None
            G.edgelist = simpleGraphImpl.EdgeList(source_col, dest_col,
                                                  value_col)

    def has_node(self, n):
        """
        Returns True if the graph contains the node n.
        """
        if self.properties.renumbered:
            tmp = self.renumber_map.to_internal_vertex_id(cudf.Series([n]))
            return tmp[0] is not cudf.NA and tmp[0] >= 0
        else:
            df = self.edgelist.edgelist_df[["src", "dst"]]
            return (df == n).any().any()

    def has_edge(self, u, v):
        """
        Returns True if the graph contains the edge (u,v).
        """
        if self.properties.renumbered:
            tmp = cudf.DataFrame({"src": [u, v]})
            tmp = tmp.astype({"src": "int"})
            tmp = self.renumber_map.add_internal_vertex_id(
                tmp, "id", "src", preserve_order=True
            )

            u = tmp["id"][0]
            v = tmp["id"][1]

        df = self.edgelist.edgelist_df
        return ((df["src"] == u) & (df["dst"] == v)).any()

    def has_self_loop(self):
        """
        Returns True if the graph has self loop.
        """
        # Detect self loop
        if self.properties.self_loop is None:
            elist = self.edgelist.edgelist_df
            if (elist["src"] == elist["dst"]).any():
                self.properties.self_loop = True
            else:
                self.properties.self_loop = False
        return self.properties.self_loop

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
        if self.edgelist is not None:
            df = self.edgelist.edgelist_df
            if self.properties.renumbered:
                # FIXME: If vertices are multicolumn
                #        this needs to return a dataframe
                # FIXME: This relies on current implementation
                #        of NumberMap, should not really expose
                #        this, perhaps add a method to NumberMap
                return self.renumber_map.implementation.df["0"]
            else:
                return cudf.concat([df["src"], df["dst"]]).unique()
        if self.adjlist is not None:
            return cudf.Series(np.arange(0, self.number_of_nodes()))

    def neighbors(self, n):
        if self.edgelist is None:
            raise RuntimeError("Graph has no Edgelist.")
        if self.properties.renumbered:
            node = self.renumber_map.to_internal_vertex_id(cudf.Series([n]))
            if len(node) == 0:
                return cudf.Series(dtype="int")
            n = node[0]

        df = self.edgelist.edgelist_df
        neighbors = df[df["src"] == n]["dst"].reset_index(drop=True)
        if self.properties.renumbered:
            # FIXME:  Multi-column vertices
            return self.renumber_map.from_internal_vertex_id(neighbors)["0"]
        else:
            return neighbors

    def vertex_column_size(self):
        if self.properties.renumbered:
            return self.renumber_map.vertex_column_size()
        else:
            return 1
