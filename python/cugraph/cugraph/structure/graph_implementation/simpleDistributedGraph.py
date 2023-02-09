# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
from cugraph.structure.symmetrize import symmetrize
import cupy
import cudf
import dask_cudf

from pylibcugraph import (
    MGGraph,
    ResourceHandle,
    GraphProperties,
)

from dask.distributed import wait, default_client
from cugraph.dask.common.input_utils import get_distributed_data
from pylibcugraph import get_two_hop_neighbors as pylibcugraph_get_two_hop_neighbors
import cugraph.dask.comms.comms as Comms


class simpleDistributedGraphImpl:
    edgeWeightCol = "value"
    edgeIdCol = "edge_id"
    edgeTypeCol = "edge_type"

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
            self.multi_edge = getattr(properties, "multi_edge", False)
            self.directed = properties.directed
            self.renumber = False
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

    def _make_plc_graph(
        sID,
        edata_x,
        graph_props,
        src_col_name,
        dst_col_name,
        store_transposed,
        num_edges,
    ):

        if simpleDistributedGraphImpl.edgeWeightCol in edata_x[0]:
            values = edata_x[0][simpleDistributedGraphImpl.edgeWeightCol]
            if values.dtype == "int32":
                values = values.astype("float32")
            elif values.dtype == "int64":
                values = values.astype("float64")
        else:
            # Some algos require the graph to be weighted
            values = cudf.Series(cupy.ones(len(edata_x[0]), dtype="float32"))

        if simpleDistributedGraphImpl.edgeIdCol in edata_x[0]:
            if simpleDistributedGraphImpl.edgeTypeCol not in edata_x[0]:
                raise ValueError("Must provide both edge id and edge type")

            values_id = edata_x[0][simpleDistributedGraphImpl.edgeIdCol]
            values_etype = edata_x[0][simpleDistributedGraphImpl.edgeTypeCol]
        else:
            values_id, values_etype = None, None

        return MGGraph(
            resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
            graph_properties=graph_props,
            src_array=edata_x[0][src_col_name],
            dst_array=edata_x[0][dst_col_name],
            weight_array=values,
            edge_id_array=values_id,
            edge_type_array=values_etype,
            store_transposed=store_transposed,
            num_edges=num_edges,
            do_expensive_check=False,
        )

    # Functions
    def __from_edgelist(
        self,
        input_ddf,
        source="source",
        destination="destination",
        edge_attr=None,
        renumber=True,
        store_transposed=False,
        legacy_renum_only=False,
    ):
        if not isinstance(input_ddf, dask_cudf.DataFrame):
            raise TypeError("input should be a dask_cudf dataFrame")

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

        # The dataframe will be symmetrized iff the graph is undirected
        # otherwise, the inital dataframe will be returned
        if edge_attr is not None:
            if isinstance(edge_attr, str):
                edge_attr = [edge_attr]
            if not (set(edge_attr).issubset(set(input_ddf.columns))):
                raise ValueError(
                    "edge_attr column name not found in input."
                    "Recheck the edge_attr parameter"
                )
            self.properties.weighted = True

            if len(edge_attr) == 1:
                input_ddf = input_ddf.rename(columns={edge_attr[0]: self.edgeWeightCol})
                value_col_names = [self.edgeWeightCol]
            elif len(edge_attr) == 3:
                weight_col, id_col, type_col = edge_attr
                input_ddf = input_ddf.rename(
                    columns={
                        weight_col: self.edgeWeightCol,
                        id_col: self.edgeIdCol,
                        type_col: self.edgeTypeCol,
                    }
                )

                value_col_names = [self.edgeWeightCol, self.edgeIdCol, self.edgeTypeCol]
            else:
                raise ValueError("Only 1 or 3 values may be provided" "for edge_attr")

            # The symmetrize step may add additional edges with unknown
            # ids and types for an undirected graph.  Therefore, only
            # directed graphs may be used with ids and types.
            if len(edge_attr) == 3:
                if not self.properties.directed:
                    raise ValueError(
                        "User-provided edge ids and edge "
                        "types are not permitted for an "
                        "undirected graph."
                    )
                if not legacy_renum_only:
                    raise ValueError(
                        "User-provided edge ids and edge "
                        "types are only permitted when "
                        "from_edgelist is called with "
                        "legacy_renum_only=True."
                    )

            source_col, dest_col, value_col = symmetrize(
                input_ddf,
                source,
                destination,
                value_col_names,
                multi=self.properties.multi_edge,
                symmetrize=not self.properties.directed,
            )

        else:
            input_ddf = input_ddf[ddf_columns]
            source_col, dest_col = symmetrize(
                input_ddf,
                source,
                destination,
                multi=self.properties.multi_edge,
                symmetrize=not self.properties.directed,
            )

        if isinstance(source_col, dask_cudf.Series):
            # Create a dask_cudf dataframe from the cudf series obtained
            # from symmetrization
            input_ddf = source_col.to_frame()
            input_ddf = input_ddf.rename(columns={source_col.name: source})
            input_ddf[destination] = dest_col
        else:
            # Multi column dask_cudf dataframe
            input_ddf = dask_cudf.concat([source_col, dest_col], axis=1)

        if edge_attr is not None:
            input_ddf[self.edgeWeightCol] = value_col[self.edgeWeightCol]
            if len(edge_attr) == 3:
                input_ddf[self.edgeIdCol] = value_col[self.edgeIdCol]
                input_ddf[self.edgeTypeCol] = value_col[self.edgeTypeCol]

        self.input_df = input_ddf

        #
        # Keep all of the original parameters so we can lazily
        # evaluate this function
        #

        # FIXME: Edge Attribute not handled
        # FIXME: the parameter below is no longer used for unrenumbering
        self.properties.renumber = renumber
        self.source_columns = source
        self.destination_columns = destination

        # If renumbering is not enabled, this function will only create
        # the edgelist_df and not do any renumbering.
        # C++ renumbering is enabled by default for algorithms that
        # support it (but only called if renumbering is on)
        self.compute_renumber_edge_list(
            transposed=store_transposed, legacy_renum_only=legacy_renum_only
        )

        self.properties.renumbered = self.renumber_map.implementation.numbered
        ddf = self.edgelist.edgelist_df

        num_edges = len(ddf)
        edge_data = get_distributed_data(ddf)
        src_col_name = self.renumber_map.renumbered_src_col_name
        dst_col_name = self.renumber_map.renumbered_dst_col_name
        graph_props = GraphProperties(
            is_multigraph=self.properties.multi_edge,
            is_symmetric=not self.properties.directed,
        )

        _client = default_client()
        self._plc_graph = {
            w: _client.submit(
                simpleDistributedGraphImpl._make_plc_graph,
                Comms.get_session_id(),
                edata,
                graph_props,
                src_col_name,
                dst_col_name,
                store_transposed,
                num_edges,
                workers=[w],
            )
            for w, edata in edge_data.worker_to_parts.items()
        }

        wait(self._plc_graph)

    @property
    def renumbered(self):
        # This property is now used to determine if a dataframe was renumbered
        # by checking the column name. Only the renumbered dataframes will have
        # their column names renamed to 'renumbered_src' and 'renumbered_dst'
        renumbered_vertex_col_names = ["renumbered_src", "renumbered_dst"]
        if self.edgelist is not None:
            if self.edgelist.edgelist_df is not None and (
                set(renumbered_vertex_col_names).issubset(
                    set(self.edgelist.edgelist_df.columns)
                )
            ):
                return True
        return False

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
        df : dask_cudf.DataFrame
            This dask_cudf.DataFrame wraps source, destination and weight
            df[src] : dask_cudf.Series
                contains the source index for each edge
            df[dst] : dask_cudf.Series
                contains the destination index for each edge
            df[weight] : dask_cudf.Series
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
                if self.renumbered is True:
                    src_col_name = self.renumber_map.renumbered_src_col_name
                    dst_col_name = self.renumber_map.renumbered_dst_col_name
                # FIXME: from_dask_cudf_edgelist() currently requires
                # renumber=True for MG, so this else block will not be
                # used. Should this else block be removed and added back when
                # the restriction is removed?
                else:
                    src_col_name = "src"
                    dst_col_name = "dst"

                ddf = self.edgelist.edgelist_df[[src_col_name, dst_col_name]]
                # ddf = self.edgelist.edgelist_df[["src", "dst"]]
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
        vertex_subset : cudf or dask_cudf object, iterable container,
            opt. (default=None)
            A container of vertices for displaying corresponding in-degree.
            If not set, degrees are computed for the entire set of vertices.

        Returns
        -------
        df : dask_cudf.DataFrame
            Distributed GPU DataFrame of size N (the default) or the size of
            the given vertices (vertex_subset) containing the in_degree.
            The ordering is relative to the adjacency list, or that given by
            the specified vertex_subset.
            df[vertex] : dask_cudf.Series
                The vertex IDs (will be identical to vertex_subset if
                specified).
            df[degree] : dask_cudf.Series
                The computed in-degree of the corresponding vertex.
        Examples
        --------
        >>> M = dask_cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
        ...                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_dask_cudf_edgelist(M, '0', '1')
        >>> df = G.in_degree([0,9,12])

        """
        src_col_name = self.source_columns
        dst_col_name = self.destination_columns

        # select only the vertex columns
        if not isinstance(src_col_name, list) and not isinstance(dst_col_name, list):
            vertex_col_names = [src_col_name] + [dst_col_name]

        df = self.input_df[vertex_col_names]
        df = df.drop(columns=src_col_name)

        nodes = self.nodes()
        if isinstance(nodes, dask_cudf.Series):
            nodes = nodes.to_frame()

        if not isinstance(dst_col_name, list):
            df = df.rename(columns={dst_col_name: "vertex"})
            dst_col_name = "vertex"

        vertex_col_names = df.columns
        nodes.columns = vertex_col_names

        df["degree"] = 1

        # FIXME: leverage the C++ in_degree for optimal performance
        in_degree = (
            df.groupby(dst_col_name)
            .degree.count(split_out=df.npartitions)
            .reset_index()
        )

        # Add vertices with zero in_degree
        in_degree = nodes.merge(in_degree, how="outer").fillna(0)

        # Convert vertex_subset to dataframe.
        if vertex_subset is not None:
            if not isinstance(vertex_subset, (dask_cudf.DataFrame, cudf.DataFrame)):
                if isinstance(vertex_subset, dask_cudf.Series):
                    vertex_subset = vertex_subset.to_frame()
                else:
                    df = cudf.DataFrame()
                    if isinstance(vertex_subset, (cudf.Series, list)):
                        df["vertex"] = vertex_subset
                        vertex_subset = df
            if isinstance(vertex_subset, (dask_cudf.DataFrame, cudf.DataFrame)):
                vertex_subset.columns = vertex_col_names
                in_degree = in_degree.merge(vertex_subset, how="inner")
            else:
                raise TypeError(
                    f"Expected type are: cudf, dask_cudf objects, "
                    f"iterable container, got "
                    f"{type(vertex_subset)}"
                )
        return in_degree

    def out_degree(self, vertex_subset=None):
        """
        Compute vertex out-degree. Vertex out-degree is the number of edges
        pointing out from the vertex. By default, this method computes vertex
        degrees for the entire set of vertices. If vertex_subset is provided,
        this method optionally filters out all but those listed in
        vertex_subset.

        Parameters
        ----------
        vertex_subset : cudf or dask_cudf object, iterable container,
            opt. (default=None)
            A container of vertices for displaying corresponding out-degree.
            If not set, degrees are computed for the entire set of vertices.

        Returns
        -------
        df : dask_cudf.DataFrame
            Distributed GPU DataFrame of size N (the default) or the size of
            the given vertices (vertex_subset) containing the out_degree.
            The ordering is relative to the adjacency list, or that given by
            the specified vertex_subset.
            df[vertex] : dask_cudf.Series
                The vertex IDs (will be identical to vertex_subset if
                specified).
            df[degree] : dask_cudf.Series
                The computed out-degree of the corresponding vertex.
        Examples
        --------
        >>> M = dask_cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
        ...                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_dask_cudf_edgelist(M, '0', '1')
        >>> df = G.out_degree([0,9,12])

        """
        src_col_name = self.source_columns
        dst_col_name = self.destination_columns

        # select only the vertex columns
        if not isinstance(src_col_name, list) and not isinstance(dst_col_name, list):
            vertex_col_names = [src_col_name] + [dst_col_name]

        df = self.input_df[vertex_col_names]
        df = df.drop(columns=dst_col_name)

        nodes = self.nodes()
        if isinstance(nodes, dask_cudf.Series):
            nodes = nodes.to_frame()

        if not isinstance(src_col_name, list):
            df = df.rename(columns={src_col_name: "vertex"})
            src_col_name = "vertex"

        vertex_col_names = df.columns

        nodes.columns = vertex_col_names

        df["degree"] = 1
        # leverage the C++ out_degree for optimal performance
        out_degree = (
            df.groupby(src_col_name)
            .degree.count(split_out=df.npartitions)
            .reset_index()
        )

        # Add vertices with zero out_degree
        out_degree = nodes.merge(out_degree, how="outer").fillna(0)

        # Convert vertex_subset to dataframe.
        if vertex_subset is not None:
            if not isinstance(vertex_subset, (dask_cudf.DataFrame, cudf.DataFrame)):
                if isinstance(vertex_subset, dask_cudf.Series):
                    vertex_subset = vertex_subset.to_frame()
                else:
                    df = cudf.DataFrame()
                    if isinstance(vertex_subset, (cudf.Series, list)):
                        df["vertex"] = vertex_subset
                        vertex_subset = df
            if isinstance(vertex_subset, (dask_cudf.DataFrame, cudf.DataFrame)):
                vertex_subset.columns = vertex_col_names
                out_degree = out_degree.merge(vertex_subset, how="inner")
            else:
                raise TypeError(
                    f"Expected type are: cudf, dask_cudf objects, "
                    f"iterable container, got "
                    f"{type(vertex_subset)}"
                )

        return out_degree

    def degree(self, vertex_subset=None):
        """
        Compute vertex degree, which is the total number of edges incident
        to a vertex (both in and out edges). By default, this method computes
        degrees for the entire set of vertices. If vertex_subset is provided,
        then this method optionally filters out all but those listed in
        vertex_subset.

        Parameters
        ----------
        vertex_subset : cudf or dask_cudf object, iterable container,
            opt. (default=None)
            a container of vertices for displaying corresponding degree. If not
            set, degrees are computed for the entire set of vertices.
        Returns
        -------
        df : dask_cudf.DataFrame
            Distributed GPU DataFrame of size N (the default) or the size of
            the given vertices (vertex_subset) containing the degree.
            The ordering is relative to the adjacency list, or that given by
            the specified vertex_subset.
            df['vertex'] : dask_cudf.Series
                The vertex IDs (will be identical to vertex_subset if
                specified).
            df['degree'] : dask_cudf.Series
                The computed degree of the corresponding vertex.
        Examples
        --------
        >>> M = dask_cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
        ...                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_dask_cudf_edgelist(M, '0', '1')
        >>> all_df = G.degree()
        >>> subset_df = G.degree([0,9,12])

        """

        vertex_in_degree = self.in_degree(vertex_subset)
        vertex_out_degree = self.out_degree(vertex_subset)
        # FIXME: leverage the C++ degree for optimal performance
        vertex_degree = dask_cudf.concat([vertex_in_degree, vertex_out_degree])
        vertex_degree = vertex_degree.groupby(["vertex"], as_index=False).sum(
            split_out=self.input_df.npartitions
        )

        return vertex_degree

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
        df : dask_cudf.DataFrame
            Distributed GPU DataFrame of size N (the default) or the size of
            the given vertices (vertex_subset) containing the degrees.
            The ordering is relative to the adjacency list, or that given by
            the specified vertex_subset.
            df['vertex'] : dask_cudf.Series
                The vertex IDs (will be identical to vertex_subset if
                specified).
            df['in_degree'] : dask_cudf.Series
                The in-degree of the vertex.
            df['out_degree'] : dask_cudf.Series
                The out-degree of the vertex.

        Examples
        --------
        >>> M = dask_cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
        ...                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_dask_cudf_edgelist(M, '0', '1')
        >>> df = G.degrees([0,9,12])

        """
        raise NotImplementedError("Not supported for distributed graph")

    def _degree(self, vertex_subset, direction=Direction.ALL):
        vertex_col, degree_col = graph_primtypes_wrapper._mg_degree(self, direction)
        df = cudf.DataFrame()
        df["vertex"] = vertex_col
        df["degree"] = degree_col

        if self.renumbered is True:
            df = self.renumber_map.unrenumber(df, "vertex")

        if vertex_subset is not None:
            df = df[df["vertex"].isin(vertex_subset)]

        return df

    def get_two_hop_neighbors(self, start_vertices=None):
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

        if isinstance(start_vertices, int):
            start_vertices = [start_vertices]

        if isinstance(start_vertices, list):
            start_vertices = cudf.Series(start_vertices)

        if start_vertices is not None:
            if self.renumbered:
                start_vertices = self.renumber_map.to_internal_vertex_id(start_vertices)
                start_vertices_type = self.edgelist.edgelist_df.dtypes[0]
            else:
                start_vertices_type = self.input_df.dtypes[0]

            if not isinstance(start_vertices, (dask_cudf.Series)):
                start_vertices = dask_cudf.from_cudf(
                    start_vertices,
                    npartitions=min(self._npartitions, len(start_vertices)),
                )
                start_vertices = start_vertices.astype(start_vertices_type)

            start_vertices = get_distributed_data(start_vertices)
            wait(start_vertices)
            start_vertices = start_vertices.worker_to_parts

        def _call_plc_two_hop_neighbors(sID, mg_graph_x, start_vertices):
            return pylibcugraph_get_two_hop_neighbors(
                resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
                graph=mg_graph_x,
                start_vertices=start_vertices,
                do_expensive_check=False,
            )

        _client = default_client()
        if start_vertices is not None:
            result = [
                _client.submit(
                    _call_plc_two_hop_neighbors,
                    Comms.get_session_id(),
                    self._plc_graph[w],
                    start_vertices[w][0],
                    workers=[w],
                    allow_other_workers=False,
                )
                for w in Comms.get_workers()
            ]
        else:
            result = [
                _client.submit(
                    _call_plc_two_hop_neighbors,
                    Comms.get_session_id(),
                    self._plc_graph[w],
                    start_vertices,
                    workers=[w],
                    allow_other_workers=False,
                )
                for w in Comms.get_workers()
            ]

        wait(result)

        def convert_to_cudf(cp_arrays):
            """
            Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
            """
            first, second = cp_arrays
            df = cudf.DataFrame()
            df["first"] = first
            df["second"] = second
            return df

        _client = default_client()
        cudf_result = [
            _client.submit(convert_to_cudf, cp_arrays) for cp_arrays in result
        ]

        wait(cudf_result)
        ddf = dask_cudf.from_delayed(cudf_result).persist()
        wait(ddf)

        # Wait until the inactive futures are released
        wait([(r.release(), c_r.release()) for r, c_r in zip(result, cudf_result)])

        if self.properties.renumbered:
            ddf = self.renumber_map.unrenumber(ddf, "first")
            ddf = self.renumber_map.unrenumber(ddf, "second")

        return ddf

    def to_directed(self, G):
        """
        Return a directed representation of the graph.

        Returns
        -------
        G : Graph(directed=True)
            A directed graph with the same nodes, and each edge (u,v,weights)
            replaced by two directed edges (u,v,weights) and (v,u,weights).

        Examples
        --------
        >>> M = dask_cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
        ...                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_dask_cudf_edgelist(M, '0', '1')
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
        >>> M = dask_cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
        ...         dtype=['int32', 'int32', 'float32'], header=None)
        >>> DiG = cugraph.Graph(directed=True)
        >>> DiG.dask_from_cudf_edgelist(M, '0', '1')
        >>> G = DiG.to_undirected()
        """

        # TODO: Add support
        raise NotImplementedError("Not supported for distributed graph")

    def has_node(self, n):
        """

        Returns True if the graph contains the node(s) n.
        Examples
        --------
        >>> M = dask_cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
        ...                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph(directed=True)
        >>> G.from_dask_cudf_edgelist(M, '0', '1')
        >>> valid_source = cudf.Series([5])
        >>> invalid_source = cudf.Series([55])
        >>> is_valid_vertex = G.has_node(valid_source)
        >>> assert is_valid_vertex is True
        >>> is_valid_vertex = G.has_node(invalid_source)
        >>> assert is_valid_vertex is False
        """

        # Convert input to dataframes so that it can be compared through merge
        if not isinstance(n, (dask_cudf.DataFrame, cudf.DataFrame)):
            if isinstance(n, dask_cudf.Series):
                n = n.to_frame()
            else:
                df = cudf.DataFrame()
                if not isinstance(n, (cudf.DataFrame, cudf.Series)):
                    n = [n]
                if isinstance(n, (cudf.Series, list)):
                    df["vertex"] = n
                    n = df

        if isinstance(n, (dask_cudf.DataFrame, cudf.DataFrame)):
            nodes = self.nodes()
            if not isinstance(self.nodes(), (dask_cudf.DataFrame, cudf.DataFrame)):
                nodes = nodes.to_frame()

            nodes.columns = n.columns

            valid_vertex = nodes.merge(n, how="inner")
            return len(valid_vertex) == len(n)

    def has_edge(self, u, v):
        """
        Returns True if the graph contains the edge (u,v).
        """
        # TODO: Verify Correctness
        if self.renumbered:
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
        if self.renumbered is True:
            src_col_name = self.renumber_map.renumbered_src_col_name
            dst_col_name = self.renumber_map.renumbered_dst_col_name
            # FIXME: from_dask_cudf_edgelist() currently requires
            # renumber=True for MG, so this else block will not be
            # used. Should this else block be removed and added back when
            # the restriction is removed?
        else:
            src_col_name = "src"
            dst_col_name = "dst"

        # return self.view_edge_list()[["src", "dst"]]
        return self.view_edge_list()[[src_col_name, dst_col_name]]

    def nodes(self):
        """
        Returns all nodes in the graph as a dask_cudf.Series.
        If multi columns vertices, return a dask_cudf.DataFrame.

        If the edgelist was renumbered, this call returns the internal
        nodes in the graph. To get the original nodes, convert the result to
        a dataframe and do 'renumber_map.unrenumber' or 'G.unrenumber'
        """

        if self.renumbered:
            # FIXME: This relies on current implementation
            #        of NumberMap, should not really expose
            #        this, perhaps add a method to NumberMap

            df = self.renumber_map.implementation.ddf.drop(columns="global_id")

            if len(df.columns) > 1:
                return df
            else:
                return df[df.columns[0]]

        else:
            df = self.input_df
            return dask_cudf.concat(
                [df[self.source_columns], df[self.destination_columns]]
            ).drop_duplicates()

    def neighbors(self, n):
        if self.edgelist is None:
            raise RuntimeError("Graph has no Edgelist.")
        # FIXME: Add renumbering of node n
        ddf = self.edgelist.edgelist_df
        return ddf[ddf["src"] == n]["dst"].reset_index(drop=True)

    def compute_renumber_edge_list(self, transposed=False, legacy_renum_only=False):
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

        legacy_renum_only : (optional) bool
            if True, The C++ renumbering will not be triggered.
            This parameter is added for new algos following the
            C/Pylibcugraph path
        """

        if not self.properties.renumber:
            self.edgelist = self.EdgeList(self.input_df)
            self.renumber_map = None
        else:
            if self.edgelist is not None:
                if self.properties.directed is False:
                    return

                if self.properties.store_transposed == transposed:
                    return

                del self.edgelist

            (
                renumbered_ddf,
                number_map,
                aggregate_segment_offsets,
            ) = NumberMap.renumber_and_segment(
                self.input_df,
                self.source_columns,
                self.destination_columns,
                store_transposed=transposed,
                legacy_renum_only=legacy_renum_only,
            )

            self.edgelist = self.EdgeList(renumbered_ddf)
            self.renumber_map = number_map
            self.aggregate_segment_offsets = aggregate_segment_offsets
            self.properties.store_transposed = transposed

    def vertex_column_size(self):
        if self.renumbered:
            return self.renumber_map.vertex_column_size()
        else:
            return 1

    @property
    def _npartitions(self) -> int:
        return len(self._plc_graph)
