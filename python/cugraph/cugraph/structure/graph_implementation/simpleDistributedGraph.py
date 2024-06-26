# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

import gc
from typing import Union, Iterable
import warnings

import cudf
import cupy as cp
import dask
import dask_cudf
from dask import delayed
from dask.distributed import wait, default_client
import numpy as np
from pylibcugraph import (
    MGGraph,
    ResourceHandle,
    GraphProperties,
    get_two_hop_neighbors as pylibcugraph_get_two_hop_neighbors,
    select_random_vertices as pylibcugraph_select_random_vertices,
    degrees as pylibcugraph_degrees,
    in_degrees as pylibcugraph_in_degrees,
    out_degrees as pylibcugraph_out_degrees,
)

from cugraph.structure.number_map import NumberMap
from cugraph.structure.symmetrize import symmetrize
from cugraph.dask.common.part_utils import (
    persist_dask_df_equal_parts_per_worker,
)
from cugraph.dask.common.mg_utils import run_gc_on_dask_cluster
import cugraph.dask.comms.comms as Comms
from cugraph.structure.symmetrize import _memory_efficient_drop_duplicates


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
        self.properties = simpleDistributedGraphImpl.Properties(properties)
        self.source_columns = None
        self.destination_columns = None
        self.weight_column = None
        self.vertex_columns = None
        self.vertex_type = None
        self.weight_type = None
        self.edge_id_type = None
        self.edge_type_id_type = None

    def _make_plc_graph(
        sID,
        edata_x,
        graph_props,
        src_col_name,
        dst_col_name,
        store_transposed,
        vertex_type,
        weight_type,
        edge_id_type,
        edge_type_id,
        drop_multi_edges,
    ):
        weights = None
        edge_ids = None
        edge_types = None

        num_arrays = len(edata_x)
        if weight_type is not None:
            weights = [
                edata_x[i][simpleDistributedGraphImpl.edgeWeightCol]
                for i in range(num_arrays)
            ]
            if weight_type == "int32":
                weights = [w_array.astype("float32") for w_array in weights]
            elif weight_type == "int64":
                weights = [w_array.astype("float64") for w_array in weights]

        if edge_id_type is not None:
            edge_ids = [
                edata_x[i][simpleDistributedGraphImpl.edgeIdCol]
                for i in range(num_arrays)
            ]
            if vertex_type == "int64" and edge_id_type != "int64":
                edge_ids = [e_id_array.astype("int64") for e_id_array in edge_ids]
                warnings.warn(
                    f"Vertex type is int64 but edge id type is {edge_ids[0].dtype}"
                    ", automatically casting edge id type to int64. "
                    "This may cause extra memory usage.  Consider passing"
                    " a int64 list of edge ids instead."
                )
        if edge_type_id is not None:
            edge_types = [
                edata_x[i][simpleDistributedGraphImpl.edgeTypeCol]
                for i in range(num_arrays)
            ]

        src_array = [edata_x[i][src_col_name] for i in range(num_arrays)]
        dst_array = [edata_x[i][dst_col_name] for i in range(num_arrays)]
        plc_graph = MGGraph(
            resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
            graph_properties=graph_props,
            src_array=src_array if src_array else cudf.Series(dtype=vertex_type),
            dst_array=dst_array if dst_array else cudf.Series(dtype=vertex_type),
            weight_array=weights
            if weights
            else ([cudf.Series(dtype=weight_type)] if weight_type else None),
            edge_id_array=edge_ids
            if edge_ids
            else ([cudf.Series(dtype=edge_id_type)] if edge_id_type else None),
            edge_type_array=edge_types
            if edge_types
            else ([cudf.Series(dtype=edge_type_id)] if edge_type_id else None),
            num_arrays=num_arrays,
            store_transposed=store_transposed,
            do_expensive_check=False,
            drop_multi_edges=drop_multi_edges,
        )
        del edata_x
        gc.collect()

        return plc_graph

    # Functions
    def __from_edgelist(
        self,
        input_ddf,
        source="source",
        destination="destination",
        edge_attr=None,
        weight=None,
        edge_id=None,
        edge_type=None,
        renumber=True,
        store_transposed=False,
        legacy_renum_only=False,
    ):
        if not isinstance(input_ddf, dask_cudf.DataFrame):
            raise TypeError("input should be a dask_cudf dataFrame")

        if renumber is False:
            if type(source) is list and type(destination) is list:
                raise ValueError("set renumber to True for multi column ids")
            elif input_ddf[source].dtype not in [np.int32, np.int64] or input_ddf[
                destination
            ].dtype not in [np.int32, np.int64]:
                raise ValueError("set renumber to True for non integer columns ids")

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
        self.vertex_columns = ddf_columns.copy()
        _client = default_client()
        workers = _client.scheduler_info()["workers"]
        # Repartition to 2 partitions per GPU for memory efficient process
        input_ddf = input_ddf.repartition(npartitions=len(workers) * 2)
        # The dataframe will be symmetrized iff the graph is undirected
        # otherwise, the inital dataframe will be returned
        if edge_attr is not None:
            if weight is not None or edge_id is not None or edge_type is not None:
                raise ValueError(
                    "If specifying edge_attr, cannot specify weight/edge_id/edge_type"
                )
            if isinstance(edge_attr, str):
                weight = edge_attr
                edge_attr = [weight]
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
                input_ddf = input_ddf[ddf_columns + [weight_col, id_col, type_col]]
                input_ddf.columns = ddf_columns + [
                    self.edgeWeightCol,
                    self.edgeIdCol,
                    self.edgeTypeCol,
                ]
                value_col_names = [self.edgeWeightCol, self.edgeIdCol, self.edgeTypeCol]
            else:
                raise ValueError("Only 1 or 3 values may be provided" "for edge_attr")

            # The symmetrize step may add additional edges with unknown
            # ids and types for an undirected graph.  Therefore, only
            # directed graphs may be used with ids and types.
            # FIXME: Drop the check in symmetrize.py as it is redundant
            if len(edge_attr) == 3:
                if not self.properties.directed:
                    raise ValueError(
                        "User-provided edge ids and/or edge "
                        "types are not permitted for an "
                        "undirected graph."
                    )

        else:
            value_col_names = {}
            if weight is not None:
                value_col_names[weight] = self.edgeWeightCol
                self.properties.weighted = True
            if edge_id is not None:
                value_col_names[edge_id] = self.edgeIdCol
            if edge_type is not None:
                value_col_names[edge_type] = self.edgeTypeCol

            if len(value_col_names.keys()) > 0:
                input_ddf = input_ddf.rename(columns=value_col_names)
            value_col_names = list(value_col_names.values())

        ddf_columns += value_col_names
        input_ddf = input_ddf[ddf_columns]

        if len(value_col_names) == 0:
            source_col, dest_col = symmetrize(
                input_ddf,
                source,
                destination,
                multi=True,  # Deprecated parameter
                symmetrize=not self.properties.directed,
            )
            value_col = None
        else:
            source_col, dest_col, value_col = symmetrize(
                input_ddf,
                source,
                destination,
                value_col_names,
                multi=True,  # Deprecated parameter
                symmetrize=not self.properties.directed,
            )

        # Create a dask_cudf dataframe from the cudf series
        # or dataframe objects obtained from symmetrization
        if isinstance(source_col, dask_cudf.Series):
            frames = [
                source_col.to_frame(name=source),
                dest_col.to_frame(name=destination),
            ]
        else:
            frames = [source_col, dest_col]

        if value_col is not None:
            frames.append(value_col[value_col_names])

        input_ddf = dask_cudf.concat(frames, axis=1)

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
        self.weight_column = weight

        # If renumbering is not enabled, this function will only create
        # the edgelist_df and not do any renumbering.
        # C++ renumbering is enabled by default for algorithms that
        # support it (but only called if renumbering is on)
        self.compute_renumber_edge_list(
            transposed=store_transposed, legacy_renum_only=legacy_renum_only
        )

        if renumber is False:
            self.properties.renumbered = False
            src_col_name = self.source_columns
            dst_col_name = self.destination_columns

        else:
            # If 'renumber' is set to 'True', an extra renumbering (python)
            # occurs if there are non-integer or multi-columns vertices
            self.properties.renumbered = self.renumber_map.is_renumbered

            src_col_name = self.renumber_map.renumbered_src_col_name
            dst_col_name = self.renumber_map.renumbered_dst_col_name

        ddf = self.edgelist.edgelist_df

        # Get the edgelist dtypes
        self.vertex_type = ddf[src_col_name].dtype
        if simpleDistributedGraphImpl.edgeWeightCol in ddf.columns:
            self.weight_type = ddf[simpleDistributedGraphImpl.edgeWeightCol].dtype
        if simpleDistributedGraphImpl.edgeIdCol in ddf.columns:
            self.edge_id_type = ddf[simpleDistributedGraphImpl.edgeIdCol].dtype
        if simpleDistributedGraphImpl.edgeTypeCol in ddf.columns:
            self.edge_type_id_type = ddf[simpleDistributedGraphImpl.edgeTypeCol].dtype

        graph_props = GraphProperties(
            is_multigraph=self.properties.multi_edge,
            is_symmetric=not self.properties.directed,
        )
        ddf = ddf.repartition(npartitions=len(workers) * 2)
        workers = _client.scheduler_info()["workers"].keys()
        persisted_keys_d = persist_dask_df_equal_parts_per_worker(
            ddf, _client, return_type="dict"
        )
        del ddf

        delayed_tasks_d = {
            w: delayed(simpleDistributedGraphImpl._make_plc_graph)(
                Comms.get_session_id(),
                edata,
                graph_props,
                src_col_name,
                dst_col_name,
                store_transposed,
                self.vertex_type,
                self.weight_type,
                self.edge_id_type,
                self.edge_type_id_type,
                not self.properties.multi_edge,
            )
            for w, edata in persisted_keys_d.items()
        }
        del persisted_keys_d
        self._plc_graph = {
            w: _client.compute(
                delayed_task, workers=w, allow_other_workers=False, pure=False
            )
            for w, delayed_task in delayed_tasks_d.items()
        }
        del delayed_tasks_d
        run_gc_on_dask_cluster(_client)
        wait(list(self._plc_graph.values()))
        run_gc_on_dask_cluster(_client)

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
        FIXME: Should this also return the edge ids and types?

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

        edgelist_df = self.input_df
        is_string_dtype = False
        is_multi_column = False
        wgtCol = simpleDistributedGraphImpl.edgeWeightCol
        if not self.properties.directed:
            srcCol = self.source_columns
            dstCol = self.destination_columns
            if self.renumber_map.unrenumbered_id_type == "object":
                # FIXME: Use the renumbered vertices instead and then un-renumber.
                # This operation can be expensive.
                is_string_dtype = True
                edgelist_df = self.edgelist.edgelist_df
                srcCol = self.renumber_map.renumbered_src_col_name
                dstCol = self.renumber_map.renumbered_dst_col_name

            if isinstance(srcCol, list):
                srcCol = self.renumber_map.renumbered_src_col_name
                dstCol = self.renumber_map.renumbered_dst_col_name
                edgelist_df = self.edgelist.edgelist_df
                # unrenumber before extracting the upper triangular part
                if len(self.source_columns) == 1:
                    edgelist_df = self.renumber_map.unrenumber(edgelist_df, srcCol)
                    edgelist_df = self.renumber_map.unrenumber(edgelist_df, dstCol)
                else:
                    is_multi_column = True

            if not self.properties.multi_edge:
                # Drop parallel edges for non MultiGraph
                # FIXME: Drop multi edges with the CAPI instead.
                _client = default_client()
                workers = _client.scheduler_info()["workers"]
                edgelist_df = _memory_efficient_drop_duplicates(
                    edgelist_df, [srcCol, dstCol], len(workers)
                )

            edgelist_df[srcCol], edgelist_df[dstCol] = edgelist_df[
                [srcCol, dstCol]
            ].min(axis=1), edgelist_df[[srcCol, dstCol]].max(axis=1)

            edgelist_df = edgelist_df.groupby(by=[srcCol, dstCol]).sum().reset_index()
            if wgtCol in edgelist_df.columns:
                # FIXME: This breaks if there are are multi edges as those will
                # be dropped during the symmetrization step and the original 'weight'
                # will be halved.
                edgelist_df[wgtCol] /= 2

        if is_string_dtype or is_multi_column:
            # unrenumber the vertices
            edgelist_df = self.renumber_map.unrenumber(edgelist_df, srcCol)
            edgelist_df = self.renumber_map.unrenumber(edgelist_df, dstCol)

        if self.properties.renumbered:
            edgelist_df = edgelist_df.rename(
                columns=self.renumber_map.internal_to_external_col_names
            )

        # If there is no 'wgt' column, nothing will happen
        edgelist_df = edgelist_df.rename(columns={wgtCol: self.weight_column})

        self.properties.edge_count = len(edgelist_df)
        return edgelist_df

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
            self.properties.node_count = len(self.nodes())
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

        if directed_edges and self.edgelist is not None:
            return len(self.edgelist.edgelist_df)

        if self.properties.edge_count is None:
            if self.edgelist is not None:
                self.view_edge_list()
            else:
                raise RuntimeError("Graph is Empty")
        return self.properties.edge_count

    def degrees_function(
        self,
        vertex_subset: Union[cudf.Series, dask_cudf.Series, Iterable] = None,
        degree_type: str = "in_degree",
    ) -> dask_cudf.DataFrame:
        """
        Compute vertex in-degree, out-degree, degree and degrees.

        1) Vertex in-degree is the number of edges pointing into the vertex.
        2) Vertex out-degree is the number of edges pointing out from the vertex.
        3) Vertex degree, is the total number of edges incident to a vertex
            (both in and out edges)
        4) Vertex degrees computes vertex in-degree and out-degree.

        By default, this method computes vertex in-degree, out-degree, degree
        or degrees for the entire set of vertices. If vertex_subset is provided,
        this method optionally filters out all but those listed in
        vertex_subset.

        Parameters
        ----------
        vertex_subset : cudf.Series or dask_cudf.Series, iterable container, optional
            A container of vertices for displaying corresponding in-degree.
            If not set, degrees are computed for the entire set of vertices.

        degree_type : str (default='in_degree')

        Returns
        -------
        df : dask_cudf.DataFrame
            GPU DataFrame of size N (the default) or the size of the given
            vertices (vertex_subset) containing the in_degree, out_degrees,
            degree or degrees. The ordering is relative to the adjacency list,
            or that given by the specified vertex_subset.

        Examples
        --------
        >>> M = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
        ...                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(M, '0', '1')
        >>> df = G.degrees_function([0,9,12], "in_degree")

        """
        _client = default_client()

        def _call_plc_degrees_function(
            sID: bytes, mg_graph_x, source_vertices: cudf.Series, degree_type: str
        ) -> cp.array:

            if degree_type == "in_degree":
                results = pylibcugraph_in_degrees(
                    resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
                    graph=mg_graph_x,
                    source_vertices=source_vertices,
                    do_expensive_check=False,
                )
            elif degree_type == "out_degree":
                results = pylibcugraph_out_degrees(
                    resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
                    graph=mg_graph_x,
                    source_vertices=source_vertices,
                    do_expensive_check=False,
                )
            elif degree_type in ["degree", "degrees"]:
                results = pylibcugraph_degrees(
                    resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
                    graph=mg_graph_x,
                    source_vertices=source_vertices,
                    do_expensive_check=False,
                )
            else:
                raise ValueError(
                    "Incorrect degree type passed, valid values are ",
                    "'in_degree', 'out_degree', 'degree' and 'degrees' ",
                    f"got '{degree_type}'",
                )

            return results

        if isinstance(vertex_subset, int):
            vertex_subset = [vertex_subset]

        if isinstance(vertex_subset, list):
            vertex_subset = cudf.Series(vertex_subset)

        if vertex_subset is not None:
            if self.renumbered:
                vertex_subset = self.renumber_map.to_internal_vertex_id(vertex_subset)
                vertex_subset_type = self.edgelist.edgelist_df.dtypes.iloc[0]
            else:
                vertex_subset_type = self.input_df.dtypes.iloc[0]

            vertex_subset = vertex_subset.astype(vertex_subset_type)

        cupy_result = [
            _client.submit(
                _call_plc_degrees_function,
                Comms.get_session_id(),
                self._plc_graph[w],
                vertex_subset,
                degree_type,
                workers=[w],
                allow_other_workers=False,
            )
            for w in Comms.get_workers()
        ]

        wait(cupy_result)

        def convert_to_cudf(cp_arrays: cp.ndarray, degree_type: bool) -> cudf.DataFrame:
            """
            Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
            """
            df = cudf.DataFrame()
            df["vertex"] = cp_arrays[0]
            if degree_type in ["in_degree", "out_degree"]:
                df["degree"] = cp_arrays[1]
            # degree_type must be either 'degree' or 'degrees'
            else:
                if degree_type == "degrees":
                    df["in_degree"] = cp_arrays[1]
                    df["out_degree"] = cp_arrays[2]
                else:
                    df["degree"] = cp_arrays[1] + cp_arrays[2]
            return df

        cudf_result = [
            _client.submit(
                convert_to_cudf,
                cp_arrays,
                degree_type,
                workers=_client.who_has(cp_arrays)[cp_arrays.key],
            )
            for cp_arrays in cupy_result
        ]

        wait(cudf_result)
        ddf = dask_cudf.from_delayed(cudf_result).persist()
        wait(ddf)

        # Wait until the inactive futures are released
        wait([(r.release(), c_r.release()) for r, c_r in zip(cupy_result, cudf_result)])

        if self.properties.renumbered:
            ddf = self.renumber_map.unrenumber(ddf, "vertex")

        return ddf

    def in_degree(
        self, vertex_subset: Union[cudf.Series, dask_cudf.Series, Iterable] = None
    ) -> dask_cudf.DataFrame:
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
        return self.degrees_function(vertex_subset, "in_degree")

    def out_degree(
        self, vertex_subset: Union[cudf.Series, dask_cudf.Series, Iterable] = None
    ) -> dask_cudf.DataFrame:
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
        return self.degrees_function(vertex_subset, "out_degree")

    def degree(
        self, vertex_subset: Union[cudf.Series, dask_cudf.Series, Iterable] = None
    ) -> dask_cudf.DataFrame:
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

        return self.degrees_function(vertex_subset, "degree")

    # FIXME:  vertex_subset could be a DataFrame for multi-column vertices
    def degrees(
        self, vertex_subset: Union[cudf.Series, dask_cudf.Series, Iterable] = None
    ) -> dask_cudf.DataFrame:
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
        return self.degrees_function(vertex_subset, "degrees")

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
        _client = default_client()

        def _call_plc_two_hop_neighbors(sID, mg_graph_x, start_vertices):
            results_ = pylibcugraph_get_two_hop_neighbors(
                resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
                graph=mg_graph_x,
                start_vertices=start_vertices,
                do_expensive_check=False,
            )
            return results_

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

            start_vertices = start_vertices.astype(start_vertices_type)

            def create_iterable_args(
                session_id, input_graph, start_vertices=None, npartitions=None
            ):
                session_id_it = [session_id] * npartitions
                graph_it = input_graph.values()
                start_vertices = cp.array_split(start_vertices.values, npartitions)
                return [
                    session_id_it,
                    graph_it,
                    start_vertices,
                ]

            result = _client.map(
                _call_plc_two_hop_neighbors,
                *create_iterable_args(
                    Comms.get_session_id(),
                    self._plc_graph,
                    start_vertices,
                    self._npartitions,
                ),
                pure=False,
            )

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

        cudf_result = [
            _client.submit(convert_to_cudf, cp_arrays, pure=False)
            for cp_arrays in result
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

    def select_random_vertices(
        self, random_state: int = None, num_vertices: int = None
    ) -> Union[dask_cudf.Series, dask_cudf.DataFrame]:
        """
        Select random vertices from the graph

        Parameters
        ----------
        random_state : int , optional(default=None)
            Random state to use when generating samples.  Optional argument,
            defaults to a hash of process id, time, and hostname.

        num_vertices : int, optional(default=None)
            Number of vertices to sample. If None, all vertices will be selected

        Returns
        -------
        return random vertices from the graph as a dask object
        """

        _client = default_client()

        def convert_to_cudf(cp_arrays: cp.ndarray) -> cudf.Series:
            """
            Creates a cudf Series from cupy arrays
            """
            vertices = cudf.Series(cp_arrays)

            return vertices

        def _call_plc_select_random_vertices(
            mg_graph_x, sID: bytes, random_state: int, num_vertices: int
        ) -> cudf.Series:
            cp_arrays = pylibcugraph_select_random_vertices(
                graph=mg_graph_x,
                resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
                random_state=random_state,
                num_vertices=num_vertices,
            )
            return convert_to_cudf(cp_arrays)

        def _mg_call_plc_select_random_vertices(
            input_graph,
            client: dask.distributed.client.Client,
            sID: bytes,
            random_state: int,
            num_vertices: int,
        ) -> dask_cudf.Series:
            result = [
                client.submit(
                    _call_plc_select_random_vertices,
                    input_graph._plc_graph[w],
                    sID,
                    hash((random_state, i)),
                    num_vertices,
                    workers=[w],
                    allow_other_workers=False,
                    pure=False,
                )
                for i, w in enumerate(Comms.get_workers())
            ]
            ddf = dask_cudf.from_delayed(result, verify_meta=False).persist()
            wait(ddf)
            wait([r.release() for r in result])
            return ddf

        ddf = _mg_call_plc_select_random_vertices(
            self,
            _client,
            Comms.get_session_id(),
            random_state,
            num_vertices,
        )

        if self.properties.renumbered:
            vertices = ddf.rename("vertex").to_frame()
            vertices = self.renumber_map.unrenumber(vertices, "vertex")
            if len(vertices.columns) == 1:
                vertices = vertices["vertex"]
        else:
            vertices = ddf

        return vertices

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

        return self.view_edge_list()[self.vertex_columns]

    def nodes(self):
        """
        Returns all nodes in the graph as a dask_cudf.Series.
        If multi columns vertices, return a dask_cudf.DataFrame.

        If the edgelist was renumbered, this call returns the internal
        nodes in the graph. To get the original nodes, convert the result to
        a dataframe and do 'renumber_map.unrenumber' or 'G.unrenumber'
        """

        if self.edgelist is not None:
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
        else:
            raise RuntimeError("Graph is Empty")

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

            This parameter is deprecated and will be removed.
        """

        if legacy_renum_only:
            warning_msg = (
                "The parameter 'legacy_renum_only' is deprecated and will be removed."
            )
            warnings.warn(warning_msg, DeprecationWarning)

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

            (renumbered_ddf, number_map,) = NumberMap.renumber_and_segment(
                self.input_df,
                self.source_columns,
                self.destination_columns,
                store_transposed=transposed,
                legacy_renum_only=legacy_renum_only,
            )

            self.edgelist = self.EdgeList(renumbered_ddf)
            self.renumber_map = number_map
            self.properties.store_transposed = transposed

    def vertex_column_size(self):
        if self.renumbered:
            return self.renumber_map.vertex_column_size()
        else:
            return 1

    @property
    def _npartitions(self) -> int:
        return len(self._plc_graph)
