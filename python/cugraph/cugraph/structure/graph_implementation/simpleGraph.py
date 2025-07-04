# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
from cugraph.structure.replicate_edgelist import replicate_cudf_dataframe
from cugraph.structure.symmetrize import symmetrize as symmetrize_df
from pylibcugraph import decompress_to_edgelist as pylibcugraph_decompress_to_edgelist
from pylibcugraph import extract_vertex_list as pylibcugraph_extract_vertex_list
from cugraph.structure.number_map import NumberMap
import cugraph.dask.common.mg_utils as mg_utils
import cudf
import dask_cudf
import cugraph.dask.comms.comms as Comms
import pandas as pd
import numpy as np
import warnings
from cugraph.dask.structure import replication
from typing import Union, Dict, Iterable
from pylibcugraph import (
    get_two_hop_neighbors as pylibcugraph_get_two_hop_neighbors,
    select_random_vertices as pylibcugraph_select_random_vertices,
    degrees as pylibcugraph_degrees,
    in_degrees as pylibcugraph_in_degrees,
    out_degrees as pylibcugraph_out_degrees,
)

from pylibcugraph import (
    ResourceHandle,
    GraphProperties,
    SGGraph,
)


# FIXME: Change to consistent camel case naming
class simpleGraphImpl:
    edgeWeightCol = "weights"
    edgeIdCol = "edge_id"
    edgeTypeCol = "edge_type"
    srcCol = "src"
    dstCol = "dst"

    class EdgeList:
        def __init__(
            self,
            source: cudf.Series,
            destination: cudf.Series,
            edge_attr: Union[cudf.DataFrame, Dict[str, cudf.DataFrame]] = None,
        ):
            self.edgelist_df = cudf.DataFrame()
            self.edgelist_df[simpleGraphImpl.srcCol] = source
            self.edgelist_df[simpleGraphImpl.dstCol] = destination
            self.weights = False
            if edge_attr is not None:
                if isinstance(edge_attr, dict):
                    if edge_attr[simpleGraphImpl.edgeWeightCol] is not None:
                        self.weights = True

                    for ea in [
                        simpleGraphImpl.edgeIdCol,
                        simpleGraphImpl.edgeTypeCol,
                        simpleGraphImpl.edgeWeightCol,
                    ]:
                        if edge_attr[ea] is not None:
                            self.edgelist_df[ea] = edge_attr[ea]
                else:
                    self.weights = True
                    self.edgelist_df[simpleGraphImpl.edgeWeightCol] = edge_attr

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
            self.multi_edge = getattr(properties, "multi_edge", False)
            self.directed = properties.directed
            self.renumbered = False
            self.self_loop = None
            self.store_transposed = False
            self.isolated_vertices = None
            self.node_count = None
            self.edge_count = None
            self.weighted = False

    def __init__(self, properties):
        # Structure
        self.edgelist = None
        self.input_df = None
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

        self.source_columns = None
        self.destination_columns = None
        self.vertex_columns = None
        self.weight_column = None

    # Functions
    # FIXME: Change to public function
    # FIXME: Make function more modular
    # edge_attr: None, weight, or (weight, id, type)
    def __from_edgelist(
        self,
        input_df,
        source="source",
        destination="destination",
        edge_attr=None,
        weight=None,
        edge_id=None,
        edge_type=None,
        renumber=True,
        store_transposed=False,
        symmetrize=None,
        vertices=None,
    ):

        if self.properties.directed and symmetrize:
            raise ValueError(
                "The edgelist can only be symmetrized for undirected graphs."
            )

        if self.properties.directed:
            if symmetrize:
                raise ValueError(
                    "The edgelist can only be symmetrized for undirected graphs."
                )
        else:
            if symmetrize or symmetrize is None:
                unsupported = False
                if edge_id is not None or edge_type is not None:
                    unsupported = True
                if isinstance(edge_attr, list):
                    if len(edge_attr) > 1:
                        unsupported = True
                if unsupported:
                    raise ValueError(
                        "Edge list containing Edge Ids or Types can't be symmetrized. "
                        "If the edges are already symmetric, set the 'symmetrize' "
                        "flag to False"
                    )

        if symmetrize is None:
            # default behavior
            symmetrize = not self.properties.directed

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
        df_columns = s_col + d_col
        self.vertex_columns = df_columns.copy()

        if edge_attr is not None:
            if weight is not None or edge_id is not None or edge_type is not None:
                raise ValueError(
                    "If specifying edge_attr, cannot specify weight/edge_id/edge_type"
                )
            if isinstance(edge_attr, str):
                weight = edge_attr
                edge_attr = [weight]
            if not (set(edge_attr).issubset(set(input_df.columns))):
                raise ValueError(
                    f"edge_attr column {edge_attr} not found in input."
                    "Recheck the edge_attr parameter"
                )
            self.properties.weighted = True

            if len(edge_attr) != 1 and len(edge_attr) != 3:
                raise ValueError(
                    f"Invalid number of edge attributes " f"passed. {edge_attr}"
                )

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

                weight, edge_id, edge_type = edge_attr
        else:
            edge_attr = []
            if weight is not None:
                edge_attr.append(weight)
                self.properties.weighted = True
            if edge_id is not None:
                edge_attr.append(edge_id)
            if edge_type is not None:
                edge_attr.append(edge_type)

        df_columns += edge_attr
        input_df = input_df[df_columns]
        # FIXME: check if the consolidated graph fits on the
        # device before gathering all the edge lists

        # Consolidation
        if isinstance(input_df, cudf.DataFrame):
            if len(input_df[source]) > 2147483100:
                raise ValueError(
                    "cudf dataFrame edge list is too big to fit in a single GPU"
                )
            elist = input_df
        elif isinstance(input_df, dask_cudf.DataFrame):
            if len(input_df[source]) > 2147483100:
                raise ValueError(
                    "dask_cudf dataFrame edge list is too big to fit in a single GPU"
                )
            elist = input_df.compute().reset_index(drop=True)
        else:
            raise TypeError("input should be a cudf.DataFrame or a dask_cudf dataFrame")
        # initial, unmodified input dataframe.
        self.input_df = elist
        self.weight_column = weight
        self.source_columns = source
        self.destination_columns = destination

        # Renumbering
        self.renumber_map = None
        self.store_transposed = store_transposed
        if renumber:
            # FIXME: Should SG do lazy evaluation like MG?
            elist, renumber_map = NumberMap.renumber(
                elist, source, destination, store_transposed=False
            )
            source = renumber_map.renumbered_src_col_name
            destination = renumber_map.renumbered_dst_col_name
            # Use renumber_map to figure out if the python renumbering occured
            self.properties.renumbered = renumber_map.is_renumbered
            self.renumber_map = renumber_map
            self.renumber_map.implementation.src_col_names = simpleGraphImpl.srcCol
            self.renumber_map.implementation.dst_col_names = simpleGraphImpl.dstCol
        else:
            if type(source) is list and type(destination) is list:
                raise ValueError("set renumber to True for multi column ids")
            elif elist[source].dtype not in [np.int32, np.int64] or elist[
                destination
            ].dtype not in [np.int32, np.int64]:
                if elist[destination].dtype in [np.uint32, np.uint64] or elist[
                    source
                ].dtype in [np.uint32, np.uint64]:
                    raise ValueError(
                        "Unsigned integers are not supported as vertex ids."
                        "  Either convert to signed integers or set renumber=True"
                    )
                raise ValueError("set renumber to True for non integer columns ids")

        # The dataframe will be symmetrized iff the graph is undirected with the
        # symmetrize flag set to None or True otherwise, the inital dataframe will
        # be returned. If set to False, the API will assume that the edges are already
        # symmetric. Duplicated edges will be dropped unless the graph is a
        # MultiGraph(Not Implemented yet)

        if edge_attr is not None:
            value_col = {
                self.edgeWeightCol: elist[weight] if weight in edge_attr else None,
                self.edgeIdCol: elist[edge_id] if edge_id in edge_attr else None,
                self.edgeTypeCol: elist[edge_type] if edge_type in edge_attr else None,
            }

        else:
            value_col = None

        # FIXME: if the user calls self.edgelist.edgelist_df after creating a
        # symmetric graph, return the symmetric edgelist?
        # FIXME: For better memory footprint, avoid storing this edgelist and instead
        # call decompress_to_edgelist to extract the edgelist from the graph
        self.edgelist = simpleGraphImpl.EdgeList(
            elist[source], elist[destination], value_col
        )

        if self.batch_enabled:
            self._replicate_edgelist()

        if vertices is not None:
            if self.properties.renumbered is True:
                if isinstance(vertices, cudf.Series):
                    vertices = self.renumber_map.to_internal_vertex_id(vertices)
                else:
                    vertices = self.lookup_internal_vertex_id(cudf.Series(vertices))

            if not isinstance(vertices, cudf.Series):
                vertex_dtype = self.edgelist.edgelist_df[simpleGraphImpl.srcCol].dtype
                vertices = cudf.Series(vertices, dtype=vertex_dtype)

        self._make_plc_graph(
            value_col=value_col,
            store_transposed=store_transposed,
            renumber=renumber,
            drop_multi_edges=not self.properties.multi_edge,
            symmetrize=symmetrize,
            vertices=vertices,
        )

    def to_pandas_edgelist(
        self,
        source="src",
        destination="dst",
        weight="weight",
    ):
        """
        Returns the graph edge list as a Pandas DataFrame.

        Parameters
        ----------
        source : str or array-like, optional (default='src')
            source column name or array of column names
        destination : str or array-like, optional (default='dst')
            destination column name or array of column names
        weight : str or array-like, optional (default='weight')
            weight column name or array of column names

        Returns
        -------
        df : pandas.DataFrame
        """

        gdf = self.view_edge_list()
        if self.properties.weighted:
            gdf.rename(
                columns={
                    simpleGraphImpl.srcCol: source,
                    simpleGraphImpl.dstCol: destination,
                    "weight": weight,
                },
                inplace=True,
            )
        else:
            gdf.rename(
                columns={
                    simpleGraphImpl.srcCol: source,
                    simpleGraphImpl.dstCol: destination,
                },
                inplace=True,
            )
        return gdf.to_pandas()

    def to_pandas_adjacency(self):
        """
        Returns the graph adjacency matrix as a Pandas DataFrame.
        """

        np_array_data = self.to_numpy_array()
        pdf = pd.DataFrame(np_array_data)

        nodes = self.nodes().values_host.tolist()
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
        nodes = self.nodes()
        for i in range(0, elen):
            # Map vertices to consecutive integers
            idx_src = nodes[nodes == df[simpleGraphImpl.srcCol].iloc[i]].index[0]
            idx_dst = nodes[nodes == df[simpleGraphImpl.dstCol].iloc[i]].index[0]
            np_array[idx_src, idx_dst] = df[self.edgeWeightCol].iloc[i]
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

            df[weight] : cudf.Series
                Column is only present for weighted Graph,
                then containing the weight value for each edge
        """
        if self.edgelist is None:
            # The graph must have an adjacency list or else the call below will fail
            src, dst, weights = graph_primtypes_wrapper.view_edge_list(self)
            self.edgelist = self.EdgeList(src, dst, weights)

        srcCol = self.source_columns
        dstCol = self.destination_columns
        """
        Only use the initial input dataframe  if the graph is directed with:
            1) single vertex column names with integer vertex type
            2) list of vertex column names of size 1 with integer vertex type
        """
        use_initial_input_df = True

        # Retrieve the renumbered edgelist if the upper triangular matrix
        # needs to be extracted otherwised, retrieve the unrenumbered version.
        # Only undirected graphs return the upper triangular matrix when calling
        # the 'view_edge_list' method.
        return_unrenumbered_edgelist = True

        if self.input_df is not None:
            if type(srcCol) is list and type(dstCol) is list:
                if len(srcCol) == 1:
                    srcCol = srcCol[0]
                    dstCol = dstCol[0]
                    if self.input_df[srcCol].dtype not in [
                        np.int32,
                        np.int64,
                    ] or self.input_df[dstCol].dtype not in [np.int32, np.int64]:
                        # hypergraph case
                        use_initial_input_df = False
                        return_unrenumbered_edgelist = False
                else:
                    use_initial_input_df = False
                    return_unrenumbered_edgelist = False

            elif self.input_df[srcCol].dtype not in [
                np.int32,
                np.int64,
            ] or self.input_df[dstCol].dtype not in [np.int32, np.int64]:
                use_initial_input_df = False
                return_unrenumbered_edgelist = False
        else:
            use_initial_input_df = False
            return_unrenumbered_edgelist = False

        if self.properties.directed:
            # If the graph is directed, no need for the renumbered edgelist
            # to extract the upper triangular matrix
            return_unrenumbered_edgelist = True

        if use_initial_input_df and self.properties.directed:
            edgelist_df = self.input_df  # Original input.
        else:
            edgelist_df = self.decompress_to_edgelist(
                return_unrenumbered_edgelist=return_unrenumbered_edgelist
            )

            if self.properties.renumbered:
                edgelist_df = edgelist_df.rename(
                    columns=self.renumber_map.internal_to_external_col_names
                )

            if srcCol is None and dstCol is None:
                srcCol = simpleGraphImpl.srcCol
                dstCol = simpleGraphImpl.dstCol

        if use_initial_input_df and not self.properties.directed:
            # unrenumber before extracting the upper triangular part
            # case when the vertex column name is of size 1
            if self.properties.renumbered:
                edgelist_df = edgelist_df.rename(
                    columns=self.renumber_map.internal_to_external_col_names
                )
                # extract the upper triangular part
                edgelist_df = edgelist_df[edgelist_df[srcCol] <= edgelist_df[dstCol]]
            else:
                edgelist_df = edgelist_df[
                    edgelist_df[simpleGraphImpl.srcCol]
                    <= edgelist_df[simpleGraphImpl.dstCol]
                ]

        elif not use_initial_input_df and self.properties.renumbered:
            # Do not unrenumber the vertices if the initial input df was used
            if not self.properties.directed:

                edgelist_df = self.decompress_to_edgelist(
                    return_unrenumbered_edgelist=return_unrenumbered_edgelist
                )

                # Need to leverage the renumbered edgelist to extract the upper
                # triangular matrix for multi-column or string vertices
                edgelist_df = edgelist_df[
                    edgelist_df[simpleGraphImpl.srcCol]
                    <= edgelist_df[simpleGraphImpl.dstCol]
                ]

                # unrenumber the edgelist
                edgelist_df = self.renumber_map.unrenumber(
                    edgelist_df, simpleGraphImpl.srcCol
                )
                edgelist_df = self.renumber_map.unrenumber(
                    edgelist_df, simpleGraphImpl.dstCol
                )
                edgelist_df = edgelist_df.rename(
                    columns=self.renumber_map.internal_to_external_col_names
                )

        if self.vertex_columns is not None and len(self.vertex_columns) == 2:
            # single column vertices internally renamed to 'simpleGraphImpl.srcCol'
            # and 'simpleGraphImpl.dstCol'.
            if not set(self.vertex_columns).issubset(set(edgelist_df.columns)):
                # Get the initial column names passed by the user.
                if srcCol is not None and dstCol is not None:
                    edgelist_df = edgelist_df.rename(
                        columns={
                            simpleGraphImpl.srcCol: srcCol,
                            simpleGraphImpl.dstCol: dstCol,
                        }
                    )

        # FIXME: When renumbered, the MG API uses renumbered col names which
        # is not consistant with the SG API.

        self.properties.edge_count = len(edgelist_df)

        wgtCol = simpleGraphImpl.edgeWeightCol
        edgelist_df = edgelist_df.rename(
            columns={wgtCol: self.weight_column}
        ).reset_index(drop=True)

        return edgelist_df

    def delete_edge_list(self):
        """
        Delete the edge list.
        """
        # decrease reference count to free memory if the referenced objects are
        # no longer used.
        self.edgelist = None

    def __from_adjlist(
        self,
        offset_col,
        index_col,
        value_col=None,
        renumber=True,
        store_transposed=False,
        symmetrize=None,
    ):

        self.adjlist = simpleGraphImpl.AdjList(offset_col, index_col, value_col)

        if self.properties.directed and symmetrize:
            raise ValueError("The edges can only be symmetrized for undirected graphs.")

        if value_col is not None:
            self.properties.weighted = True
        self._make_plc_graph(
            value_col=value_col,
            store_transposed=store_transposed,
            renumber=renumber,
            symmetrize=not self.properties.directed
            if symmetrize is None
            else symmetrize,
        )

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
            if self.transposedadjlist is not None and self.properties.directed is False:
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
            raise RuntimeError(
                "MG Batch needs a Dask Client and the "
                "Communicator needs to be initialized."
            )

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

        # FIXME: There  might be a better way to control it
        if client is None:
            return

        self.batch_edgelists = replicate_cudf_dataframe(self.edgelist.edgelist_df)

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

    def get_two_hop_neighbors(self, start_vertices=None):
        """
        Compute vertex pairs that are two hops apart. The resulting pairs are
        sorted before returning.

        Parameters
        ----------
        start_vertices : Int or List (default=None)
        Subset of vertices to compute two hop neighbors on. If None, compute
        for all nodes.

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

        if self.properties.renumbered is True:
            if start_vertices is not None:
                start_vertices = self.renumber_map.to_internal_vertex_id(start_vertices)
                start_vertices_type = self.edgelist.edgelist_df["src"].dtype
                start_vertices = start_vertices.astype(start_vertices_type)
        do_expensive_check = False
        first, second = pylibcugraph_get_two_hop_neighbors(
            resource_handle=ResourceHandle(),
            graph=self._plc_graph,
            start_vertices=start_vertices,
            do_expensive_check=do_expensive_check,
        )

        df = cudf.DataFrame()
        df["first"] = first
        df["second"] = second

        if self.properties.renumbered is True:
            df = self.renumber_map.unrenumber(df, "first")
            df = self.renumber_map.unrenumber(df, "second")

        return df

    def decompress_to_edgelist(
        self, return_unrenumbered_edgelist: bool = True
    ) -> cudf.DataFrame:
        """
        Extract a the edgelist from a graph.

        Parameters
        ----------
        return_unrenumbered_edgelist : bool (default=True)
            Flag determining whether to return the original input edgelist
            if 'True' or the renumbered one if 'False' and the edgelist was
            renumbered.

        Returns
        -------

        df : cudf.DataFrame
            GPU data frame containing all sources identifiers,
            destination identifiers and if applicable edge weights, edge ids and
            edge types

        Examples
        --------
        >>> from cugraph.datasets import karate
        >>> G = karate.get_graph(download=True)
        >>> edgelist = G.decompress_to_edgelist()

        """

        do_expensive_check = False
        (
            source,
            destination,
            weight,
            edge_ids,
            edge_type_ids,
        ) = pylibcugraph_decompress_to_edgelist(
            resource_handle=ResourceHandle(),
            graph=self._plc_graph,
            do_expensive_check=do_expensive_check,
        )

        df = cudf.DataFrame()
        df["src"] = source
        df["dst"] = destination
        if weight is not None:
            df["weight"] = weight
        if edge_ids is not None:
            df["edge_ids"] = edge_ids
        if edge_type_ids is not None:
            df["edge_type_ids"] = edge_type_ids

        if self.properties.renumbered and return_unrenumbered_edgelist:
            df, _ = self.renumber_map.unrenumber(df, "src", get_column_names=True)
            df, _ = self.renumber_map.unrenumber(df, "dst", get_column_names=True)

        return df

    def extract_vertex_list(
        self, return_unrenumbered_vertices: bool = True
    ) -> cudf.DataFrame:
        """
        Extract the vertices from a graph.

        Parameters
        ----------
        return_unrenumbered_vertices : bool (default=True)
            Flag determining whether to return the original input input vertices
            if 'True' or the renumbered one if 'False' and the edgelist was
            renumbered.

        Returns
        -------

        series : cudf.Series
            GPU Series containing all the vertices in the graph including
            isolated vertices.

        Examples
        --------
        >>> from cugraph.datasets import karate
        >>> G = karate.get_graph(download=True)
        >>> vertices = G.extract_vertex_list()

        """

        do_expensive_check = False
        vertices = pylibcugraph_extract_vertex_list(
            resource_handle=ResourceHandle(),
            graph=self._plc_graph,
            do_expensive_check=do_expensive_check,
        )

        vertices = cudf.Series(
            vertices, dtype=self.edgelist.edgelist_df[simpleGraphImpl.srcCol].dtype
        )

        if self.properties.renumbered and return_unrenumbered_vertices:
            df_ = cudf.DataFrame()
            df_["vertex"] = vertices
            df_ = self.renumber_map.unrenumber(df_, "vertex")
            if len(df_.columns) > 1:
                vertices = df_
            else:
                vertices = df_["vertex"]

        return vertices.sort_values(ignore_index=True)

    def select_random_vertices(
        self,
        random_state: int = None,
        num_vertices: int = None,
    ) -> Union[cudf.Series, cudf.DataFrame]:
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
        return random vertices from the graph as a cudf
        """
        vertices = pylibcugraph_select_random_vertices(
            resource_handle=ResourceHandle(),
            graph=self._plc_graph,
            random_state=random_state,
            num_vertices=num_vertices,
        )

        vertices = cudf.Series(vertices)
        if self.properties.renumbered is True:
            df_ = cudf.DataFrame()
            df_["vertex"] = vertices
            df_ = self.renumber_map.unrenumber(df_, "vertex")
            if len(df_.columns) > 1:
                vertices = df_
            else:
                vertices = df_["vertex"]

        return vertices

    def number_of_vertices(self):
        """
        Get the number of nodes in the graph.
        """
        if self.properties.node_count is None:
            if self.adjlist is not None:
                self.properties.node_count = len(self.adjlist.offsets) - 1
            elif self.transposedadjlist is not None:
                self.properties.node_count = len(self.transposedadjlist.offsets) - 1
            elif self.edgelist is not None:
                self.properties.node_count = len(self.nodes())
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
            return len(self.decompress_to_edgelist(return_unrenumbered_edgelist=False))
        if self.properties.edge_count is None:
            if self.edgelist is not None:
                edgelist_df = self.decompress_to_edgelist()
                if self.properties.directed is False:
                    self.properties.edge_count = len(
                        edgelist_df[
                            edgelist_df[simpleGraphImpl.srcCol]
                            >= edgelist_df[simpleGraphImpl.dstCol]
                        ]
                    )
                else:
                    self.properties.edge_count = len(edgelist_df)
            elif self.adjlist is not None:
                self.properties.edge_count = len(self.adjlist.indices)
            elif self.transposedadjlist is not None:
                self.properties.edge_count = len(self.transposedadjlist.indices)
            else:
                raise ValueError("Graph is Empty")
        return self.properties.edge_count

    def degrees_function(
        self,
        vertex_subset: Union[cudf.Series, Iterable] = None,
        degree_type: str = "in_degree",
    ) -> cudf.DataFrame:
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
        vertex_subset : cudf.Series or iterable container, optional
            A container of vertices for displaying corresponding in-degree.
            If not set, degrees are computed for the entire set of vertices.

        degree_type : str (default='in_degree')

        Returns
        -------
        df : cudf.DataFrame
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
        if vertex_subset is not None:
            if not isinstance(vertex_subset, cudf.Series):
                vertex_subset = cudf.Series(vertex_subset)
                if self.properties.renumbered is True:
                    vertex_subset = self.renumber_map.to_internal_vertex_id(
                        vertex_subset
                    )
                    vertex_subset_type = self.edgelist.edgelist_df.dtypes.iloc[0]
                else:
                    vertex_subset_type = self.input_df.dtypes.iloc[0]

                vertex_subset = vertex_subset.astype(vertex_subset_type)

        do_expensive_check = False
        df = cudf.DataFrame()
        vertex = None

        if degree_type == "in_degree":
            vertex, in_degrees = pylibcugraph_in_degrees(
                resource_handle=ResourceHandle(),
                graph=self._plc_graph,
                source_vertices=vertex_subset,
                do_expensive_check=do_expensive_check,
            )
            df["degree"] = in_degrees
        elif degree_type == "out_degree":
            vertex, out_degrees = pylibcugraph_out_degrees(
                resource_handle=ResourceHandle(),
                graph=self._plc_graph,
                source_vertices=vertex_subset,
                do_expensive_check=do_expensive_check,
            )
            df["degree"] = out_degrees
        elif degree_type in ["degree", "degrees"]:
            vertex, in_degrees, out_degrees = pylibcugraph_degrees(
                resource_handle=ResourceHandle(),
                graph=self._plc_graph,
                source_vertices=vertex_subset,
                do_expensive_check=do_expensive_check,
            )
            if degree_type == "degrees":
                df["in_degree"] = in_degrees
                df["out_degree"] = out_degrees

            else:
                df["degree"] = in_degrees + out_degrees
        else:
            raise ValueError(
                "Incorrect degree type passed, valid values are ",
                "'in_degree', 'out_degree', 'degree' and 'degrees' ",
                f"got '{degree_type}'",
            )
        df["vertex"] = vertex
        if self.properties.renumbered is True:
            df = self.renumber_map.unrenumber(df, "vertex")

        return df

    def in_degree(
        self, vertex_subset: Union[cudf.Series, Iterable] = None
    ) -> cudf.DataFrame:
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
        return self.degrees_function(vertex_subset, "in_degree")

    def out_degree(
        self, vertex_subset: Union[cudf.Series, Iterable] = None
    ) -> cudf.DataFrame:
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
        return self.degrees_function(vertex_subset, "out_degree")

    def degree(
        self, vertex_subset: Union[cudf.Series, Iterable] = None
    ) -> cudf.DataFrame:
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
        return self.degrees_function(vertex_subset, "degree")

    # FIXME:  vertex_subset could be a DataFrame for multi-column vertices
    def degrees(
        self, vertex_subset: Union[cudf.Series, Iterable] = None
    ) -> cudf.DataFrame:
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
        return self.degrees_function(vertex_subset, "degrees")

    def _make_plc_graph(
        self,
        value_col: Dict[str, cudf.DataFrame] = None,
        store_transposed: bool = False,
        renumber: bool = True,
        drop_multi_edges: bool = False,
        symmetrize: bool = False,
        vertices: cudf.Series = None,
    ):

        """
        Parameters
        ----------
        value_col : cudf.DataFrame or dict[str, cudf.DataFrame]
            If a single dataframe is provided, this is assumed
            to contain the edge weight values.
            If a dictionary of dataframes is provided, then it is
            assumed to contain edge properties.
        store_transposed : bool (default=False)
            Whether to store the graph in a transposed
            format.  Required by some algorithms.
        renumber : bool (default=True)
            Whether to renumber the vertices of the graph.
            Required if inputted vertex ids are not of
            int32 or int64 type.
        drop_multi_edges: bool (default=False)
            Whether to drop multi edges
        symmetrize: bool (default=False)
            Whether to symmetrize
        vertices: cudf.Series = None
            vertices in the graph
        """

        if value_col is None:
            weight_col, id_col, type_col = None, None, None
        elif isinstance(value_col, (cudf.DataFrame, cudf.Series)):
            weight_col, id_col, type_col = value_col, None, None
        elif isinstance(value_col, dict):
            weight_col = value_col[self.edgeWeightCol]
            id_col = value_col[self.edgeIdCol]
            type_col = value_col[self.edgeTypeCol]
        else:
            raise ValueError(f"Illegal value col {type(value_col)}")

        graph_props = GraphProperties(
            is_multigraph=self.properties.multi_edge,
            is_symmetric=not self.properties.directed,
        )

        if self.edgelist is not None:
            input_array_format = "COO"
            src_or_offset_array = self.edgelist.edgelist_df[simpleGraphImpl.srcCol]
            dst_or_index_array = self.edgelist.edgelist_df[simpleGraphImpl.dstCol]

        elif self.adjlist is not None:
            input_array_format = "CSR"
            src_or_offset_array = self.adjlist.offsets
            dst_or_index_array = self.adjlist.indices

        else:
            raise TypeError(
                "Edges need to be represented in either in COO or CSR format."
            )

        if weight_col is not None:
            weight_t = weight_col.dtype

            if weight_t == "int32":
                weight_col = weight_col.astype("float32")
            if weight_t == "int64":
                weight_col = weight_col.astype("float64")

        if id_col is not None:
            if src_or_offset_array.dtype == "int64" and id_col.dtype != "int64":
                id_col = id_col.astype("int64")
                warnings.warn(
                    f"Vertex type is int64 but edge id type is {id_col.dtype}"
                    ", automatically casting edge id type to int64. "
                    "This may cause extra memory usage.  Consider passing"
                    " a int64 list of edge ids instead."
                )

        self._plc_graph = SGGraph(
            resource_handle=ResourceHandle(),
            graph_properties=graph_props,
            src_or_offset_array=src_or_offset_array,
            dst_or_index_array=dst_or_index_array,
            weight_array=weight_col,
            edge_id_array=id_col,
            edge_type_array=type_col,
            store_transposed=store_transposed,
            renumber=renumber,
            do_expensive_check=True,
            input_array_format=input_array_format,
            vertices_array=vertices,
            drop_multi_edges=drop_multi_edges,
            symmetrize=symmetrize,
        )

    def to_directed(self, DiG, store_transposed=False):
        """
        Return a directed representation of the graph Implementation.
        This function copies the internal structures and returns the
        directed view.

        Note: this will discard any edge ids or edge types but will
        preserve edge weights if present.
        """
        DiG.properties.renumbered = self.properties.renumbered
        DiG.renumber_map = self.renumber_map
        DiG.edgelist = self.edgelist
        DiG.adjlist = self.adjlist
        DiG.transposedadjlist = self.transposedadjlist

        if simpleGraphImpl.edgeWeightCol in self.edgelist.edgelist_df:
            value_col = self.edgelist.edgelist_df[simpleGraphImpl.edgeWeightCol]
        else:
            value_col = None

        DiG._make_plc_graph(value_col, store_transposed)

    def to_undirected(self, G, store_transposed=False):

        """
        Return an undirected copy of the graph.

        Note: This will discard any edge ids or edge types but will
        preserve edge weights if present.
        """
        # FIXME: Update this function to not call the deprecated
        # symmetrize function.
        #   1) Import the C++ function that symmetrize a graph
        #   2) decompress the edgelist to update 'simpleGraphImpl.EdgeList'
        # Doesn't work for edgelists with edge_ids and edge_types.
        G.properties.renumbered = self.properties.renumbered
        G.renumber_map = self.renumber_map
        if self.properties.directed is False:
            G.edgelist = self.edgelist
            G.adjlist = self.adjlist
            G.transposedadjlist = self.transposedadjlist
        else:
            df = self.edgelist.edgelist_df
            if self.edgelist.weights:
                source_col, dest_col, value_col = symmetrize_df(
                    df,
                    simpleGraphImpl.srcCol,
                    simpleGraphImpl.dstCol,
                    simpleGraphImpl.edgeWeightCol,
                )
            else:
                source_col, dest_col = symmetrize_df(
                    df, simpleGraphImpl.srcCol, simpleGraphImpl.dstCol
                )
                value_col = None
            G.edgelist = simpleGraphImpl.EdgeList(source_col, dest_col, value_col)

        if simpleGraphImpl.edgeWeightCol in self.edgelist.edgelist_df:
            value_col = self.edgelist.edgelist_df[simpleGraphImpl.edgeWeightCol]
        else:
            value_col = None

        G._make_plc_graph(value_col, store_transposed)

    def has_node(self, n):
        """
        Returns True if the graph contains the node n.
        """

        return (self.nodes() == n).any().any()

    def has_edge(self, u, v):
        """
        Returns True if the graph contains the edge (u,v).
        """
        if self.properties.renumbered:
            tmp = cudf.DataFrame({simpleGraphImpl.srcCol: [u, v]})
            tmp = tmp.astype({simpleGraphImpl.srcCol: "int"})
            tmp = self.renumber_map.add_internal_vertex_id(
                tmp, "id", simpleGraphImpl.srcCol, preserve_order=True
            )

            u = tmp["id"][0]
            v = tmp["id"][1]

        df = self.edgelist.edgelist_df

        if self.edgelist.weights:
            # FIXME: Update this function to not call the deprecated
            # symmetrize function.
            source_col, dest_col, value_col = symmetrize_df(
                df,
                simpleGraphImpl.srcCol,
                simpleGraphImpl.dstCol,
                simpleGraphImpl.edgeWeightCol,
                symmetrize=not self.properties.directed,
            )
        else:
            source_col, dest_col = symmetrize_df(
                df,
                simpleGraphImpl.srcCol,
                simpleGraphImpl.dstCol,
                symmetrize=not self.properties.directed,
            )
            value_col = None

        self.edgelist = simpleGraphImpl.EdgeList(source_col, dest_col, value_col)

        return (
            (df[simpleGraphImpl.srcCol] == u) & (df[simpleGraphImpl.dstCol] == v)
        ).any()

    def has_self_loop(self):
        """
        Returns True if the graph has self loop.
        """
        # Detect self loop
        if self.properties.self_loop is None:
            elist = self.edgelist.edgelist_df
            if (elist[simpleGraphImpl.srcCol] == elist[simpleGraphImpl.dstCol]).any():
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
        return self.view_edge_list()[self.vertex_columns]

    def nodes(self):
        """
        Returns all the nodes in the graph as a cudf.Series, in order of appearance
        in the edgelist (source column first, then destination column).
        If multi columns vertices, return a cudf.DataFrame.
        """
        if self.edgelist is not None:
            return self.extract_vertex_list(return_unrenumbered_vertices=False)
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
        neighbors = df[df[simpleGraphImpl.srcCol] == n][
            simpleGraphImpl.dstCol
        ].reset_index(drop=True)
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
