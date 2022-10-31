# Copyright (c) 2022, NVIDIA CORPORATION.
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

from collections import defaultdict

import cudf
import dask_cudf
import cugraph
from cugraph.experimental import PropertyGraph, MGPropertyGraph
from functools import cached_property

from .utils.add_data import _update_feature_map
from .utils.sampling import sample_multiple_sgs, sample_single_sg
from .utils.sampling import (
    get_subgraph_and_src_range_from_edgelist,
    get_underlying_dtype_from_sg,
)
from .utils.sampling import create_dlpack_d
from .feature_storage import CuFeatureStorage


src_n = PropertyGraph.src_col_name
dst_n = PropertyGraph.dst_col_name
type_n = PropertyGraph.type_col_name
eid_n = PropertyGraph.edge_id_col_name
vid_n = PropertyGraph.vertex_col_name


class CuGraphStore:
    """
    A wrapper around a cuGraph Property Graph that
    then adds functions to basically match the DGL GraphStorage API.
    This is not a full duck-types match to a DGL GraphStore.

    This class return dlpack types and has additional functional arguments.
    """

    def __init__(self, graph, backend_lib="torch"):
        """
        Parameters
        ----------
        graph : PropertyGraph or MGPropertyGraph
            Contains nodes, edges and their properties
        backend_lib : string (default="torch")
            Contains the supported backend to use. Can be one of torch, tf,
            or cupy.

        """
        if isinstance(graph, (PropertyGraph, MGPropertyGraph)):
            self.__G = graph
        else:
            raise ValueError("graph must be a PropertyGraph or MGPropertyGraph")
        # dict to map column names corresponding to edge features
        # of each type
        self.edata_feat_col_d = defaultdict(list)
        # dict to map column names corresponding to node features
        # of each type
        self.ndata_feat_col_d = defaultdict(list)
        self.backend_lib = backend_lib

    def add_node_data(
        self,
        df,
        node_col_name,
        ntype=None,
        feat_name=None,
        contains_vector_features=False,
    ):
        """
        Add a dataframe describing node properties to the PropertyGraph.

        Parameters
        ----------
        dataframe : DataFrame-compatible instance
            A DataFrame instance with a compatible Pandas-like DataFrame
            interface.
        node_col_name : string
            The column name that contains the values to be used as vertex IDs.
        ntype : string
            The node type to be added.
            For example, if dataframe contains data about users, ntype
            might be "users".
            If not specified, the type of properties will be added as
            an empty string.
        feat_name : {} or string
            A map of feature names under which we should save the added
            properties like {"feat_1":[f1, f2], "feat_2":[f3, f4]}
            (ignored if contains_vector_features=False and the col names of
            the dataframe are treated as corresponding feature names)
        contains_vector_features : False
            Whether to treat the columns of the dataframe being added as
            as 2d features
        Returns
        -------
        None
        >>> import cudf
        >>> from cugraph.experimental import PropertyGraph
        >>> from cugraph.gnn import CuGraphStore
        >>> pg = PropertyGraph()
        >>> gs = CuGraphStore(pg, backend_lib="cupy")
        >>> df = cudf.DataFrame()
        >>> df["node_id"] = [1, 2, 3]
        >>> df["node_scaler_feat_1"] = [10, 20, 30]
        >>> df["node_scaler_feat_2"] = [15, 25, 35]
        >>> gs.add_node_data(df, "node_id", contains_vector_features=False)
        >>> print(gs.get_node_storage("node_scaler_feat_1").fetch([1, 3]))
        [10 30]
        """
        self.gdata.add_vertex_data(df, vertex_col_name=node_col_name, type_name=ntype)
        columns = [col for col in list(df.columns) if col != node_col_name]

        _update_feature_map(
            self.ndata_feat_col_d, feat_name, contains_vector_features, columns
        )
        # Clear properties if set as data has changed

        self.__clear_cached_properties()

    def add_edge_data(
        self,
        df,
        node_col_names,
        canonical_etype=None,
        feat_name=None,
        contains_vector_features=False,
    ):
        """
        Add a dataframe describing edge properties to the PropertyGraph.

        Parameters
        ----------
        dataframe : DataFrame-compatible instance
            A DataFrame instance with a compatible Pandas-like DataFrame
            interface.
        node_col_names : string
            The column names that contain the values to be used as the source
            and destination vertex IDs for the edges.
        canonical_etype : string
            The edge type to be added. This should follow the string format
            '(src_type),(edge_type),(dst_type)'
            If not specified, the type of properties will be added as
            an empty string.
        feat_name : string or dict {}
            The feature name under which we should save the added properties
            (ignored if contains_vector_features=False and the col names of
            the dataframe are treated as corresponding feature names)
        contains_vector_features : False
            Whether to treat the columns of the dataframe being added as
            as 2d features
        Returns
        -------
        None

        Examples
        --------
        >>> import cugraph
        >>> import cudf
        >>> import numpy as np
        >>> from cugraph.experimental import PropertyGraph
        >>> from cugraph.gnn import CuGraphStore
        >>> src_ser = cudf.Series([1, 1, 1, 1, 1, 2, 2, 3])
        >>> dst_ser = cudf.Series([2, 3, 4, 5, 6, 3, 4, 7])
        >>> df = cudf.DataFrame(
        ...     {"src": src_ser, "dst": dst_ser, "edge_id": np.arange(len(src_ser))})
        >>> pg = PropertyGraph()
        >>> gs = CuGraphStore(pg)
        >>> gs.add_edge_data(df, ["src", "dst"])
        >>> gs.num_edges()
        8
        """
        self.gdata.add_edge_data(
            df, vertex_col_names=node_col_names, type_name=canonical_etype
        )
        columns = [col for col in list(df.columns) if col not in node_col_names]
        _update_feature_map(
            self.edata_feat_col_d, feat_name, contains_vector_features, columns
        )

        # Clear properties if set as data has changed
        self.__clear_cached_properties()

    def get_node_storage(self, key, ntype=None, indices_offset=0):
        """
        Parameters
        ----------
        key : str
            Name of the edge property to return
        ntype : str, optional
            The type of nodes to return. Required
            if there is more than one edge type.
        indices_offset : int (default=0)
            Starting value for the contiguous node id
            values.gi

        Examples
        --------
        >>> import cudf
        >>> from cugraph.experimental import PropertyGraph
        >>> from cugraph.gnn import CuGraphStore
        >>> pg = PropertyGraph()
        >>> gs = CuGraphStore(pg, backend_lib="cupy")
        >>> df = cudf.DataFrame()
        >>> df["node_id"] = [1, 2, 3]
        >>> df["node_scaler_feat_1"] = [10, 20, 30]
        >>> df["node_scaler_feat_2"] = [15, 25, 35]
        >>> gs.add_node_data(df, "node_id", contains_vector_features=False)
        >>> print(gs.get_node_storage("node_scaler_feat_1").fetch([1, 3]))
        [10 30]
        """
        if ntype is None:
            ntypes = self.ntypes
            if len(self.ntypes) > 1:
                raise ValueError(
                    (
                        "Node type name must be specified if there "
                        "are more than one node types."
                    )
                )
            ntype = ntypes[0]
        if key not in self.ndata_feat_col_d:
            raise ValueError(
                f"key {key} not found in CuGraphStore node features",
                f" {list(self.ndata_feat_col_d.keys())}",
            )

        columns = self.ndata_feat_col_d[key]

        return CuFeatureStorage(
            pg=self.gdata,
            columns=columns,
            storage_type="node",
            indices_offset=indices_offset,
            backend_lib=self.backend_lib,
        )

    def get_edge_storage(self, key, etype=None, indices_offset=0):
        """
        Gets the property data from edges in the graph

        Parameters
        ----------
        key : str
            Name of the edge property to return
        etype : str, optional
            The type of the edge to return. Required
            if there is more than one edge type.
        indices_offset : int (default=0)
            Starting value for the contiguous edge id
            values.

        Returns
        -------
        CuFeatureStorage
            Storage of an edge column feature in the existing
            background lib.
        Examples
        --------
        CuFeatureStorage
            Storage containing the edge property of the edge
            type specified.
        """
        if etype is None:
            etypes = self.etypes
            if len(self.etypes) > 1:
                raise ValueError(
                    (
                        "Edge type name must be specified if there "
                        "are more than one edge types."
                    )
                )

            etype = etypes[0]
        if key not in self.edata_feat_col_d:
            raise ValueError(
                f"key {key} not found in CuGraphStore" " edge features",
                f" {list(self.edata_feat_col_d.keys())}",
            )
        columns = self.edata_feat_col_d[key]

        return CuFeatureStorage(
            pg=self.gdata,
            columns=columns,
            storage_type="edge",
            backend_lib=self.backend_lib,
            indices_offset=indices_offset,
        )

    def num_nodes(self, ntype=None):
        return self.gdata.get_num_vertices(ntype)

    def num_edges(self, etype=None):
        return self.gdata.get_num_edges(etype)

    @cached_property
    def has_multiple_etypes(self):
        return len(self.etypes) > 1

    @property
    def ntypes(self):
        return sorted(self.gdata.vertex_types)

    @property
    def etypes(self):
        return sorted(self.gdata.edge_types)

    @property
    def is_mg(self):
        return isinstance(self.gdata, MGPropertyGraph)

    @property
    def gdata(self):
        return self.__G

    ######################################
    # Sampling APIs
    ######################################

    def sample_neighbors(
        self, nodes_cap, fanout=-1, edge_dir="in", prob=None, replace=False
    ):
        """
        Sample neighboring edges of the given nodes and return the subgraph.

        Parameters
        ----------
        nodes_cap : Dlpack or dict of Dlpack of Node IDs
                    to sample neighbors from.
        fanout : int
            The number of edges to be sampled for each node on each edge type.
            If -1 is given all the neighboring edges for each node on
            each edge type will be selected.
        edge_dir : str {"in" or "out"}
            Determines whether to sample inbound or outbound edges.
            Can take either in for inbound edges or out for outbound edges.
        prob : str
            Feature name used as the (unnormalized) probabilities associated
            with each neighboring edge of a node. Each feature must be a
            scalar. The features must be non-negative floats, and the sum of
            the features of inbound/outbound edges for every node must be
            positive (though they don't have to sum up to one). Otherwise,
            the result will be undefined. If not specified, sample uniformly.
        replace : bool
            If True, sample with replacement.

        Returns
        -------
        DLPack capsule
            The src nodes for the sampled bipartite graph.
        DLPack capsule
            The sampled dst nodes for the sampledbipartite graph.
        DLPack capsule
            The corresponding eids for the sampled bipartite graph
        """

        if edge_dir not in ["in", "out"]:
            raise ValueError(
                f"edge_dir must be either 'in' or 'out' got {edge_dir} instead"
            )

        if isinstance(nodes_cap, dict):
            nodes = {t: cudf.from_dlpack(n) for t, n in nodes_cap.items()}
        else:
            nodes = cudf.from_dlpack(nodes_cap)

        if self.is_mg:
            sample_f = cugraph.dask.uniform_neighbor_sample
        else:
            sample_f = cugraph.uniform_neighbor_sample

        if self.has_multiple_etypes:
            # TODO: Convert into a single call when
            # https://github.com/rapidsai/cugraph/issues/2696 lands
            if edge_dir == "in":
                sgs = self.extracted_reverse_subgraphs_per_type
            else:
                sgs = self.extracted_subgraphs_per_type
            # Uniform sampling fails when the dtype
            # of the seed dtype is not same as the node dtype

            self.set_sg_node_dtype(list(sgs.values())[0][0])
            sampled_df = sample_multiple_sgs(
                sgs,
                sample_f,
                nodes,
                self._sg_node_dtype,
                edge_dir,
                fanout,
                replace,
            )
        else:
            if edge_dir == "in":
                sg, start_list_range = self.extracted_reverse_subgraph
            else:
                sg, start_list_range = self.extracted_subgraph
            self.set_sg_node_dtype(sg)
            sampled_df = sample_single_sg(
                sg,
                sample_f,
                nodes,
                self._sg_node_dtype,
                start_list_range,
                fanout,
                replace,
            )

        # we reverse directions when directions=='in'
        if edge_dir == "in":
            sampled_df = sampled_df.rename(
                columns={"destinations": src_n, "sources": dst_n}
            )
        else:
            sampled_df = sampled_df.rename(
                columns={"sources": src_n, "destinations": dst_n}
            )
        # Transfer data to client
        if isinstance(sampled_df, dask_cudf.DataFrame):
            sampled_df = sampled_df.compute()

        if self.has_multiple_etypes:
            # Heterogeneous graph case
            d = self._get_edgeid_type_d(sampled_df["indices"], self.etypes)
            d = create_dlpack_d(d)
            return d
        else:
            return (
                sampled_df[src_n].to_dlpack(),
                sampled_df[dst_n].to_dlpack(),
                sampled_df["indices"].to_dlpack(),
            )

    ######################################
    # Utilities
    ######################################
    @property
    def num_vertices(self):
        return self.gdata.get_num_vertices()

    def get_vertex_ids(self):
        return self.gdata.vertices_ids()

    def _get_edgeid_type_d(self, edge_ids, etypes):
        if isinstance(edge_ids, cudf.Series):
            # Work around for below issue
            # https://github.com/rapidsai/cudf/issues/11877
            edge_ids = edge_ids.values_host
        df = self.gdata.get_edge_data(edge_ids=edge_ids, columns=[type_n])
        if isinstance(df, dask_cudf.DataFrame):
            df = df.compute()
        return {etype: df[df[type_n] == etype] for etype in etypes}

    @cached_property
    def extracted_subgraph(self):
        edge_list = self.gdata.get_edge_data(columns=[src_n, dst_n, type_n])
        edge_list = edge_list.reset_index(drop=True)

        return get_subgraph_and_src_range_from_edgelist(
            edge_list, self.is_mg, reverse_edges=False
        )

    @cached_property
    def extracted_reverse_subgraph(self):
        edge_list = self.gdata.get_edge_data(columns=[src_n, dst_n, type_n])
        return get_subgraph_and_src_range_from_edgelist(
            edge_list, self.is_mg, reverse_edges=True
        )

    @cached_property
    def extracted_subgraphs_per_type(self):
        sg_d = {}
        for etype in self.etypes:
            edge_list = self.gdata.get_edge_data(
                columns=[src_n, dst_n, type_n], types=[etype]
            )
            sg_d[etype] = get_subgraph_and_src_range_from_edgelist(
                edge_list, self.is_mg, reverse_edges=False
            )
        return sg_d

    @cached_property
    def extracted_reverse_subgraphs_per_type(self):
        sg_d = {}
        for etype in self.etypes:
            edge_list = self.gdata.get_edge_data(
                columns=[src_n, dst_n, type_n], types=[etype]
            )
            sg_d[etype] = get_subgraph_and_src_range_from_edgelist(
                edge_list, self.is_mg, reverse_edges=True
            )
        return sg_d

    @cached_property
    def num_nodes_dict(self):
        """
        Return num_nodes_dict of the graph
        """
        return {ntype: self.num_nodes(ntype) for ntype in self.ntypes}

    @cached_property
    def num_edges_dict(self):
        return {etype: self.num_edges(etype) for etype in self.etypes}

    def set_sg_node_dtype(self, sg):
        if hasattr(self, "_sg_node_dtype"):
            return self._sg_node_dtype
        else:
            self._sg_node_dtype = get_underlying_dtype_from_sg(sg)
        return self._sg_node_dtype

    def find_edges(self, edge_ids_cap, etype):
        """Return the source and destination node IDs given the edge IDs within
        the given edge type.

        Parameters
        ----------
        edge_ids_cap :  Dlpack of Node IDs (single dimension)
            The edge ids  to find

        Returns
        -------
        DLPack capsule
            The src nodes for the given ids

        DLPack capsule
            The dst nodes for the given ids
        """
        edge_ids = cudf.from_dlpack(edge_ids_cap)
        subset_df = self.gdata.get_edge_data(
            edge_ids=edge_ids, columns=type_n, types=[etype]
        )
        if isinstance(subset_df, dask_cudf.DataFrame):
            subset_df = subset_df.compute()
        return subset_df[src_n].to_dlpack(), subset_df[dst_n].to_dlpack()

    def node_subgraph(
        self,
        nodes=None,
        create_using=cugraph.MultiGraph,
    ):
        """
        Return a subgraph induced on the given nodes.

        A node-induced subgraph is a graph with edges whose endpoints are both
        in the specified node set.

        Parameters
        ----------
        nodes : Tensor
            The nodes to form the subgraph.

        Returns
        -------
        cuGraph
            The sampled subgraph with the same node ID space with the original
            graph.
        """
        _g = self.gdata.extract_subgraph(
            create_using=create_using, check_multi_edges=True
        )

        if nodes is None:
            return _g
        else:
            _n = cudf.Series(nodes)
            _subg = cugraph.subgraph(_g, _n)
            return _subg

    def __clear_cached_properties(self):
        # Check for cached properties using self.__dict__ because calling
        # hasattr() accesses the attribute and forces computation
        if "has_multiple_etypes" in self.__dict__:
            del self.has_multiple_etypes

        if "num_nodes_dict" in self.__dict__:
            del self.num_nodes_dict

        if "num_edges_dict" in self.__dict__:
            del self.num_edges_dict

        if "extracted_subgraph" in self.__dict__:
            del self.extracted_subgraph

        if "extracted_reverse_subgraph" in self.__dict__:
            del self.extracted_reverse_subgraph

        if "extracted_subgraphs_per_type" in self.__dict__:
            del self.extracted_subgraphs_per_type

        if "extracted_reverse_subgraphs_per_type" in self.__dict__:
            del self.extracted_reverse_subgraphs_per_type
