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
import cupy as cp
from functools import cached_property


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
        if isinstance(graph, (PropertyGraph, MGPropertyGraph)):
            self.__G = graph
        else:
            raise ValueError("graph must be a PropertyGraph or" " MGPropertyGraph")
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
        feat_name=None,
        ntype=None,
        is_single_vector_feature=True,
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
        feat_name : string
            The feature name under which we should save the added properties
            (ignored if is_single_vector_feature=False and the col names of
            the dataframe are treated as corresponding feature names)
        ntype : string
            The node type to be added.
            For example, if dataframe contains data about users, ntype
            might be "users".
            If not specified, the type of properties will be added as
            an empty string.
        is_single_vector_feature : True
            Whether to treat all the columns of the dataframe being added as
            a single 2d feature
        Returns
        -------
        None
        """
        self.gdata.add_vertex_data(df, vertex_col_name=node_col_name, type_name=ntype)
        columns = [col for col in list(df.columns) if col != node_col_name]

        if is_single_vector_feature:
            if feat_name is None:
                raise ValueError(
                    "feature name must be provided when wrapping"
                    + " multiple columns under a single feature name"
                )

        elif feat_name:
            raise ValueError(
                "feat_name is only valid when wrapping"
                + " multiple columns under a single feature name"
            )

        if is_single_vector_feature:
            self.ndata_feat_col_d[feat_name] = columns
        else:
            for col in columns:
                self.ndata_feat_col_d[col] = [col]
        # Clear properties if set as data has changed
        self.__clear_cached_properties()

    def add_edge_data(
        self,
        df,
        node_col_names,
        feat_name=None,
        etype=None,
        is_single_vector_feature=True,
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
        feat_name : string
            The feature name under which we should save the added properties
            (ignored if is_single_vector_feature=False and the col names of
            the dataframe are treated as corresponding feature names)
        etype : string
            The edge type to be added. This should follow the string format
            '(src_type),(edge_type),(dst_type)'
            If not specified, the type of properties will be added as
            an empty string.
        is_single_vector_feature : True
            Wether to treat all the columns of the dataframe being
            added as a single 2d feature
        Returns
        -------
        None
        """
        self.gdata.add_edge_data(df, vertex_col_names=node_col_names, type_name=etype)
        columns = [col for col in list(df.columns) if col not in node_col_names]
        if is_single_vector_feature:
            if feat_name is None:
                raise ValueError(
                    "feature name must be provided when wrapping"
                    + " multiple columns under a single feature name"
                )

        elif feat_name:
            raise ValueError(
                "feat_name is only valid when wrapping"
                + " multiple columns under a single feature name"
            )

        if is_single_vector_feature:
            self.edata_feat_col_d[feat_name] = columns
        else:
            for col in columns:
                self.edata_feat_col_d[col] = [col]

        # Clear properties if set as data has changed
        self.__clear_cached_properties()

    def get_node_storage(self, feat_name, ntype=None):
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
        if feat_name not in self.ndata_feat_col_d:
            raise ValueError(
                f"feat_name {feat_name} not found in CuGraphStore" " node features",
                f" {list(self.ndata_feat_col_d.keys())}",
            )

        columns = self.ndata_feat_col_d[feat_name]

        return CuFeatureStorage(
            pg=self.gdata,
            columns=columns,
            storage_type="node",
            backend_lib=self.backend_lib,
        )

    def get_edge_storage(self, feat_name, etype=None):
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
        if feat_name not in self.edata_feat_col_d:
            raise ValueError(
                f"feat_name {feat_name} not found in CuGraphStore" " edge features",
                f" {list(self.edata_feat_col_d.keys())}",
            )
        columns = self.edata_feat_col_d[feat_name]

        return CuFeatureStorage(
            pg=self.gdata,
            columns=columns,
            storage_type="edge",
            backend_lib=self.backend_lib,
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

            self.set_sg_node_dtype(list(sgs.values())[0])
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
                sg = self.extracted_reverse_subgraph
            else:
                sg = self.extracted_subgraph
            self.set_sg_node_dtype(sg)
            sampled_df = sample_single_sg(
                sg, sample_f, nodes, self._sg_node_dtype, fanout, replace
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
            d = return_dlpack_d(d)
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

        return get_subgraph_from_edgelist(edge_list, self.is_mg, reverse_edges=False)

    @cached_property
    def extracted_reverse_subgraph(self):
        edge_list = self.gdata.get_edge_data(columns=[src_n, dst_n, type_n])
        return get_subgraph_from_edgelist(edge_list, self.is_mg, reverse_edges=True)

    @cached_property
    def extracted_subgraphs_per_type(self):
        sg_d = {}
        for etype in self.etypes:
            edge_list = self.gdata.get_edge_data(
                columns=[src_n, dst_n, type_n], types=[etype]
            )
            sg_d[etype] = get_subgraph_from_edgelist(
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
            sg_d[etype] = get_subgraph_from_edgelist(
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
            # FIXME: Remove after we have consistent naming
            # https://github.com/rapidsai/cugraph/issues/2618
            sg_columns = sg.edgelist.edgelist_df.columns
            if "src" in sg_columns:
                # src for single node graph
                self._sg_node_dtype = sg.edgelist.edgelist_df["src"].dtype
            elif src_n in sg_columns:
                # _SRC_ for multi-node graphs
                self._sg_node_dtype = sg.edgelist.edgelist_df[src_n].dtype
            else:
                raise ValueError(f"Source column {src_n} not found in the subgraph")
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
        if hasattr(self, "has_multiple_etypes"):
            del self.has_multiple_etypes

        if hasattr(self, "num_nodes_dict"):
            del self.num_nodes_dict

        if hasattr(self, "num_edges_dict"):
            del self.num_edges_dict

        if hasattr(self, "extracted_subgraph"):
            del self.extracted_subgraph

        if hasattr(self, "extracted_reverse_subgraph"):
            del self.extracted_reverse_subgraph

        if hasattr(self, "extracted_subgraphs_per_type"):
            del self.extracted_subgraphs_per_type

        if hasattr(self, "extracted_reverse_subgraphs_per_type"):
            del self.extracted_reverse_subgraphs_per_type


class CuFeatureStorage:
    """Storage for node/edge feature data.

    Either subclassing this class or implementing the same set of interfaces
    is fine. DGL simply uses duck-typing to implement its sampling pipeline.
    """

    def __init__(self, pg, columns, storage_type, backend_lib="torch"):
        self.pg = pg
        self.columns = columns
        if backend_lib == "torch":
            from torch.utils.dlpack import from_dlpack
        elif backend_lib == "tf":
            from tensorflow.experimental.dlpack import from_dlpack
        elif backend_lib == "cupy":
            from cupy import from_dlpack
        else:
            raise NotImplementedError(
                "Only pytorch and tensorflow backends are currently supported"
            )
        if storage_type not in ["edge", "node"]:
            raise NotImplementedError("Only edge and node storage is supported")

        self.storage_type = storage_type

        self.from_dlpack = from_dlpack

    def fetch(self, indices, device=None, pin_memory=False, **kwargs):
        """Fetch the features of the given node/edge IDs to the
        given device.

        Parameters
        ----------
        indices : Tensor
            Node or edge IDs.
        device : Device
            Device context.
        pin_memory :

        Returns
        -------
        Tensor
            Feature data stored in PyTorch Tensor.
        """
        # Default implementation uses synchronous fetch.

        indices = cp.asarray(indices)
        if isinstance(self.pg, MGPropertyGraph):
            # dask_cudf loc breaks if we provide cudf series/cupy array
            # https://github.com/rapidsai/cudf/issues/11877
            indices = indices.get()
        else:
            indices = cudf.Series(indices)
        if self.storage_type == "node":
            subset_df = self.pg.get_vertex_data(
                vertex_ids=indices, columns=self.columns
            )
        else:
            subset_df = self.pg.get_edge_data(edge_ids=indices, columns=self.columns)

        subset_df = subset_df[self.columns]

        if isinstance(subset_df, dask_cudf.DataFrame):
            subset_df = subset_df.compute()
        if len(subset_df) == 0:
            raise ValueError(f"{indices=} not found in FeatureStorage")
        else:
            tensor = self.from_dlpack(subset_df.to_dlpack())

        if isinstance(tensor, cp.ndarray):
            # can not transfer to
            # a different device for cupy
            return tensor
        else:
            if device:
                tensor = tensor.to(device)
            else:
                return tensor


def return_dlpack_d(d):
    dlpack_d = {}
    for k, df in d.items():
        if len(df) == 0:
            dlpack_d[k] = (None, None, None)
        else:
            dlpack_d[k] = (
                df[src_n].to_dlpack(),
                df[dst_n].to_dlpack(),
                df[eid_n].to_dlpack(),
            )

    return dlpack_d


def sample_single_sg(
    sg, sample_f, start_list, start_list_dtype, fanout, with_replacement
):
    if isinstance(start_list, dict):
        start_list = cudf.concat(list(start_list.values()))

    # Uniform sampling fails when the dtype
    # of the seed dtype is not same as the node dtype
    start_list = start_list.astype(start_list_dtype)
    sampled_df = sample_f(
        sg,
        start_list=start_list,
        fanout_vals=[fanout],
        with_replacement=with_replacement,
        # FIXME: is_edge_ids=True does not seem to do anything
        # issue https://github.com/rapidsai/cugraph/issues/2562
    )
    return sampled_df


def sample_multiple_sgs(
    sgs,
    sample_f,
    start_list_d,
    start_list_dtype,
    edge_dir,
    fanout,
    with_replacement,
):
    start_list_types = list(start_list_d.keys())
    output_dfs = []
    for can_etype, sg in sgs.items():
        can_etype = _convert_can_etype_s_to_tup(can_etype)
        if _edge_types_contains_canonical_etype(can_etype, start_list_types, edge_dir):
            if edge_dir == "in":
                subset_type = can_etype[2]
            else:
                subset_type = can_etype[0]

            output = sample_single_sg(
                sg,
                sample_f,
                start_list_d[subset_type],
                start_list_dtype,
                fanout,
                with_replacement,
            )
            output_dfs.append(output)

    if len(output_dfs) == 0:
        empty_df = cudf.DataFrame({"sources": [], "destinations": [], "indices": []})
        return empty_df.astype(cp.int32)

    if isinstance(output_dfs[0], dask_cudf.DataFrame):
        return dask_cudf.concat(output_dfs, ignore_index=True)
    else:
        return cudf.concat(output_dfs, ignore_index=True)


def _edge_types_contains_canonical_etype(can_etype, edge_types, edge_dir):
    src_type, _, dst_type = can_etype
    if edge_dir == "in":
        return dst_type in edge_types
    else:
        return src_type in edge_types


def _convert_can_etype_s_to_tup(canonical_etype_s):
    src_type, etype, dst_type = canonical_etype_s.split(",")
    src_type = src_type[2:-1]
    dst_type = dst_type[2:-2]
    etype = etype[2:-1]
    return (src_type, etype, dst_type)


def get_subgraph_from_edgelist(edge_list, is_mg, reverse_edges=False):
    if reverse_edges:
        edge_list = edge_list.rename(columns={src_n: dst_n, dst_n: src_n})

    subgraph = cugraph.MultiGraph(directed=True)
    if is_mg:
        # FIXME: Can not switch to renumber = False
        # For MNMG Algos
        # Remove when https://github.com/rapidsai/cugraph/issues/2437
        # lands
        create_subgraph_f = subgraph.from_dask_cudf_edgelist
        renumber = True
    else:
        # Note: We have to keep renumber = False
        # to handle cases when the seed_nodes is not present in sugraph
        create_subgraph_f = subgraph.from_cudf_edgelist
        renumber = False

    create_subgraph_f(
        edge_list,
        source=src_n,
        destination=dst_n,
        edge_attr=eid_n,
        renumber=renumber,
        # FIXME: renumber=False is not supported for MNMG algos
        legacy_renum_only=True,
    )

    return subgraph
