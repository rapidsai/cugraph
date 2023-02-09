# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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


from cugraph.gnn.dgl_extensions.base_cugraph_store import BaseCuGraphStore

from functools import cached_property
from cugraph.gnn.dgl_extensions.utils.find_edges import find_edges
from cugraph.gnn.dgl_extensions.utils.node_subgraph import node_subgraph
from cugraph.gnn.dgl_extensions.utils.add_data import (
    add_edge_data_from_parquet,
    add_edge_data_from_dataframe,
    add_node_data_from_parquet,
    add_node_data_from_dataframe,
)
from cugraph.gnn.dgl_extensions.utils.sampling import (
    sample_pg,
    get_subgraph_and_src_range_from_pg,
)
from cugraph.gnn.dgl_extensions.utils.sampling import get_underlying_dtype_from_sg
from cugraph.gnn.dgl_extensions.feature_storage import CuFeatureStorage


class CuGraphStore(BaseCuGraphStore):
    """
    A wrapper around a cuGraph Property Graph that
    then adds functions to basically match the DGL GraphStorage API.
    This is not a full duck-types match to a DGL GraphStore.

    This class return dlpack types and has additional functional arguments.
    """

    def __init__(self, graph, backend_lib="torch"):

        if type(graph).__name__ in ["PropertyGraph", "MGPropertyGraph"]:
            self.__G = graph
        else:
            raise ValueError("graph must be a PropertyGraph or MGPropertyGraph")
        super().__init__(graph)
        # dict to map column names corresponding to edge features
        # of each type
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
        """
        add_node_data_from_dataframe(
            df=df,
            node_col_name=node_col_name,
            ntype=ntype,
            feat_name=feat_name,
            contains_vector_features=contains_vector_features,
            pG=self.gdata,
        )
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
        """

        # Clear properties if set as data has changed
        add_edge_data_from_dataframe(
            df=df,
            node_col_names=node_col_names,
            canonical_etype=canonical_etype,
            feat_name=feat_name,
            contains_vector_features=contains_vector_features,
            pG=self.gdata,
        )
        self.__clear_cached_properties()

    def add_node_data_from_parquet(
        self,
        file_path,
        node_col_name,
        ntype=None,
        node_offset=0,
        feat_name=None,
        contains_vector_features=False,
    ):
        """
        Add a dataframe describing node properties to the PropertyGraph.

        Parameters
        ----------
        file_path: string
            Path of the files on the server
        node_col_name : string
            The column name that contains the values to be used as vertex IDs.
        ntype : string
            The node type to be added.
            For example, if dataframe contains data about users, ntype
            might be "users".
            If not specified, the type of properties will be added as
            an empty string.
        node_offset: int,
            The offset to add for the particular ntype
            defaults to zero
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
        """
        add_node_data_from_parquet(
            file_path=file_path,
            node_col_name=node_col_name,
            node_offset=node_offset,
            ntype=ntype,
            feat_name=feat_name,
            contains_vector_features=contains_vector_features,
            pG=self.gdata,
        )
        # Clear properties if set as data has changed
        self.__clear_cached_properties()

    def add_edge_data_from_parquet(
        self,
        file_path,
        node_col_names,
        src_offset=0,
        dst_offset=0,
        canonical_etype=None,
        feat_name=None,
        contains_vector_features=False,
    ):
        """
        Add a dataframe describing edge properties to the PropertyGraph.

        Parameters
        ----------
        file_path : string
            Path of file on server
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

        src_offset: int,
            The offset to add for the source node type
            defaults to zero
        dst_offset: int,
            The offset to add for the dst node type
            defaults to zero
        contains_vector_features : False
            Whether to treat the columns of the dataframe being added as
            as 2d features
        Returns
        -------
        None
        """

        add_edge_data_from_parquet(
            file_path=file_path,
            node_col_names=node_col_names,
            canonical_etype=canonical_etype,
            src_offset=src_offset,
            dst_offset=dst_offset,
            feat_name=feat_name,
            contains_vector_features=contains_vector_features,
            pG=self.gdata,
        )
        self.__clear_cached_properties()

    def get_node_storage(self, key, ntype=None, indices_offset=0):
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
        return CuFeatureStorage(
            pg=self.gdata,
            column=key,
            storage_type="node",
            indices_offset=indices_offset,
            backend_lib=self.backend_lib,
            types_to_fetch=[ntype],
        )

    def get_edge_storage(self, key, etype=None, indices_offset=0):
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

        return CuFeatureStorage(
            pg=self.gdata,
            column=key,
            storage_type="edge",
            backend_lib=self.backend_lib,
            indices_offset=indices_offset,
            types_to_fetch=[etype],
        )

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
            The sampled dst nodes for the sampled bipartite graph.
        DLPack capsule
            The corresponding eids for the sampled bipartite graph
        """

        if edge_dir not in ["in", "out"]:
            raise ValueError(
                f"edge_dir must be either 'in' or 'out' got {edge_dir} instead"
            )

        if self.has_multiple_etypes:
            # TODO: Convert into a single call when
            # https://github.com/rapidsai/cugraph/issues/2696 lands
            if edge_dir == "in":
                sgs_obj, sgs_src_range_obj = self.extracted_reverse_subgraphs_per_type
            else:
                sgs_obj, sgs_src_range_obj = self.extracted_subgraphs_per_type
            first_sg = list(sgs_obj.values())[0]
        else:
            if edge_dir == "in":
                sgs_obj, sgs_src_range_obj = self.extracted_reverse_subgraph
            else:
                sgs_obj, sgs_src_range_obj = self.extracted_subgraph

            first_sg = sgs_obj
        # Uniform sampling fails when the dtype
        # of the seed dtype is not same as the node dtype
        self.set_sg_node_dtype(first_sg)

        # Below will be called from remote storage
        sampled_result_arrays = sample_pg(
            self.gdata,
            has_multiple_etypes=self.has_multiple_etypes,
            etypes=self.etypes,
            sgs_obj=sgs_obj,
            sgs_src_range_obj=sgs_src_range_obj,
            sg_node_dtype=self._sg_node_dtype,
            nodes_ar=nodes_cap,
            replace=replace,
            fanout=fanout,
            edge_dir=edge_dir,
        )
        return create_dlpack_results_from_arrays(sampled_result_arrays, self.etypes)

    ######################################
    # Utilities
    ######################################
    @property
    def num_vertices(self):
        return self.gdata.get_num_vertices()

    def get_vertex_ids(self):
        return self.gdata.vertices_ids()

    @cached_property
    def extracted_subgraph(self):
        return get_subgraph_and_src_range_from_pg(
            self.gdata, reverse_edges=False, etype=None
        )

    @cached_property
    def extracted_reverse_subgraph(self):
        return get_subgraph_and_src_range_from_pg(
            self.gdata, reverse_edges=True, etype=None
        )

    @cached_property
    def extracted_subgraphs_per_type(self):
        sg_d = {}
        sg_src_range_d = {}
        for etype in self.etypes:
            sg_d[etype], sg_src_range_d[etype] = get_subgraph_and_src_range_from_pg(
                self.gdata, reverse_edges=False, etype=etype
            )
        return sg_d, sg_src_range_d

    @cached_property
    def extracted_reverse_subgraphs_per_type(self):
        sg_d = {}
        sg_src_range_d = {}
        for etype in self.etypes:
            sg_d[etype], sg_src_range_d[etype] = get_subgraph_and_src_range_from_pg(
                self.gdata, reverse_edges=True, etype=etype
            )
        return sg_d, sg_src_range_d

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
        return find_edges(edge_ids_cap, etype)

    def node_subgraph(
        self,
        nodes=None,
        create_using=None,
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
        return node_subgraph(self.gdata, nodes, create_using)

    def __clear_cached_properties(self):
        # Check for cached properties using self.__dict__ because calling
        # hasattr() accesses the attribute and forces computation
        if "has_multiple_etypes" in self.__dict__:
            del self.has_multiple_etypes

        if "etypes" in self.__dict__:
            del self.etypes

        if "ntypes" in self.__dict__:
            del self.ntypes

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


def create_dlpack_results_from_arrays(sampled_result_arrays, etypes):
    # TODO: Extend to pytorch/numpy/etc
    import cupy as cp

    if len(etypes) <= 1:
        s, d, e_id = sampled_result_arrays
        # Handle numpy array, cupy array, lists etc
        s, d, e_id = cp.asarray(s), cp.asarray(d), cp.asarray(e_id)
        return s.toDlpack(), d.toDlpack(), e_id.toDlpack()
    else:
        result_d = {}
        array_start_offset = 0
        for etype in etypes:
            s = sampled_result_arrays[array_start_offset]
            d = sampled_result_arrays[array_start_offset + 1]
            e_id = sampled_result_arrays[array_start_offset + 2]
            s, d, e_id = cp.asarray(s), cp.asarray(d), cp.asarray(e_id)
            array_start_offset = array_start_offset + 3
            if s is not None and len(s) >= 0:
                s, d, e_id = s.toDlpack(), d.toDlpack(), e_id.toDlpack()
            else:
                s, d, e_id = None, None, None
            result_d[etype] = (s, d, e_id)
        return result_d
