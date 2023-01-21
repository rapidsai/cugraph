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

from functools import cached_property


class BaseCuGraphStore:
    """
    BaseClass for DGL GraphStore and RemoteGraphStore
    """

    def __init__(self, graph):
        self.__G = graph

    def get_edge_storage(self, key, etype=None, indices_offset=0):
        raise NotImplementedError

    def get_node_storage(self, key, ntype=None, indices_offset=0):
        raise NotImplementedError

    @property
    def gdata(self):
        return self.__G

    @property
    def num_vertices(self):
        return self.gdata.get_num_vertices()

    def num_nodes(self, ntype=None):
        return self.gdata.get_num_vertices(ntype)

    def num_edges(self, etype=None):
        return self.gdata.get_num_edges(etype)

    @cached_property
    def has_multiple_etypes(self):
        return len(self.etypes) > 1

    @cached_property
    def ntypes(self):
        return sorted(self.gdata.vertex_types)

    @cached_property
    def etypes(self):
        return sorted(self.gdata.edge_types)

    ######################################
    # Sampling APIs
    ######################################

    def sample_neighbors(
        self, nodes_cap, fanout=-1, edge_dir="in", prob=None, replace=False
    ):
        raise NotImplementedError

    ######################################
    # Utilities
    ######################################
    @property
    def extracted_subgraph(self):
        raise NotImplementedError

    @cached_property
    def num_nodes_dict(self):
        """
        Return num_nodes_dict of the graph
        """
        return {ntype: self.num_nodes(ntype) for ntype in self.ntypes}

    @cached_property
    def num_edges_dict(self):
        return {etype: self.num_edges(etype) for etype in self.etypes}
