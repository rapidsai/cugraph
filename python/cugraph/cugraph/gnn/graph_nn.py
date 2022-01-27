# Copyright (c) 2021, NVIDIA CORPORATION.
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

import warnings
import numbers

from cugraph import PropertyGraph

from cugraph.utilities.utils import import_optional
pd = import_optional("pandas")
np = import_optional("numpy")

dask_df = import_optional("dask_cudf")
cu_df = import_optional("cudf")


class GraphNN:
    """
    A GPU Graph Object that adheres to the DGLGraph structure, but with
    additional cuGraph related functions.  The class mainly wraps a
    cuGraph Property Graph

    When a GraphNN is created, a empty Property Graph is also created.


    Parameters
    ----------
    dask_client:    None or a dask.distributed.Client.  When a Client is
                    specified, the code will  run MNMG based on how the Client
                    was configured
    


    Examples
    --------

    """

    class GraphNNProps:
        def __init__(self, dask_client=None):
            self.client = dask_client
            self.G = PropertyGraph()


    def __init__(self, cluster=None):
        self.properties = GraphNN.GraphNNProps(cluster)

    #######################################################
    # Functions for Loading data
    #------------------------------------------------------
    def load_graph_attributes():
        pass






    #######################################################
    # Functions for querying graph structure
    #------------------------------------------------------

    def number_of_nodes(self, ntype=None):
        return self.num_nodes(ntype)

    def num_nodes(self, ntype=None):
        """
        Return the number of nodes in the graph.
        """
        return self.properties.G.number_of_nodes()

    def number_of_edges(self, etype=None):
        return self.num_edges(etype)

    def num_edges(self, etype=None):
        """ 
        Return the number of edges in the graph.
        """
        return self.properties.G.number_of_edges()

    def number_of_src_nodes(self, ntype=None):
        return self.num_src_nodes(ntype)

    def num_src_nodes(self, ntype=None):
        """
        Return the number of source nodes in a bipartite graph.
        """
        return self.properties.G.number_of_src_nodes(ntype)

    def number_of_dst_nodes(self, ntype=None):
        return self.num_dst_nodes(ntype)

    def num_dst_nodes(self, ntype=None):
        """
        Return the number of destination nodes in a bipartite graph.
        """
        return self.properties.G.number_of_dst_nodes(ntype)


    def is_multigraph(self):
        """
        Return whether the graph is a multigraph with parallel edges.
        """
        self.properties.G.is_multigraph()

    def is_unibipartite(self):
        """
        Return whether the graph is a is_unibipartite (bipartite) with parallel edges.
        """
        self.properties.G.is_bipartite()

    def is_homogeneous(self):
        """
        Return whether the graph is a is_unibipartite (bipartite) with parallel edges.
        """
        self.properties.G.is_homogeneous()


    def has_nodes(self, vid, ntype=None):
        """
        Return whether the graph contains the given nodes.
        """
        return self.properties.G.has_nodes(vid)


    def has_edges_between(self, u, v, etype=None):
        """
        Return whether the graph contains the given edges
        """
        return self.properties.G.has_edges_between(u,v)

    def predecessors(self, v, etype=None):
        """
        Return the predecessor(s) of a particular node with the specified edge type.
        """
        raise NotImplementedError("predecessors is not yet implemented")

    def successors(self, v, etype=None):
        """
        Return the successor(s) of a particular node with the specified edge type.
        """
        raise NotImplementedError("successors is not yet implemented")

    def edge_ids(self, u, v):
        # FIXME: handle types
        return self.properties.G.edge_ids(u,v)

    def find_edge(self, eid):
        return self.properties.G.find_edge(eid)

    def find_edges(self, eid):
        return self.properties.G.find_edges(eid)

    def in_edges(self, v):
        return self.properties.G.in_edges(v)

    def out_edges(self, v):
        return self.properties.G.out_edges(v)

    def in_degree(self, v):
        return self.properties.G.in_degree(v)

    def out_degree(self, v):
        return self.properties.G.out_degree(v)

    def out_degrees(self, v):
        return self.properties.G.out_degrees(v)

