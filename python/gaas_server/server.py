# Copyright (c) 2022, NVIDIA CORPORATION.
#
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

import cudf
import cugraph
from cugraph.experimental import PropertyGraph

from gaas_client import defaults
from gaas_client.exceptions import GaasError
from gaas_client.types import Node2vecResult


class GaasHandler:
    """
    Class which handles RPC requests for a GaasService.
    """
    def __init__(self):
        self.__next_graph_id = defaults.graph_id + 1
        self.__graph_objs = {}

    ############################################################################
    # Environment management

    # FIXME: do we need environment mgmt functions?  This could be things
    # related to querying the state of the dask cluster,
    # starting/stopping/restarting, etc.

    ############################################################################
    # Graph management
    def create_graph(self):
        """
        Create a new graph associated with a new unique graph ID, return the new
        graph ID.
        """
        pG = PropertyGraph()
        return self.__add_graph(pG)

    def delete_graph(self, graph_id):
        """
        Remove the graph identified by graph_id from the server.
        """
        if self.__graph_objs.pop(graph_id, None) is None:
            raise GaasError(f"invalid graph_id {graph_id}")

    def get_graph_ids(self):
        return list(self.__graph_objs.keys())

    def load_csv_as_vertex_data(self,
                                csv_file_name,
                                delimiter,
                                dtypes,
                                header,
                                vertex_col_name,
                                type_name,
                                property_columns,
                                graph_id
                                ):
        pG = self.__get_graph(graph_id)
        if header == -1:
            header = "infer"
        elif header == -2:
            header = None
        # FIXME: error check that file exists
        # FIXME: error check that edgelist was read correctly
        gdf = cudf.read_csv(csv_file_name,
                            delimiter=delimiter,
                            dtype=dtypes,
                            header=header)
        pG.add_vertex_data(gdf,
                           type_name=type_name,
                           vertex_col_name=vertex_col_name,
                           property_columns=property_columns)

    def load_csv_as_edge_data(self,
                              csv_file_name,
                              delimiter,
                              dtypes,
                              header,
                              vertex_col_names,
                              type_name,
                              property_columns,
                              graph_id
                              ):
        pG = self.__get_graph(graph_id)
        # FIXME: error check that file exists
        # FIXME: error check that edgelist read correctly
        if header == -1:
            header = "infer"
        elif header == -2:
            header = None
        gdf = cudf.read_csv(csv_file_name,
                            delimiter=delimiter,
                            dtype=dtypes,
                            header=header)
        pG.add_edge_data(gdf,
                         type_name=type_name,
                         vertex_col_names=vertex_col_names,
                         property_columns=property_columns)

    def get_num_edges(self, graph_id):
        pG = self.__get_graph(graph_id)
        # FIXME: ensure non-PropertyGraphs that compute num_edges differently
        # work too.
        return pG.num_edges

    def extract_subgraph(self,
                         create_using,
                         selection,
                         edge_weight_property,
                         default_edge_weight,
                         allow_multi_edges,
                         graph_id
                         ):
        """
        Extract a subgraph, return a new graph ID
        """
        pG = self.__get_graph(graph_id)
        if not(isinstance(pG, PropertyGraph)):
            raise GaasError("extract_subgraph() can only be called on a graph "
                            "with properties.")
        # Convert defaults needed for the Thrift API into defaults used by
        # PropertyGraph.extract_subgraph()
        create_using = create_using or cugraph.Graph
        selection = selection or None
        edge_weight_property = edge_weight_property or None

        G = pG.extract_subgraph(create_using,
                                selection,
                                edge_weight_property,
                                default_edge_weight,
                                allow_multi_edges)

        return self.__add_graph(G)

    ############################################################################
    # Algos
    def node2vec(self, start_vertices, max_depth, graph_id):
        """
        """
        # FIXME: exception handling
        G = self.__get_graph(graph_id)
        if isinstance(G, PropertyGraph):
            raise GaasError("node2vec() cannot operate directly on a graph with"
                            " properties, call extract_subgraph() then call "
                            "node2vec() on the extracted subgraph instead.")

        # FIXME: this should not be needed, need to update cugraph.node2vec to
        # also accept a list
        start_vertices = cudf.Series(start_vertices, dtype="int32")

        (paths, weights, path_sizes) = \
            cugraph.node2vec(G, start_vertices, max_depth)

        node2vec_result = Node2vecResult(
            vertex_paths = paths.to_arrow().to_pylist(),
            edge_weights = weights.to_arrow().to_pylist(),
            path_sizes = path_sizes.to_arrow().to_pylist()
        )
        return node2vec_result

    def pagerank(self, graph_id):
        """
        """
        raise NotImplementedError

    ############################################################################
    # Private
    def __add_graph(self, G):
        gid = self.__next_graph_id
        self.__graph_objs[gid] = G
        self.__next_graph_id += 1
        return gid

    def __get_graph(self, graph_id):
        pG = self.__graph_objs.get(graph_id)
        if pG is None:
            # Always create the default graph if it does not exist
            if graph_id == defaults.graph_id:
                pG = PropertyGraph()
                self.__graph_objs[graph_id] = pG
            else:
                raise GaasError(f"invalid graph_id {graph_id}")
        return pG



if __name__ == '__main__':
    from gaas_client.gaas_thrift import create_server

    # FIXME: add CLI options to set non-default host and port values, and
    # possibly other options.
    server = create_server(GaasHandler(),
                           host=defaults.host,
                           port=defaults.port)
    print('Starting the server...')
    server.serve()
    print('done.')
