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

from pathlib import Path
import importlib

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
        self.__graph_creation_extensions = {}

    ############################################################################
    # Environment management

    def load_graph_creation_extensions(self, extension_dir_path):
        extension_dir = Path(extension_dir_path)

        if (not extension_dir.exists()) or (not extension_dir.is_dir()):
            raise NotADirectoryError(extension_dir)

        num_files_read = 0

        for ext_file in extension_dir.glob("*_extension.py"):
            module_name = ext_file.stem
            spec = importlib.util.spec_from_file_location(module_name, ext_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.__graph_creation_extensions[module_name] = module
            num_files_read += 1

        return num_files_read

    def unload_graph_creation_extensions(self):
        """
        Removes all graph creation extensions.
        """
        self.__graph_creation_extensions.clear()

    def call_graph_creation_extension(self, func_name,
                                      func_args_repr, func_kwargs_repr):
        if not(func_name.startswith("__")):
            for module in self.__graph_creation_extensions.values():
                # Ignore private functions
                func = getattr(module, func_name, None)
                if func is not None:
                    func_args = eval(func_args_repr)
                    func_kwargs = eval(func_kwargs_repr)
                    graph_obj = func(*func_args, **func_kwargs)
                    return self.__add_graph(graph_obj)

        raise AttributeError(func_name)

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
        pG = self._get_graph(graph_id)
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
        pG = self._get_graph(graph_id)
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
        pG = self._get_graph(graph_id)
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
        pG = self._get_graph(graph_id)
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
        G = self._get_graph(graph_id)
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
    # "Protected" interface - used for both implementation and test/debug. Will
    # not be exposed to a GaaS client.
    def _get_graph(self, graph_id):
        pG = self.__graph_objs.get(graph_id)
        if pG is None:
            # Always create the default graph if it does not exist
            if graph_id == defaults.graph_id:
                pG = PropertyGraph()
                self.__graph_objs[graph_id] = pG
            else:
                raise GaasError(f"invalid graph_id {graph_id}")
        return pG

    ############################################################################
    # Private
    def __add_graph(self, G):
        gid = self.__next_graph_id
        self.__graph_objs[gid] = G
        self.__next_graph_id += 1
        return gid
