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
import time
import traceback

import cudf
import cugraph
from cugraph.experimental import PropertyGraph

from gaas_client import defaults
from gaas_client.exceptions import GaasError
from gaas_client.types import BatchedEgoGraphsResult, Node2vecResult


class GaasHandler:
    """
    Class which handles RPC requests for a GaasService.
    """
    def __init__(self):
        self.__next_graph_id = defaults.graph_id + 1
        self.__graph_objs = {}
        self.__graph_creation_extensions = {}
        self.__start_time = int(time.time())

    ############################################################################
    # Environment management
    def uptime(self):
        """
        Return the server uptime in seconds. This is often used as a "ping".
        """
        return int(time.time()) - self.__start_time

    def load_graph_creation_extensions(self, extension_dir_path):
        """
        Loads ("imports") all modules matching the pattern *_extension.py in the
        directory specified by extension_dir_path.

        The modules are searched and their functions are called (if a match is
        found) when call_graph_creation_extension() is called.
        """
        extension_dir = Path(extension_dir_path)

        if (not extension_dir.exists()) or (not extension_dir.is_dir()):
            raise GaasError(f"bad directory: {extension_dir}")

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
        """
        Calls the graph creation extension function func_name and passes it the
        eval'd func_args_repr and func_kwargs_repr objects.

        The arg/kwarg reprs are eval'd prior to calling in order to pass actual
        python objects to func_name (this is needed to allow arbitrary arg
        objects to be serialized as part of the RPC call from the
        client).

        func_name cannot be a private name (name starting with __).

        All loaded extension modules are checked when searching for func_name,
        and the first extension module that contains it will have its function
        called.
        """
        if not(func_name.startswith("__")):
            for module in self.__graph_creation_extensions.values():
                # Ignore private functions
                func = getattr(module, func_name, None)
                if func is not None:
                    func_args = eval(func_args_repr)
                    func_kwargs = eval(func_kwargs_repr)
                    try:
                        graph_obj = func(*func_args, **func_kwargs)
                    except:
                        # FIXME: raise a more detailed error
                        raise GaasError(f"error running {func_name} : "
                                        f"{traceback.format_exc()}")
                    return self.__add_graph(graph_obj)

        raise GaasError(f"{func_name} is not a graph creation extension")

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
        """
        Returns a list of the graph IDs currently in use.
        """
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
        """
        Given a CSV csv_file_name present on the server's file system, read it
        and apply it as edge data to the graph specified by graph_id, or the
        default graph if not specified.
        """
        pG = self._get_graph(graph_id)
        if header == -1:
            header = "infer"
        elif header == -2:
            header = None
        # FIXME: error check that file exists
        # FIXME: error check that edgelist was read correctly
        try:
            gdf = cudf.read_csv(csv_file_name,
                                delimiter=delimiter,
                                dtype=dtypes,
                                header=header)
            pG.add_vertex_data(gdf,
                               type_name=type_name,
                               vertex_col_name=vertex_col_name,
                               property_columns=property_columns)
        except:
            raise GaasError(f"{traceback.format_exc()}")

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
        """
        Given a CSV csv_file_name present on the server's file system, read it
        and apply it as vertex data to the graph specified by graph_id, or the
        default graph if not specified.
        """
        pG = self._get_graph(graph_id)
        # FIXME: error check that file exists
        # FIXME: error check that edgelist read correctly
        if header == -1:
            header = "infer"
        elif header == -2:
            header = None
        try:
            gdf = cudf.read_csv(csv_file_name,
                                delimiter=delimiter,
                                dtype=dtypes,
                                header=header)
            pG.add_edge_data(gdf,
                             type_name=type_name,
                             vertex_col_names=vertex_col_names,
                             property_columns=property_columns)
        except:
            raise GaasError(f"{traceback.format_exc()}")

    def get_num_edges(self, graph_id):
        """
        Return the number of edges for the graph specified by graph_id.
        """
        pG = self._get_graph(graph_id)
        # FIXME: ensure non-PropertyGraphs that compute num_edges differently
        # work too.
        return pG.num_edges

    def get_num_vertices(self, graph_id):
        """
        Return the number of vertices for the graph specified by graph_id.
        """
        pG = self._get_graph(graph_id)
        return pG.num_vertices

    def get_edge_IDs_for_vertices(self, src_vert_IDs, dst_vert_IDs, graph_id):
        """
        """
        # FIXME: write docstring above
        G = self._get_graph(graph_id)
        if isinstance(G, PropertyGraph):
            # FIXME: also support PropertyGraph instances
            raise GaasError("get_edge_IDs_for_vertices() only accepts an "
                            "extracted subgraph ID, got a PropertyGraph ID.")

        # Lookup each edge ID in the graph edge_data (created during
        # extract_subgraph())
        edge_IDs = []
        num_edges = len(src_vert_IDs)
        for i in range(num_edges):
            src_mask = G.edge_data[PropertyGraph.src_col_name] == \
                src_vert_IDs[i]
            dst_mask = G.edge_data[PropertyGraph.dst_col_name] == \
                dst_vert_IDs[i]
            value = G.edge_data[src_mask & dst_mask]\
                [PropertyGraph.edge_id_col_name].values_host[0]
            edge_IDs.append(value)

        return edge_IDs

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

        try:
            G = pG.extract_subgraph(create_using,
                                    selection,
                                    edge_weight_property,
                                    default_edge_weight,
                                    allow_multi_edges)
        except:
            raise GaasError(f"{traceback.format_exc()}")

        return self.__add_graph(G)

    def get_graph_vertex_dataframe_rows(self,
                                        index_or_indices,
                                        null_replacement_value,
                                        graph_id):
        """
        """
        pG = self._get_graph(graph_id)

        # FIXME: consider a better API on PG for getting tabular vertex data, or
        # just make the "internal" _vertex_prop_dataframe a proper public API.
        # FIXME: this should not assume _vertex_prop_dataframe != None
        df = self.__get_dataframe_from_user_props(pG._vertex_prop_dataframe)

        return self.__get_dataframe_rows_as_numpy_bytes(df,
                                                        index_or_indices,
                                                        null_replacement_value)

    def get_graph_vertex_dataframe_shape(self, graph_id):
        """
        """
        pG = self._get_graph(graph_id)
        # FIXME: consider a better API on PG for getting tabular vertex data, or
        # just make the "internal" _vertex_prop_dataframe a proper public API.
        # FIXME: this should not assume _vertex_prop_dataframe != None
        df = self.__get_dataframe_from_user_props(pG._vertex_prop_dataframe)
        return df.shape

    def get_graph_edge_dataframe_rows(self,
                                      index_or_indices,
                                      null_replacement_value,
                                      graph_id):
        """
        """
        pG = self._get_graph(graph_id)

        # FIXME: consider a better API on PG for getting tabular edge data, or
        # just make the "internal" _edge_prop_dataframe a proper public API.
        # FIXME: this should not assume _edge_prop_dataframe != None
        df = self.__get_dataframe_from_user_props(pG._edge_prop_dataframe)

        return self.__get_dataframe_rows_as_numpy_bytes(df,
                                                        index_or_indices,
                                                        null_replacement_value)

    def get_graph_edge_dataframe_shape(self, graph_id):
        """
        """
        pG = self._get_graph(graph_id)
        # FIXME: consider a better API on PG for getting tabular edge data, or
        # just make the "internal" _edge_prop_dataframe a proper public API.
        # FIXME: this should not assume _edge_prop_dataframe != None
        df = self.__get_dataframe_from_user_props(pG._edge_prop_dataframe)
        return df.shape

    ############################################################################
    # Algos
    def batched_ego_graphs(self, seeds, radius, graph_id):
        """
        """
        st=time.time()
        print("\n----- [GaaS] -----> starting egonet", flush=True)
        # FIXME: finish docstring above
        # FIXME: exception handling
        G = self._get_graph(graph_id)
        if isinstance(G, PropertyGraph):
            raise GaasError("batched_ego_graphs() cannot operate directly on "
                            "a graph with properties, call extract_subgraph() "
                            "then call batched_ego_graphs() on the extracted "
                            "subgraph instead.")
        try:
            # FIXME: this should not be needed, need to update
            # cugraph.batched_ego_graphs to also accept a list
            seeds = cudf.Series(seeds, dtype="int32")
            st2=time.time()
            print("  ----- [GaaS] -----> calling cuGraph", flush=True)
            (ego_edge_list, seeds_offsets) = \
                cugraph.batched_ego_graphs(G, seeds, radius)

            print(f"  ----- [GaaS] -----> FINISHED calling cuGraph, time was: {time.time()-st2}s", flush=True)
            st2=time.time()
            print("  ----- [GaaS] -----> copying to host", flush=True)
            print(f"  ----- [GaaS] -----> {len(ego_edge_list['src'])} num edges", flush=True)
            #batched_ego_graphs_result = BatchedEgoGraphsResult(
            #    src_verts=ego_edge_list["src"].values_host.tobytes(),  #int32
            #    dst_verts=ego_edge_list["dst"].values_host.tobytes(),  #int32
            #    edge_weights=ego_edge_list["weight"].values_host.tobytes(),  #float64
            #    seeds_offsets=seeds_offsets.values_host.tobytes()  #int64
            #)
            batched_ego_graphs_result = BatchedEgoGraphsResult(
                src_verts=ego_edge_list["src"].values_host,
                dst_verts=ego_edge_list["dst"].values_host,
                edge_weights=ego_edge_list["weight"].values_host,
                seeds_offsets=seeds_offsets.values_host
            )
            print(f"  ----- [GaaS] -----> FINISHED copying to host, time was: {time.time()-st2}s", flush=True)
            return batched_ego_graphs_result
        except:
            raise GaasError(f"{traceback.format_exc()}")

        print(f"----- [GaaS] -----> FINISHED egonet, time was: {time.time()-st}s", flush=True)
        return batched_ego_graphs_result

    def node2vec(self, start_vertices, max_depth, graph_id):
        """
        """
        # FIXME: finish docstring above
        # FIXME: exception handling
        G = self._get_graph(graph_id)
        if isinstance(G, PropertyGraph):
            raise GaasError("node2vec() cannot operate directly on a graph with"
                            " properties, call extract_subgraph() then call "
                            "node2vec() on the extracted subgraph instead.")

        try:
            # FIXME: this should not be needed, need to update cugraph.node2vec to
            # also accept a list
            start_vertices = cudf.Series(start_vertices, dtype="int32")

            (paths, weights, path_sizes) = \
                cugraph.node2vec(G, start_vertices, max_depth)

            node2vec_result = Node2vecResult(
                vertex_paths = paths.values_host,
                edge_weights = weights.values_host,
                path_sizes = path_sizes.values_host,
            )
        except:
            raise GaasError(f"{traceback.format_exc()}")

        return node2vec_result

    def pagerank(self, graph_id):
        """
        """
        raise NotImplementedError

    ############################################################################
    # "Protected" interface - used for both implementation and test/debug. Will
    # not be exposed to a GaaS client.
    def _get_graph(self, graph_id):
        """
        Return the cuGraph Graph object (likely a PropertyGraph) associated with
        graph_id.

        If the graph_id is the default graph ID and the default graph has not
        been created, then instantiate a new PropertyGraph as the default graph
        and return it.
        """
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
        """
        Create a new graph ID for G and add G to the internal mapping of
        graph ID:graph instance.
        """
        gid = self.__next_graph_id
        self.__graph_objs[gid] = G
        self.__next_graph_id += 1
        return gid

    def __get_dataframe_from_user_props(self, dataframe):
        """
        """
        internal_columns=[PropertyGraph.vertex_col_name,
                          PropertyGraph.src_col_name,
                          PropertyGraph.dst_col_name,
                          PropertyGraph.type_col_name,
                          PropertyGraph.edge_id_col_name,
                          PropertyGraph.vertex_id_col_name,
                          PropertyGraph.weight_col_name]

        # Create a list of user-visible columns by removing the internals while
        # preserving order
        all_user_columns = list(dataframe.columns)
        for col_name in internal_columns:
            if col_name in all_user_columns:
                all_user_columns.remove(col_name)

        # This should NOT be a copy of the dataframe data
        return dataframe[all_user_columns]

    def __get_dataframe_rows_as_numpy_bytes(self,
                                            dataframe,
                                            index_or_indices,
                                            null_replacement_value):
        """
        """
        try:
            # index_or_indices and null_replacement_value are considered
            # "unions", meaning only one of their members will have a value.
            i = self.__get_value_from_union(index_or_indices)
            n = self.__get_value_from_union(null_replacement_value)

            # index -1 is the entire table
            if isinstance(i, int) and (i < -1):
                raise IndexError(f"an index must be -1 or greater, got {i}")
            elif i == -1:
                rows_df = dataframe
            else:
                # FIXME: dask_cudf does not support iloc
                rows_df = dataframe.iloc[i]

            # This needs to be a copy of the df data to replace NA values
            # FIXME: should something other than a numpy type be serialized to
            # prevent a copy? (note: any other type required to be de-serialzed
            # on the client end could add dependencies on the client)
            rows_numpy = rows_df.to_numpy(na_value=n)
            return rows_numpy.dumps()

        except:
            raise GaasError(f"{traceback.format_exc()}")

    @staticmethod
    def __get_value_from_union(union):
        """
        """
        not_members = set(["default_spec", "thrift_spec", "read", "write"])
        attrs = [a for a in dir(union)
                    if not(a.startswith("_")) and a not in not_members]
        for a in attrs:
            val = getattr(union, a)
            if val is not None:
                return val

        return None
