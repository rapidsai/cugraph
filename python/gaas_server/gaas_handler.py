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
from inspect import signature

import numpy as np
import cudf
import dask_cudf
import cugraph
from dask.distributed import Client
from dask_cuda.initialize import initialize as dask_initialize
from cugraph.experimental import PropertyGraph, MGPropertyGraph
from cugraph.dask.comms import comms as Comms
from cugraph import uniform_neighbor_sample
from cugraph.dask import uniform_neighbor_sample as mg_uniform_neighbor_sample
from cugraph.structure.graph_implementation.simpleDistributedGraph import (
    simpleDistributedGraphImpl,
)

from gaas_client import defaults
from gaas_client.exceptions import GaasError
from gaas_client.types import (
    BatchedEgoGraphsResult,
    Node2vecResult,
    UniformNeighborSampleResult,
    ValueWrapper,
    GraphVertexEdgeIDWrapper,
)


def call_algo(sg_algo_func, G, **kwargs):
    """
    Calls the appropriate algo function based on the graph G being MG or SG. If
    G is SG, sg_algo_func will be called and passed kwargs, otherwise the MG
    version of sg_algo_func will be called with kwargs.
    """
    is_mg_graph = isinstance(G._Impl, simpleDistributedGraphImpl)

    if sg_algo_func is uniform_neighbor_sample:
        if is_mg_graph:
            possible_args = ["start_list", "fanout_vals", "with_replacement"]
            kwargs_to_pass = {a:kwargs[a] for a in possible_args
                              if a in kwargs}
            data = mg_uniform_neighbor_sample(G, **kwargs_to_pass)
            data = data.compute()
        else:
            possible_args = ["start_list", "fanout_vals", "with_replacement",
                             "is_edge_ids"]
            kwargs_to_pass = {a:kwargs[a] for a in possible_args
                              if a in kwargs}
            data = uniform_neighbor_sample(G, **kwargs_to_pass)

        return UniformNeighborSampleResult(
            sources=data.sources.values_host,
            destinations=data.destinations.values_host,
            indices=data.indices.values_host
        )

    else:
        raise RuntimeError(f"internal error: {sg_algo_func} is not supported")


class ExtensionServerFacade:
    """
    Instances of this class are passed to server extension functions to be used
    to access various aspects of the GaaS server from within the
    extension. This provideas a means to insulate the GaaS handler (considered
    here to be the "server") from direct access by end user extensions,
    allowing extension code to query/access the server as needed without giving
    extensions the ability to call potentially unsafe methods directly on the
    GaasHandler.

    An example is using an instance of a ExtensionServerFacade to allow a Graph
    creation extension to query the SG/MG state the server is using in order to
    determine how to create a Graph instance.
    """
    def __init__(self, gaas_handler):
        self.__handler = gaas_handler

    @property
    def is_mg(self):
        return self.__handler.is_mg

    def get_server_info(self):
        # The handler returns objects suitable for serialization over RPC so
        # convert them to regular py objs since this call is originating
        # server-side.
        return {k:ValueWrapper(v).get_py_obj() for (k, v)
                in self.__handler.get_server_info().items()}


class GaasHandler:
    """
    Class which handles RPC requests for a GaasService.
    """

    # The name of the param that should be set to a ExtensionServerFacade
    # instance for server extension functions.
    __server_facade_extension_param_name = "gaas_server"

    def __init__(self):
        self.__next_graph_id = defaults.graph_id + 1
        self.__graph_objs = {}
        self.__graph_creation_extensions = {}
        self.__dask_client = None
        self.__dask_cluster = None
        self.__start_time = int(time.time())

    def __del__(self):
        self.shutdown_dask_client()

    ############################################################################
    # Environment management
    @property
    def is_mg(self):
        """
        True if the GaasHandler has multiple GPUs available via a dask cluster.
        """
        return self.__dask_client is not None

    def uptime(self):
        """
        Return the server uptime in seconds. This is often used as a "ping".
        """
        return int(time.time()) - self.__start_time

    def get_server_info(self):
        """
        Returns a dictionary of meta-data about the server.

        Dictionary items are string:union_objs, where union_objs are Value
        "unions" used for RPC serialization.
        """
        # FIXME: expose self.__dask_client.scheduler_info() as needed
        if self.__dask_client is not None:
            num_gpus = len(self.__dask_client.scheduler_info()["workers"])
        else:
            # The assumption is that GaaS requires at least 1 GPU (ie.
            # currently there is no CPU-only version of GaaS)
            num_gpus = 1

        return {"num_gpus": ValueWrapper(num_gpus).union}

    def load_graph_creation_extensions(self, extension_dir_path):
        """
        Loads ("imports") all modules matching the pattern *_extension.py in
        the directory specified by extension_dir_path.

        The modules are searched and their functions are called (if a match is
        found) when call_graph_creation_extension() is called.

        """
        extension_dir = Path(extension_dir_path)

        if (not extension_dir.exists()) or (not extension_dir.is_dir()):
            raise GaasError(f"bad directory: {extension_dir}")

        num_files_read = 0

        for ext_file in extension_dir.glob("*_extension.py"):
            module_name = ext_file.stem
            spec = importlib.util.spec_from_file_location(module_name,
                                                          ext_file)
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
                    func_sig = signature(func)
                    func_params = list(func_sig.parameters.keys())
                    facade_param = self.__server_facade_extension_param_name

                    # Graph creation extensions that have the last arg named
                    # self.__server_facade_extension_param_name are passed a
                    # ExtensionServerFacade instance to allow them to query the
                    # "server" in a safe way, if needed.
                    if (facade_param in func_params):
                        if func_params[-1] == facade_param:
                            func_kwargs[facade_param] = \
                                ExtensionServerFacade(self)
                        else:
                            raise GaasError(f"{facade_param}, if specified, "
                                            "must be the last param.")

                    try:
                        graph_obj = func(*func_args, **func_kwargs)
                    except:
                        # FIXME: raise a more detailed error
                        raise GaasError(f"error running {func_name} : "
                                        f"{traceback.format_exc()}")
                    return self.__add_graph(graph_obj)

        raise GaasError(f"{func_name} is not a graph creation extension")

    def initialize_dask_client(self, dask_scheduler_file=None):
        """
        Initialize a dask client to be used for MG operations.
        """
        if dask_scheduler_file is not None:
            # Env var UCX_MAX_RNDV_RAILS=1 must be set too.
            dask_initialize(enable_tcp_over_ucx=True,
                            enable_nvlink=True,
                            enable_infiniband=True,
                            enable_rdmacm=True,
                            # net_devices="mlx5_0:1",
                            )
            self.__dask_client = Client(scheduler_file=dask_scheduler_file)
        else:
            # FIXME: LocalCUDACluster init. Implement when tests are in place.
            raise NotImplementedError

        if not Comms.is_initialized():
            Comms.initialize(p2p=True)

    def shutdown_dask_client(self):
        """
        Shutdown/cleanup the dask client for this handler instance.
        """
        if self.__dask_client is not None:
            Comms.destroy()
            self.__dask_client.close()

            if self.__dask_cluster is not None:
                self.__dask_cluster.close()
                self.__dask_cluster = None

            self.__dask_client = None

    ############################################################################
    # Graph management
    def create_graph(self):
        """
        Create a new graph associated with a new unique graph ID, return the new
        graph ID.
        """
        pG = self.__create_graph()
        return self.__add_graph(pG)

    def delete_graph(self, graph_id):
        """
        Remove the graph identified by graph_id from the server.
        """
        dG = self.__graph_objs.pop(graph_id, None)
        if dG is None:
            raise GaasError(f"invalid graph_id {graph_id}")

        del dG
        print(f'deleted graph with id {graph_id}')

    def get_graph_ids(self):
        """
        Returns a list of the graph IDs currently in use.
        """
        return list(self.__graph_objs.keys())

    def get_graph_info(self, keys, graph_id):
        """
        Returns a dictionary of meta-data about the graph identified by
        graph_id. If keys passed, only returns the values in keys.

        Dictionary items are string:union_objs, where union_objs are Value
        "unions" used for RPC serialization.
        """
        valid_keys = set(["num_vertices",
                          "num_vertices_from_vertex_data",
                          "num_edges",
                          "num_vertex_properties",
                          "num_edge_properties",
                          ])
        if len(keys) == 0:
            keys = valid_keys
        else:
            invalid_keys = set(keys) - valid_keys
            if len(invalid_keys) != 0:
                raise GaasError(f"got invalid keys: {invalid_keys}")

        G = self._get_graph(graph_id)
        info = {}
        if isinstance(G, (PropertyGraph, MGPropertyGraph)):
            for k in keys:
                if k == "num_vertices":
                    info[k] = G.get_num_vertices()
                elif k == "num_vertices_from_vertex_data":
                    info[k] = G.get_num_vertices(include_edge_data=False)
                elif k == "num_edges":
                    info[k] = G.get_num_edges()
                elif k == "num_vertex_properties":
                    info[k] = len(G.vertex_property_names)
                elif k == "num_edge_properties":
                    info[k] = len(G.edge_property_names)
        else:
            for k in keys:
                if k == "num_vertices":
                    info[k] = G.number_of_vertices()
                elif k == "num_vertices_from_vertex_data":
                    info[k] = 0
                elif k == "num_edges":
                    info[k] = G.number_of_edges()
                elif k == "num_vertex_properties":
                    info[k] = 0
                elif k == "num_edge_properties":
                    info[k] = 0

        return {key:ValueWrapper(value).union for (key, value) in info.items()}

    def get_graph_type(self, graph_id):
        """
        Returns a string repr of the graph type associated with graph_id.
        """
        return repr(type(self._get_graph(graph_id)))

    def load_csv_as_vertex_data(self,
                                csv_file_name,
                                delimiter,
                                dtypes,
                                header,
                                vertex_col_name,
                                type_name,
                                property_columns,
                                graph_id,
                                names
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

        if len(names) == 0:
            names = None

        # FIXME: error check that file exists
        # FIXME: error check that edgelist was read correctly
        try:
            gdf = self.__get_dataframe_from_csv(csv_file_name,
                                                delimiter=delimiter,
                                                dtypes=dtypes,
                                                header=header,
                                                names=names)
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
                              graph_id,
                              names
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

        if len(names) == 0:
            names = None

        try:
            gdf = self.__get_dataframe_from_csv(csv_file_name,
                                                delimiter=delimiter,
                                                dtypes=dtypes,
                                                header=header,
                                                names=names)
            pG.add_edge_data(gdf,
                             type_name=type_name,
                             vertex_col_names=vertex_col_names,
                             property_columns=property_columns)
        except:
            raise GaasError(f"{traceback.format_exc()}")

    # FIXME: ensure edge IDs can also be filtered by edge type
    # See: https://github.com/rapidsai/cugraph/issues/2655
    def get_edge_IDs_for_vertices(self, src_vert_IDs, dst_vert_IDs, graph_id):
        """
        Return a list of edge IDs corresponding to the vertex IDs in each of
        src_vert_IDs and dst_vert_IDs that, when combined, define an edge in the
        graph associated with graph_id.

        For example, if src_vert_IDs is [0, 1, 2] and dst_vert_IDs is [7, 8, 9],
        return the edge IDs for edges (0, 7), (1, 8), and (2, 9).

        graph_id must be associated with a Graph extracted from a PropertyGraph
        (MG or SG).
        """
        G = self._get_graph(graph_id)
        if isinstance(G, (PropertyGraph, MGPropertyGraph)):
            raise GaasError("get_edge_IDs_for_vertices() only accepts an "
                            "extracted subgraph ID, got an ID for a "
                            f"{type(G)}.")

        return self.__get_edge_IDs_from_graph_edge_data(G,
                                                        src_vert_IDs,
                                                        dst_vert_IDs)

    def extract_subgraph(self,
                         create_using,
                         selection,
                         edge_weight_property,
                         default_edge_weight,
                         allow_multi_edges,
                         renumber_graph,
                         add_edge_data,
                         graph_id
                         ):
        """
        Extract a subgraph, return a new graph ID
        """
        pG = self._get_graph(graph_id)
        if not(isinstance(pG, (PropertyGraph, MGPropertyGraph))):
            raise GaasError("extract_subgraph() can only be called on a graph "
                            "with properties.")
        # Convert defaults needed for the RPC API into defaults used by
        # PropertyGraph.extract_subgraph()
        create_using = create_using or cugraph.Graph
        selection = selection or None
        edge_weight_property = edge_weight_property or None

        # FIXME: create_using and selection should not be strings at this point

        try:
            G = pG.extract_subgraph(create_using,
                                    selection,
                                    edge_weight_property,
                                    default_edge_weight,
                                    allow_multi_edges,
                                    renumber_graph,
                                    add_edge_data)
        except:
            raise GaasError(f"{traceback.format_exc()}")

        return self.__add_graph(G)

    def get_graph_vertex_data(self,
                              id_or_ids,
                              null_replacement_value,
                              graph_id,
                              property_keys):
        """
        Returns the vertex data as a serialized numpy array for the given
        id_or_ids.  null_replacement_value must be provided if the data
        contains NA values, since NA values cannot be serialized.
        """
        pG = self._get_graph(graph_id)
        ids = GraphVertexEdgeIDWrapper(id_or_ids).get_py_obj()
        if ids == -1:
            ids = None
        elif not isinstance(ids, list):
            ids = [ids]
        if property_keys == []:
            columns = None
        else:
            columns = property_keys
        df = pG.get_vertex_data(vertex_ids=ids, columns=columns)
        return self.__get_graph_data_as_numpy_bytes(df, null_replacement_value)

    def get_graph_edge_data(self,
                            id_or_ids,
                            null_replacement_value,
                            graph_id,
                            property_keys):
        """
        Returns the edge data as a serialized numpy array for the given
        id_or_ids.  null_replacement_value must be provided if the data
        contains NA values, since NA values cannot be serialized.
        """
        pG = self._get_graph(graph_id)
        ids = GraphVertexEdgeIDWrapper(id_or_ids).get_py_obj()
        if ids == -1:
            ids = None
        elif not isinstance(ids, list):
            ids = [ids]
        if property_keys == []:
            columns = None
        else:
            columns = property_keys
        df = pG.get_edge_data(edge_ids=ids, columns=columns)
        return self.__get_graph_data_as_numpy_bytes(df, null_replacement_value)

    def is_vertex_property(self, property_key, graph_id):
        G = self._get_graph(graph_id)
        if isinstance(G, (PropertyGraph, MGPropertyGraph)):
            return property_key in G.vertex_property_names

        raise GaasError('Graph does not contain properties')

    def is_edge_property(self, property_key, graph_id):
        G = self._get_graph(graph_id)
        if isinstance(G, (PropertyGraph, MGPropertyGraph)):
            return property_key in G.edge_property_names

        raise GaasError('Graph does not contain properties')

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
        # FIXME: write test to catch an MGPropertyGraph being passed in
        if isinstance(G, PropertyGraph):
            raise GaasError("batched_ego_graphs() cannot operate directly on "
                            "a graph with properties, call extract_subgraph() "
                            "then call batched_ego_graphs() on the extracted "
                            "subgraph instead.")
        try:
            # FIXME: update this to use call_algo()
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
        # FIXME: write test to catch an MGPropertyGraph being passed in
        if isinstance(G, PropertyGraph):
            raise GaasError("node2vec() cannot operate directly on a graph with"
                            " properties, call extract_subgraph() then call "
                            "node2vec() on the extracted subgraph instead.")

        try:
            # FIXME: update this to use call_algo()
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

    def uniform_neighbor_sample(self,
                                start_list,
                                fanout_vals,
                                with_replacement,
                                graph_id,
                                ):
        G = self._get_graph(graph_id)
        if isinstance(G, (MGPropertyGraph, PropertyGraph)):
            raise GaasError("uniform_neighbor_sample() cannot operate directly "
                            "on a graph with properties, call "
                            "extract_subgraph() then call "
                            "uniform_neighbor_sample() on the extracted "
                            "subgraph instead.")

        try:
            return call_algo(
                uniform_neighbor_sample,
                G,
                start_list=start_list,
                fanout_vals=fanout_vals,
                with_replacement=with_replacement
            )
        except:
            raise GaasError(f"{traceback.format_exc()}")

    def pagerank(self, graph_id):
        """
        """
        raise NotImplementedError

    ############################################################################
    # "Protected" interface - used for both implementation and test/debug. Will
    # not be exposed to a GaaS client.
    def _get_graph(self, graph_id):
        """
        Return the cuGraph Graph object associated with graph_id.

        If the graph_id is the default graph ID and the default graph has not
        been created, then instantiate a new PropertyGraph as the default graph
        and return it.
        """
        pG = self.__graph_objs.get(graph_id)

        # Always create the default graph if it does not exist
        if pG is None:
            if graph_id == defaults.graph_id:
                pG = self.__create_graph()
                self.__graph_objs[graph_id] = pG
            else:
                raise GaasError(f"invalid graph_id {graph_id}")

        return pG

    ############################################################################
    # Private
    def __get_dataframe_from_csv(self,
                                 csv_file_name,
                                 delimiter,
                                 dtypes,
                                 header,
                                 names):
        """
        Read a CSV into a DataFrame and return it. This will use either a cuDF
        DataFrame or a dask_cudf DataFrame based on if the handler is configured
        to use a dask cluster or not.
        """
        gdf = cudf.read_csv(csv_file_name,
                            delimiter=delimiter,
                            dtype=dtypes,
                            header=header,
                            names=names)
        if self.is_mg:
            num_gpus = len(self.__dask_client.scheduler_info()["workers"])
            return dask_cudf.from_cudf(gdf, npartitions=num_gpus)

        return gdf

    def __add_graph(self, G):
        """
        Create a new graph ID for G and add G to the internal mapping of
        graph ID:graph instance.
        """
        gid = self.__next_graph_id
        self.__graph_objs[gid] = G
        self.__next_graph_id += 1
        return gid

    def __create_graph(self):
        """
        Instantiate a graph object using a type appropriate for the handler (
        either SG or MG)
        """
        return MGPropertyGraph() if self.is_mg else PropertyGraph()

    # FIXME: consider adding this to PropertyGraph
    def __remove_internal_columns(self, pg_column_names):
        """
        Removes all column names from pg_column_names that are "internal" (ie.
        used for PropertyGraph bookkeeping purposes only)
        """
        internal_column_names=[PropertyGraph.vertex_col_name,
                               PropertyGraph.src_col_name,
                               PropertyGraph.dst_col_name,
                               PropertyGraph.type_col_name,
                               PropertyGraph.edge_id_col_name,
                               PropertyGraph.vertex_id_col_name,
                               PropertyGraph.weight_col_name]

        # Create a list of user-visible columns by removing the internals while
        # preserving order
        user_visible_column_names = list(pg_column_names)
        for internal_column_name in internal_column_names:
            if internal_column_name in user_visible_column_names:
                user_visible_column_names.remove(internal_column_name)

        return user_visible_column_names

    # FIXME: consider adding this to PropertyGraph
    def __get_edge_IDs_from_graph_edge_data(self,
                                            G,
                                            src_vert_IDs,
                                            dst_vert_IDs):
        """
        Return a list of edge IDs corresponding to the vertex IDs in each of
        src_vert_IDs and dst_vert_IDs that, when combined, define an edge in G.

        For example, if src_vert_IDs is [0, 1, 2] and dst_vert_IDs is [7, 8, 9],
        return the edge IDs for edges (0, 7), (1, 8), and (2, 9).

        G must have an "edge_data" attribute.
        """
        edge_IDs = []
        num_edges = len(src_vert_IDs)

        for i in range(num_edges):
            src_mask = G.edge_data[PropertyGraph.src_col_name] == \
                src_vert_IDs[i]
            dst_mask = G.edge_data[PropertyGraph.dst_col_name] == \
                dst_vert_IDs[i]
            value = G.edge_data[src_mask & dst_mask]\
                [PropertyGraph.edge_id_col_name]

            # FIXME: This will compute the result (if using dask) then transfer
            # to host memory for each iteration - is there a more efficient way?
            if self.is_mg:
                value = value.compute()
            edge_IDs.append(value.values_host[0])

        return edge_IDs

    def __get_graph_data_as_numpy_bytes(self,
                                        dataframe,
                                        null_replacement_value):
        """
        Returns a byte array repr of the vertex or edge graph data. Since the byte
        array cannot represent NA values, null_replacement_value must be
        provided to be used in place of NAs.
        """
        try:
            if dataframe is None:
                return np.ndarray(shape=(0, 0)).dumps()
            elif isinstance(dataframe, dask_cudf.DataFrame):
                df = dataframe.compute()
            else:
                df = dataframe

            # null_replacement_value is a Value "union"
            n = ValueWrapper(null_replacement_value).get_py_obj()

            # This needs to be a copy of the df data to replace NA values
            # FIXME: should something other than a numpy type be serialized to
            # prevent a copy? (note: any other type required to be de-serialzed
            # on the client end could add dependencies on the client)
            df_numpy = df.to_numpy(na_value=n)
            return df_numpy.dumps()

        except:
            raise GaasError(f"{traceback.format_exc()}")
