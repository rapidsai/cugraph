# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

from functools import cached_property
from pathlib import Path
import importlib
import time
import traceback
import re
from inspect import signature
import asyncio
import tempfile

# FIXME This optional import is required to support graph creation
# extensions that use OGB.  It should be removed when a better
# workaround is found.
from cugraph.utilities.utils import import_optional

import numpy as np
import cupy as cp
import ucp
import cudf
import dask_cudf
import rmm
from cugraph import (
    batched_ego_graphs,
    uniform_neighbor_sample,
    node2vec,
    Graph,
    MultiGraph,
)
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from dask_cuda.initialize import initialize as dask_initialize
from cugraph.experimental import PropertyGraph, MGPropertyGraph
from cugraph.dask.comms import comms as Comms
from cugraph.dask import uniform_neighbor_sample as mg_uniform_neighbor_sample
from cugraph.structure.graph_implementation.simpleDistributedGraph import (
    simpleDistributedGraphImpl,
)
from cugraph.dask.common.mg_utils import get_visible_devices

from cugraph_service_client import defaults
from cugraph_service_client import (
    extension_return_dtype_map,
    supported_extension_return_dtypes,
)
from cugraph_service_client.exceptions import CugraphServiceError
from cugraph_service_client.types import (
    BatchedEgoGraphsResult,
    Node2vecResult,
    UniformNeighborSampleResult,
    ValueWrapper,
    GraphVertexEdgeIDWrapper,
    Offsets,
)

ogb = import_optional("ogb")


def call_algo(sg_algo_func, G, **kwargs):
    """
    Calls the appropriate algo function based on the graph G being MG or SG. If
    G is SG, sg_algo_func will be called and passed kwargs, otherwise the MG
    version of sg_algo_func will be called with kwargs.
    """
    is_multi_gpu_graph = isinstance(G._Impl, simpleDistributedGraphImpl)

    if sg_algo_func is uniform_neighbor_sample:
        if is_multi_gpu_graph:
            possible_args = ["start_list", "fanout_vals", "with_replacement"]
            kwargs_to_pass = {a: kwargs[a] for a in possible_args if a in kwargs}
            result_ddf = mg_uniform_neighbor_sample(G, **kwargs_to_pass)
            # Convert DataFrame into CuPy arrays for returning to the client
            result_df = result_ddf.compute()
            sources = result_df["sources"].to_cupy()
            destinations = result_df["destinations"].to_cupy()
            indices = result_df["indices"].to_cupy()
        else:
            possible_args = [
                "start_list",
                "fanout_vals",
                "with_replacement",
                "is_edge_ids",
            ]
            kwargs_to_pass = {a: kwargs[a] for a in possible_args if a in kwargs}
            result_df = uniform_neighbor_sample(G, **kwargs_to_pass)
            # Convert DataFrame into CuPy arrays for returning to the client
            sources = result_df["sources"].to_cupy()
            destinations = result_df["destinations"].to_cupy()
            indices = result_df["indices"].to_cupy()

        return UniformNeighborSampleResult(
            sources=sources,
            destinations=destinations,
            indices=indices,
        )

    else:
        raise RuntimeError(f"internal error: {sg_algo_func} is not supported")


class ExtensionServerFacade:
    """
    Instances of this class are passed to server extension functions to be used
    to access various aspects of the cugraph_service_client server from within
    the extension. This provideas a means to insulate the CugraphHandler
    (considered here to be the "server") from direct access by end user
    extensions, allowing extension code to query/access the server as needed
    without giving extensions the ability to call potentially unsafe methods
    directly on the CugraphHandler.

    An example is using an instance of a ExtensionServerFacade to allow a Graph
    creation extension to query the SG/MG state the server is using in order to
    determine how to create a Graph instance.
    """

    def __init__(self, cugraph_handler):
        self.__handler = cugraph_handler

    @property
    def is_multi_gpu(self):
        return self.__handler.is_multi_gpu

    def get_server_info(self):
        # The handler returns objects suitable for serialization over RPC so
        # convert them to regular py objs since this call is originating
        # server-side.
        return {
            k: ValueWrapper(v).get_py_obj()
            for (k, v) in self.__handler.get_server_info().items()
        }

    def get_graph_ids(self):
        return self.__handler.get_graph_ids()

    def get_graph(self, graph_id):
        return self.__handler._get_graph(graph_id)

    def add_graph(self, G):
        return self.__handler._add_graph(G)


class CugraphHandler:
    """
    Class which handles RPC requests for a cugraph_service server.
    """

    # The name of the param that should be set to a ExtensionServerFacade
    # instance for server extension functions.
    __server_facade_extension_param_name = "server"

    def __init__(self):
        self.__next_graph_id = defaults.graph_id + 1
        self.__graph_objs = {}
        self.__graph_creation_extensions = {}
        self.__extensions = {}
        self.__dask_client = None
        self.__dask_cluster = None
        self.__start_time = int(time.time())
        self.__next_test_array_id = 0
        self.__test_arrays = {}

    def __del__(self):
        self.shutdown_dask_client()

    ###########################################################################
    # Environment management
    @cached_property
    def is_multi_gpu(self):
        """
        True if the CugraphHandler has multiple GPUs available via a dask
        cluster.
        """
        return self.__dask_client is not None

    @cached_property
    def num_gpus(self):
        """
        If dask is not available, this returns "1".  Otherwise it returns
        the number of GPUs accessible through dask.
        """
        return (
            len(self.__dask_client.scheduler_info()["workers"])
            if self.is_multi_gpu
            else 1
        )

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

        return {
            "num_gpus": ValueWrapper(self.num_gpus).union,
            "extensions": ValueWrapper(list(self.__extensions.keys())).union,
            "graph_creation_extensions": ValueWrapper(
                list(self.__graph_creation_extensions.keys())
            ).union,
        }

    def load_graph_creation_extensions(self, extension_dir_or_mod_path):
        """
        Loads ("imports") all modules matching the pattern *_extension.py in the
        directory specified by extension_dir_or_mod_path. extension_dir_or_mod_path
        can be either a path to a directory on disk, or a python import path to a
        package.

        The modules are searched and their functions are called (if a match is
        found) when call_graph_creation_extension() is called.

        The extensions loaded are to be used for graph creation, and the server assumes
        the return value of the extension functions is a Graph-like object which is
        registered and assigned a unique graph ID.
        """
        modules_loaded = []
        try:
            extension_files = self.__get_extension_files_from_path(
                extension_dir_or_mod_path
            )

            for ext_file in extension_files:
                module_file_path = ext_file.absolute().as_posix()
                spec = importlib.util.spec_from_file_location(
                    module_file_path, ext_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.__graph_creation_extensions[module_file_path] = module
                modules_loaded.append(module_file_path)

            return modules_loaded

        except Exception:
            raise CugraphServiceError(f"{traceback.format_exc()}")

    def load_extensions(self, extension_dir_or_mod_path):
        """
        Loads ("imports") all modules matching the pattern *_extension.py in the
        directory specified by extension_dir_or_mod_path. extension_dir_or_mod_path
        can be either a path to a directory on disk, or a python import path to a
        package.

        The modules are searched and their functions are called (if a match is
        found) when call_graph_creation_extension() is called.
        """
        modules_loaded = []

        try:
            extension_files = self.__get_extension_files_from_path(
                extension_dir_or_mod_path
            )

            for ext_file in extension_files:
                module_file_path = ext_file.absolute().as_posix()
                spec = importlib.util.spec_from_file_location(
                    module_file_path, ext_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.__extensions[module_file_path] = module
                modules_loaded.append(module_file_path)

            return modules_loaded

        except Exception:
            raise CugraphServiceError(f"{traceback.format_exc()}")

    def unload_extension_module(self, modname):
        """
        Removes all extension functions in modname.
        """
        if (self.__graph_creation_extensions.pop(modname, None) is None) and (
            self.__extensions.pop(modname, None) is None
        ):
            raise CugraphServiceError(f"bad extension module {modname}")

    def call_graph_creation_extension(
        self, func_name, func_args_repr, func_kwargs_repr
    ):
        """
        Calls the graph creation extension function func_name and passes it the
        eval'd func_args_repr and func_kwargs_repr objects.  If successful, it
        associates the graph returned by the extension function with a new graph
        ID and returns it.

        func_name cannot be a private name (name starting with __).
        """
        graph_obj = self.__call_extension(
            self.__graph_creation_extensions,
            func_name,
            func_args_repr,
            func_kwargs_repr,
        )
        # FIXME: ensure graph_obj is a graph obj
        return self._add_graph(graph_obj)

    def call_extension(
        self,
        func_name,
        func_args_repr,
        func_kwargs_repr,
        result_host=None,
        result_port=None,
    ):
        """
        Calls the extension function func_name and passes it the eval'd
        func_args_repr and func_kwargs_repr objects. If successful, returns a
        Value object containing the results returned by the extension function.

        func_name cannot be a private name (name starting with __).
        """
        try:
            result = self.__call_extension(
                self.__extensions, func_name, func_args_repr, func_kwargs_repr
            )
            if self.__check_host_port_args(result_host, result_port):
                # Ensure result is in list format for calling __ucx_send_results so it
                # sends the contents as individual arrays.
                if isinstance(result, (list, tuple)):
                    result_list = result
                else:
                    result_list = [result]

                # Form the meta-data array to send first. This array contains uint8
                # values which map to dtypes the client uses when converting bytes to
                # values.
                meta_data = []
                for r in result_list:
                    if hasattr(r, "dtype"):
                        dtype_str = str(r.dtype)
                    else:
                        dtype_str = type(r).__name__

                    dtype_enum_val = extension_return_dtype_map.get(dtype_str)
                    if dtype_enum_val is None:
                        raise TypeError(
                            f"extension {func_name} returned an invalid type "
                            f"{dtype_str}, only "
                            f"{supported_extension_return_dtypes} are supported"
                        )
                    meta_data.append(dtype_enum_val)
                # FIXME: meta_data should not need to be a cupy array
                meta_data = cp.array(meta_data, dtype="uint8")

                asyncio.run(
                    self.__ucx_send_results(
                        result_host,
                        result_port,
                        meta_data,
                        *result_list,
                    )
                )
                # FIXME: Thrift still expects something of the expected type to
                # be returned to be serialized and sent. Look into a separate
                # API that uses the Thrift "oneway" modifier when returning
                # results via client device.
                return ValueWrapper(None)
            else:
                return ValueWrapper(result)

        except Exception:
            raise CugraphServiceError(f"{traceback.format_exc()}")

    def initialize_dask_client(
        self,
        protocol=None,
        rmm_pool_size=None,
        dask_worker_devices=None,
        dask_scheduler_file=None,
    ):
        """
        Initialize a dask client to be used for MG operations.
        """
        if dask_scheduler_file is not None:
            dask_initialize()
            self.__dask_client = Client(scheduler_file=dask_scheduler_file)
        else:
            # The tempdir created by tempdir_object should be cleaned up once
            # tempdir_object goes out-of-scope and is deleted.
            tempdir_object = tempfile.TemporaryDirectory()
            cluster = LocalCUDACluster(
                local_directory=tempdir_object.name,
                protocol=protocol,
                rmm_pool_size=rmm_pool_size,
                CUDA_VISIBLE_DEVICES=dask_worker_devices,
            )
            # Initialize the client to use RMM pool allocator if cluster is
            # using it.
            if rmm_pool_size is not None:
                rmm.reinitialize(pool_allocator=True)

            self.__dask_client = Client(cluster)

            if dask_worker_devices is not None:
                # FIXME: this assumes a properly formatted string with commas
                num_workers = len(dask_worker_devices.split(","))
            else:
                num_workers = len(get_visible_devices())
            self.__dask_client.wait_for_workers(num_workers)

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

    ###########################################################################
    # Graph management
    def create_graph(self):
        """
        Create a new graph associated with a new unique graph ID, return the
        new graph ID.
        """
        pG = self.__create_graph()
        return self._add_graph(pG)

    def delete_graph(self, graph_id):
        """
        Remove the graph identified by graph_id from the server.
        """
        dG = self.__graph_objs.pop(graph_id, None)
        if dG is None:
            raise CugraphServiceError(f"invalid graph_id {graph_id}")

        del dG
        print(f"deleted graph with id {graph_id}")

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
        valid_keys = set(
            [
                "num_vertices",
                "num_vertices_from_vertex_data",
                "num_edges",
                "num_vertex_properties",
                "num_edge_properties",
                "is_multi_gpu",
            ]
        )
        if len(keys) == 0:
            keys = valid_keys
        else:
            invalid_keys = set(keys) - valid_keys
            if len(invalid_keys) != 0:
                raise CugraphServiceError(f"got invalid keys: {invalid_keys}")

        G = self._get_graph(graph_id)
        info = {}
        try:
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
                    elif k == "is_multi_gpu":
                        info[k] = isinstance(G, MGPropertyGraph)
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
                    elif k == "is_multi_gpu":
                        info[k] = G.is_multi_gpu()
        except Exception:
            raise CugraphServiceError(f"{traceback.format_exc()}")

        return {key: ValueWrapper(value) for (key, value) in info.items()}

    def load_csv_as_vertex_data(
        self,
        csv_file_name,
        delimiter,
        dtypes,
        header,
        vertex_col_name,
        type_name,
        property_columns,
        graph_id,
        names,
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
            gdf = self.__get_dataframe_from_csv(
                csv_file_name,
                delimiter=delimiter,
                dtypes=dtypes,
                header=header,
                names=names,
            )
            pG.add_vertex_data(
                gdf,
                type_name=type_name,
                vertex_col_name=vertex_col_name,
                property_columns=property_columns,
            )
        except Exception:
            raise CugraphServiceError(f"{traceback.format_exc()}")

    def load_csv_as_edge_data(
        self,
        csv_file_name,
        delimiter,
        dtypes,
        header,
        vertex_col_names,
        type_name,
        property_columns,
        graph_id,
        names,
        edge_id_col_name,
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

        if edge_id_col_name == "":
            edge_id_col_name = None

        try:
            gdf = self.__get_dataframe_from_csv(
                csv_file_name,
                delimiter=delimiter,
                dtypes=dtypes,
                header=header,
                names=names,
            )
            pG.add_edge_data(
                gdf,
                type_name=type_name,
                vertex_col_names=vertex_col_names,
                property_columns=property_columns,
                edge_id_col_name=edge_id_col_name,
            )
        except Exception:
            raise CugraphServiceError(f"{traceback.format_exc()}")

    # FIXME: ensure edge IDs can also be filtered by edge type
    # See: https://github.com/rapidsai/cugraph/issues/2655
    def get_edge_IDs_for_vertices(self, src_vert_IDs, dst_vert_IDs, graph_id):
        """
        Return a list of edge IDs corresponding to the vertex IDs in each of
        src_vert_IDs and dst_vert_IDs that, when combined, define an edge in
        the graph associated with graph_id.

        For example, if src_vert_IDs is [0, 1, 2] and dst_vert_IDs is [7, 8, 9]
        return the edge IDs for edges (0, 7), (1, 8), and (2, 9).

        graph_id must be associated with a Graph extracted from a PropertyGraph
        (MG or SG).
        """
        G = self._get_graph(graph_id)
        if isinstance(G, (PropertyGraph, MGPropertyGraph)):
            raise CugraphServiceError(
                "get_edge_IDs_for_vertices() only "
                "accepts an extracted subgraph ID, got "
                f"an ID for a {type(G)}."
            )

        return self.__get_edge_IDs_from_graph_edge_data(G, src_vert_IDs, dst_vert_IDs)

    def renumber_vertices_by_type(self, prev_id_column: str, graph_id: int) -> Offsets:
        G = self._get_graph(graph_id)
        if isinstance(G, (PropertyGraph, MGPropertyGraph)):
            if prev_id_column == "":
                prev_id_column = None

            offset_df = G.renumber_vertices_by_type(prev_id_column=prev_id_column)
            if self.is_multi_gpu:
                offset_df = offset_df.compute()

            # type needs be converted twice due to cudf bug
            offsets_obj = Offsets(
                type=offset_df.index.values_host.to_numpy(),
                start=offset_df.start.to_numpy(),
                stop=offset_df.stop.to_numpy(),
            )

            return offsets_obj
        else:
            raise CugraphServiceError(
                "Renumbering graphs without properties is currently unsupported"
            )

    def renumber_edges_by_type(self, prev_id_column: str, graph_id: int) -> Offsets:
        G = self._get_graph(graph_id)
        if isinstance(G, (PropertyGraph, MGPropertyGraph)):
            if prev_id_column == "":
                prev_id_column = None

            offset_df = G.renumber_edges_by_type(prev_id_column=prev_id_column)
            if self.is_multi_gpu:
                offset_df = offset_df.compute()

            # type needs be converted twice due to cudf bug
            offsets_obj = Offsets(
                type=offset_df.index.values_host.to_numpy(),
                start=offset_df.start.to_numpy(),
                stop=offset_df.stop.to_numpy(),
            )

            return offsets_obj
        else:
            raise CugraphServiceError(
                "Renumbering graphs without properties is currently unsupported"
            )

    def extract_subgraph(
        self,
        create_using,
        selection,
        edge_weight_property,
        default_edge_weight,
        check_multi_edges,
        renumber_graph,
        add_edge_data,
        graph_id,
    ):
        """
        Extract a subgraph, return a new graph ID
        """
        pG = self._get_graph(graph_id)
        if not (isinstance(pG, (PropertyGraph, MGPropertyGraph))):
            raise CugraphServiceError(
                "extract_subgraph() can only be called " "on a graph with properties."
            )
        # Convert defaults needed for the RPC API into defaults used by
        # PropertyGraph.extract_subgraph()
        try:
            if create_using == "":
                create_using = None
            elif create_using is not None:
                create_using = self.__parse_create_using_string(create_using)
            edge_weight_property = edge_weight_property or None
            if selection == "":
                selection = None
            elif selection is not None:
                selection = pG.select_edges(selection)

            # FIXME: create_using and selection should not be strings at this point

            G = pG.extract_subgraph(
                create_using=create_using,
                selection=selection,
                edge_weight_property=edge_weight_property,
                default_edge_weight=default_edge_weight,
                check_multi_edges=check_multi_edges,
                renumber_graph=renumber_graph,
                add_edge_data=add_edge_data,
            )
        except Exception:
            raise CugraphServiceError(f"{traceback.format_exc()}")

        return self._add_graph(G)

    def get_graph_vertex_data(
        self, id_or_ids, null_replacement_value, property_keys, types, graph_id
    ):
        """
        Returns the vertex data as a serialized numpy array for the given
        id_or_ids.  null_replacement_value must be provided if the data
        contains NA values, since NA values cannot be serialized.

        If the graph is a structural graph (a graph without properties),
        this method does not accept the id_or_ids, property_keys, or types
        arguments, and instead returns a list of valid vertex ids.
        """
        G = self._get_graph(graph_id)
        ids = GraphVertexEdgeIDWrapper(id_or_ids).get_py_obj()
        null_replacement_value = ValueWrapper(null_replacement_value).get_py_obj()

        if ids == -1:
            ids = None
        elif not isinstance(ids, list):
            ids = [ids]
        if property_keys == []:
            columns = None
        else:
            columns = property_keys
        if types == []:
            types = None
        if isinstance(G, (PropertyGraph, MGPropertyGraph)):
            if columns is not None and G.vertex_col_name in columns:
                raise CugraphServiceError(
                    f"ID key {G.vertex_col_name} is not allowed for property query. "
                    f"Vertex IDs are always returned in query."
                )

            try:
                df = G.get_vertex_data(vertex_ids=ids, columns=columns, types=types)
                if isinstance(df, dask_cudf.DataFrame):
                    df = df.compute()
            except KeyError:
                df = None
        else:
            if (columns is not None) or (ids is not None) or (types is not None):
                raise CugraphServiceError("Graph does not contain properties")
            if self.is_multi_gpu:
                # FIXME may run out of memory for very lage graphs.
                s = (
                    dask_cudf.concat(
                        [
                            G.edgelist.edgelist_df[
                                G.renumber_map.renumbered_src_col_name
                            ],
                            G.edgelist.edgelist_df[
                                G.renumber_map.renumbered_dst_col_name
                            ],
                        ]
                    )
                    .unique()
                    .compute()
                )
                df = cudf.DataFrame()
                df["id"] = s
                df = dask_cudf.from_cudf(df, npartitions=self.num_gpus)
            else:
                s = cudf.concat(
                    [
                        G.edgelist.edgelist_df[G.srcCol],
                        G.edgelist.edgelist_df[G.dstCol],
                    ]
                ).unique()
                df = cudf.DataFrame()
                df["id"] = s
            if G.is_renumbered():
                df = G.unrenumber(df, "id", preserve_order=True)

        return self.__get_graph_data_as_numpy_bytes(df, null_replacement_value)

    def get_graph_edge_data(
        self, id_or_ids, null_replacement_value, property_keys, types, graph_id
    ):
        """
        Returns the edge data as a serialized numpy array for the given
        id_or_ids.  null_replacement_value must be provided if the data
        contains NA values, since NA values cannot be serialized.
        """
        G = self._get_graph(graph_id)
        ids = GraphVertexEdgeIDWrapper(id_or_ids).get_py_obj()
        if ids == -1:
            ids = None
        elif not isinstance(ids, list):
            ids = [ids]
        if property_keys == []:
            columns = None
        else:
            columns = property_keys
        if types == []:
            types = None
        if isinstance(G, (PropertyGraph, MGPropertyGraph)):
            try:
                df = G.get_edge_data(edge_ids=ids, columns=columns, types=types)
            except KeyError:
                df = None
        else:
            if columns is not None:
                raise CugraphServiceError(
                    f"Graph does not contain properties. {columns}"
                )

            # Get the edgelist; API expects edge id, src, dst, type
            df = G.edgelist.edgelist_df

            if G.edgeIdCol in df.columns:
                if ids is not None:
                    if self.is_multi_gpu:
                        # FIXME use ids = cudf.Series(ids) after dask_cudf fix
                        ids = np.array(ids)
                        df = df.reindex(df[G.edgeIdCol]).loc[ids]
                    else:
                        ids = cudf.Series(ids)
                        df = df.reindex(df[G.edgeIdCol]).loc[ids]
            else:
                if ids is not None:
                    raise CugraphServiceError("Graph does not have edge ids")
                df[G.edgeIdCol] = df.index

            if G.edgeTypeCol in df.columns:
                if types is not None:
                    df = df[df[G.edgeTypeCol].isin(types)]
            else:
                if types is not None:
                    raise CugraphServiceError("Graph does not have typed edges")
                df[G.edgeTypeCol] = ""

            src_col_name = (
                G.renumber_map.renumbered_src_col_name
                if self.is_multi_gpu
                else G.srcCol
            )
            dst_col_name = (
                G.renumber_map.renumbered_dst_col_name
                if self.is_multi_gpu
                else G.dstCol
            )
            if G.is_renumbered():
                df = G.unrenumber(df, src_col_name, preserve_order=True)
                df = G.unrenumber(df, dst_col_name, preserve_order=True)

            df = df[[G.edgeIdCol, src_col_name, dst_col_name, G.edgeTypeCol]]

        if isinstance(df, dask_cudf.DataFrame):
            df = df.compute()
        return self.__get_graph_data_as_numpy_bytes(df, null_replacement_value)

    def is_vertex_property(self, property_key, graph_id):
        G = self._get_graph(graph_id)
        if isinstance(G, (PropertyGraph, MGPropertyGraph)):
            return property_key in G.vertex_property_names

        raise CugraphServiceError("Graph does not contain properties")

    def is_edge_property(self, property_key, graph_id):
        G = self._get_graph(graph_id)
        if isinstance(G, (PropertyGraph, MGPropertyGraph)):
            return property_key in G.edge_property_names

        raise CugraphServiceError("Graph does not contain properties")

    def get_graph_vertex_property_names(self, graph_id):
        G = self._get_graph(graph_id)
        if isinstance(G, (PropertyGraph, MGPropertyGraph)):
            return G.vertex_property_names

        return []

    def get_graph_edge_property_names(self, graph_id):
        G = self._get_graph(graph_id)
        if isinstance(G, (PropertyGraph, MGPropertyGraph)):
            return G.edge_property_names

        return []

    def get_graph_vertex_types(self, graph_id):
        G = self._get_graph(graph_id)
        if isinstance(G, (PropertyGraph, MGPropertyGraph)):
            return G.vertex_types
        else:
            return [""]

    def get_graph_edge_types(self, graph_id):
        G = self._get_graph(graph_id)
        if isinstance(G, (PropertyGraph, MGPropertyGraph)):
            return G.edge_types
        else:
            # FIXME should call G.vertex_types (See issue #2889)
            if G.edgeTypeCol in G.edgelist.edgelist_df.columns:
                return (
                    G.edgelist.edgelist_df[G.edgeTypeCol]
                    .unique()
                    .astype("str")
                    .values_host
                )
            else:
                return [""]

    def get_num_vertices(self, vertex_type, include_edge_data, graph_id):
        # FIXME should include_edge_data always be True in the remote case?
        G = self._get_graph(graph_id)
        if isinstance(G, (PropertyGraph, MGPropertyGraph)):
            if vertex_type == "":
                return G.get_num_vertices(include_edge_data=include_edge_data)
            else:
                return G.get_num_vertices(
                    type=vertex_type, include_edge_data=include_edge_data
                )

        else:
            if vertex_type != "":
                raise CugraphServiceError("Graph does not support vertex types")
            return G.number_of_vertices()

    def get_num_edges(self, edge_type, graph_id):
        G = self._get_graph(graph_id)
        if isinstance(G, (PropertyGraph, MGPropertyGraph)):
            if edge_type == "":
                return G.get_num_edges()
            else:
                return G.get_num_edges(type=edge_type)

        else:
            if edge_type == "":
                return G.number_of_edges()
            else:
                # FIXME Issue #2899, call get_num_edges() instead.
                mask = G.edgelist.edgelist_df[G.edgeTypeCol] == edge_type
                return G.edgelist.edgelist_df[mask].count()
        # FIXME this should be valid for a graph without properties

    ###########################################################################
    # Algos
    def batched_ego_graphs(self, seeds, radius, graph_id):
        """ """
        # FIXME: finish docstring above
        # FIXME: exception handling
        G = self._get_graph(graph_id)
        # FIXME: write test to catch an MGPropertyGraph being passed in
        if isinstance(G, PropertyGraph):
            raise CugraphServiceError(
                "batched_ego_graphs() cannot operate "
                "directly on a graph with properties, "
                "call extract_subgraph() then call "
                "batched_ego_graphs() on the extracted "
                "subgraph instead."
            )
        try:
            # FIXME: update this to use call_algo()
            # FIXME: this should not be needed, need to update
            # cugraph.batched_ego_graphs to also accept a list
            seeds = cudf.Series(seeds, dtype="int32")
            (ego_edge_list, seeds_offsets) = batched_ego_graphs(G, seeds, radius)

            # batched_ego_graphs_result = BatchedEgoGraphsResult(
            #     src_verts=ego_edge_list["src"].values_host.tobytes(), #i32
            #     dst_verts=ego_edge_list["dst"].values_host.tobytes(), #i32
            #     edge_weights=ego_edge_list["weight"].values_host.tobytes(),
            #                                                             #f64
            #     seeds_offsets=seeds_offsets.values_host.tobytes() #i64
            # )
            batched_ego_graphs_result = BatchedEgoGraphsResult(
                src_verts=ego_edge_list["src"].values_host,
                dst_verts=ego_edge_list["dst"].values_host,
                edge_weights=ego_edge_list["weight"].values_host,
                seeds_offsets=seeds_offsets.values_host,
            )
            return batched_ego_graphs_result
        except Exception:
            raise CugraphServiceError(f"{traceback.format_exc()}")

        return batched_ego_graphs_result

    def node2vec(self, start_vertices, max_depth, graph_id):
        """ """
        # FIXME: finish docstring above
        # FIXME: exception handling
        G = self._get_graph(graph_id)
        # FIXME: write test to catch an MGPropertyGraph being passed in
        if isinstance(G, PropertyGraph):
            raise CugraphServiceError(
                "node2vec() cannot operate directly on "
                "a graph with properties, call "
                "extract_subgraph() then call "
                "node2vec() on the extracted subgraph "
                "instead."
            )

        try:
            # FIXME: update this to use call_algo()
            # FIXME: this should not be needed, need to update cugraph.node2vec
            # to also accept a list
            start_vertices = cudf.Series(start_vertices, dtype="int32")

            (paths, weights, path_sizes) = node2vec(G, start_vertices, max_depth)

            node2vec_result = Node2vecResult(
                vertex_paths=paths.values_host,
                edge_weights=weights.values_host,
                path_sizes=path_sizes.values_host,
            )
        except Exception:
            raise CugraphServiceError(f"{traceback.format_exc()}")

        return node2vec_result

    def uniform_neighbor_sample(
        self,
        start_list,
        fanout_vals,
        with_replacement,
        graph_id,
        result_host,
        result_port,
    ):
        print("SERVER: running uns", flush=True)
        try:
            G = self._get_graph(graph_id)
            if isinstance(G, (MGPropertyGraph, PropertyGraph)):
                # Implicitly extract a subgraph containing the entire multigraph.
                # G will be garbage collected when this function returns.
                G = G.extract_subgraph(
                    create_using=MultiGraph(directed=True),
                    default_edge_weight=1.0,
                )

            print("SERVER: starting sampling...")
            st = time.perf_counter_ns()
            uns_result = call_algo(
                uniform_neighbor_sample,
                G,
                start_list=start_list,
                fanout_vals=fanout_vals,
                with_replacement=with_replacement,
            )
            print(
                f"SERVER: done sampling, took {((time.perf_counter_ns() - st) / 1e9)}s"
            )

            if self.__check_host_port_args(result_host, result_port):
                print("SERVER: calling ucx_send_results...")
                st = time.perf_counter_ns()
                asyncio.run(
                    self.__ucx_send_results(
                        result_host,
                        result_port,
                        uns_result.sources,
                        uns_result.destinations,
                        uns_result.indices,
                    )
                )
                print(
                    "SERVER: done ucx_send_results, took "
                    f"{((time.perf_counter_ns() - st) / 1e9)}s"
                )
                # FIXME: Thrift still expects something of the expected type to
                # be returned to be serialized and sent. Look into a separate
                # API that uses the Thrift "oneway" modifier when returning
                # results via client device.
                return UniformNeighborSampleResult()

            else:
                uns_result.sources = cp.asnumpy(uns_result.sources)
                uns_result.destinations = cp.asnumpy(uns_result.destinations)
                uns_result.indices = cp.asnumpy(uns_result.indices)
                return uns_result

        except Exception:
            raise CugraphServiceError(f"{traceback.format_exc()}")

    def pagerank(self, graph_id):
        """ """
        raise NotImplementedError

    ###########################################################################
    # Test/Debug APIs
    def create_test_array(self, nbytes):
        """
        Creates an array of bytes (int8 values set to 1) and returns an ID to
        use to reference the array in later test calls.

        The test array must be deleted by calling delete_test_array().
        """
        aid = self.__next_test_array_id
        self.__test_arrays[aid] = cp.ones(nbytes, dtype="int8")
        self.__next_test_array_id += 1
        return aid

    def delete_test_array(self, test_array_id):
        """
        Deletes the test array identified by test_array_id.
        """
        a = self.__test_arrays.pop(test_array_id, None)
        if a is None:
            raise CugraphServiceError(f"invalid test_array_id {test_array_id}")
        del a

    def receive_test_array(self, test_array_id):
        """
        Returns the test array identified by test_array_id to the client.

        This can be used to verify transfer speeds from server to client are
        performing as expected.
        """
        return self.__test_arrays[test_array_id]

    def receive_test_array_to_device(self, test_array_id, result_host, result_port):
        """
        Returns the test array identified by test_array_id to the client via
        UCX-Py listening on result_host/result_port.

        This can be used to verify transfer speeds from server to client are
        performing as expected.
        """
        asyncio.run(
            self.__ucx_send_results(
                result_host, result_port, self.__test_arrays[test_array_id]
            )
        )

    def get_graph_type(self, graph_id):
        """
        Returns a string repr of the graph type associated with graph_id.
        """
        return repr(type(self._get_graph(graph_id)))

    ###########################################################################
    # "Protected" interface - used for both implementation and test/debug. Will
    # not be exposed to a cugraph_service client, but will be used by extensions
    # via the ExtensionServerFacade.
    def _add_graph(self, G):
        """
        Create a new graph ID for G and add G to the internal mapping of
        graph ID:graph instance.
        """
        gid = self.__next_graph_id
        self.__graph_objs[gid] = G
        self.__next_graph_id += 1
        return gid

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
                raise CugraphServiceError(f"invalid graph_id {graph_id}")

        return pG

    ###########################################################################
    # Private

    def __parse_create_using_string(self, create_using):
        match = re.match(r"([MultiGraph|Graph]+)(\(.*\))?", create_using)
        if match is None:
            raise TypeError(f"Invalid graph type {create_using}")
        else:
            graph_type, args = match.groups()
            args_dict = {}
            if args is not None and args != "" and args != "()":
                for arg in args[1:-1].replace(" ", "").split(","):
                    try:
                        k, v = arg.split("=")
                        if v == "True":
                            args_dict[k] = True
                        elif v == "False":
                            args_dict[k] = False
                        else:
                            raise ValueError(f"Could not parse value {v}")
                    except Exception as e:
                        raise ValueError(f"Could not parse argument {arg}", e)

            if graph_type == "Graph":
                graph_type = Graph
            else:
                graph_type = MultiGraph

            return graph_type(**args_dict)

    @staticmethod
    def __check_host_port_args(result_host, result_port):
        """
        Return True if host and port are set correctly, False if not set, and raise
        ValueError if set incorrectly.
        """
        if (result_host is not None) or (result_port is not None):
            if (result_host is None) or (result_port is None):
                raise ValueError(
                    "both result_host and result_port must be set if either is set. "
                    f"Got: {result_host=}, {result_port=}"
                )
            return True
        return False

    @staticmethod
    def __get_extension_files_from_path(extension_dir_or_mod_path):
        extension_path = Path(extension_dir_or_mod_path)
        # extension_dir_path is either a path on disk or an importable module path
        # (eg. import foo.bar.module)
        if (not extension_path.exists()) or (not extension_path.is_dir()):
            try:
                mod = importlib.import_module(str(extension_path))
            except ModuleNotFoundError:
                raise CugraphServiceError(f"bad path: {extension_dir_or_mod_path}")

            mod_file_path = Path(mod.__file__).absolute()

            # If mod is a package, find all the .py files in it
            if mod_file_path.name == "__init__.py":
                extension_files = mod_file_path.parent.glob("*.py")
            else:
                extension_files = [mod_file_path]
        else:
            extension_files = extension_path.glob("*_extension.py")

        return extension_files

    async def __ucx_send_results(self, result_host, result_port, *results):
        # The cugraph_service_client should have set up a UCX listener waiting
        # for the result. Create an endpoint, send results, and close.
        ep = await ucp.create_endpoint(result_host, result_port)
        for r in results:
            await ep.send_obj(r)
        await ep.close()

    def __get_dataframe_from_csv(self, csv_file_name, delimiter, dtypes, header, names):

        """
        Read a CSV into a DataFrame and return it. This will use either a cuDF
        DataFrame or a dask_cudf DataFrame based on if the handler is
        configured to use a dask cluster or not.
        """
        gdf = cudf.read_csv(
            csv_file_name, delimiter=delimiter, dtype=dtypes, header=header, names=names
        )
        if self.is_multi_gpu:
            return dask_cudf.from_cudf(gdf, npartitions=self.num_gpus)

        return gdf

    def __create_graph(self):
        """
        Instantiate a graph object using a type appropriate for the handler (
        either SG or MG)
        """
        return MGPropertyGraph() if self.is_multi_gpu else PropertyGraph()

    # FIXME: consider adding this to PropertyGraph
    def __remove_internal_columns(self, pg_column_names):
        """
        Removes all column names from pg_column_names that are "internal" (ie.
        used for PropertyGraph bookkeeping purposes only)
        """
        internal_column_names = [
            PropertyGraph.vertex_col_name,
            PropertyGraph.src_col_name,
            PropertyGraph.dst_col_name,
            PropertyGraph.type_col_name,
            PropertyGraph.edge_id_col_name,
            PropertyGraph.vertex_id_col_name,
            PropertyGraph.weight_col_name,
        ]

        # Create a list of user-visible columns by removing the internals while
        # preserving order
        user_visible_column_names = list(pg_column_names)
        for internal_column_name in internal_column_names:
            if internal_column_name in user_visible_column_names:
                user_visible_column_names.remove(internal_column_name)

        return user_visible_column_names

    # FIXME: consider adding this to PropertyGraph
    def __get_edge_IDs_from_graph_edge_data(self, G, src_vert_IDs, dst_vert_IDs):
        """
        Return a list of edge IDs corresponding to the vertex IDs in each of
        src_vert_IDs and dst_vert_IDs that, when combined, define an edge in G.

        For example, if src_vert_IDs is [0, 1, 2] and dst_vert_IDs is [7, 8, 9]
        return the edge IDs for edges (0, 7), (1, 8), and (2, 9).

        G must have an "edge_data" attribute.
        """
        edge_IDs = []
        num_edges = len(src_vert_IDs)

        for i in range(num_edges):
            src_mask = G.edge_data[PropertyGraph.src_col_name] == src_vert_IDs[i]
            dst_mask = G.edge_data[PropertyGraph.dst_col_name] == dst_vert_IDs[i]
            value = G.edge_data[src_mask & dst_mask][PropertyGraph.edge_id_col_name]

            # FIXME: This will compute the result (if using dask) then transfer
            # to host memory for each iteration - is there a more efficient
            # way?
            if self.is_multi_gpu:
                value = value.compute()
            edge_IDs.append(value.values_host[0])

        return edge_IDs

    def __get_graph_data_as_numpy_bytes(self, dataframe, null_replacement_value):
        """
        Returns a byte array repr of the vertex or edge graph data. Since the
        byte array cannot represent NA values, null_replacement_value must be
        provided to be used in place of NAs.
        """
        try:
            if dataframe is None:
                return np.ndarray(shape=(0, 0)).dumps()

            # null_replacement_value is a Value "union"
            n = ValueWrapper(null_replacement_value).get_py_obj()

            # This needs to be a copy of the df data to replace NA values
            # FIXME: should something other than a numpy type be serialized to
            # prevent a copy? (note: any other type required to be de-serialzed
            # on the client end could add dependencies on the client)
            df_numpy = dataframe.to_numpy(na_value=n)
            return df_numpy.dumps()

        except Exception:
            raise CugraphServiceError(f"{traceback.format_exc()}")

    def __call_extension(
        self, extension_dict, func_name, func_args_repr, func_kwargs_repr
    ):
        """
        Calls the extension function func_name and passes it the eval'd
        func_args_repr and func_kwargs_repr objects. If successful, returns a
        Value object containing the results returned by the extension function.

        The arg/kwarg reprs are eval'd prior to calling in order to pass actual
        python objects to func_name (this is needed to allow arbitrary arg
        objects to be serialized as part of the RPC call from the
        client).

        func_name cannot be a private name (name starting with __).

        All loaded extension modules are checked when searching for func_name,
        and the first extension module that contains it will have its function
        called.
        """
        if func_name.startswith("__"):
            raise CugraphServiceError(f"Cannot call private function {func_name}")

        for module in extension_dict.values():
            func = getattr(module, func_name, None)
            if func is not None:
                # FIXME: look for a way to do this without using eval()
                func_args = eval(func_args_repr)
                func_kwargs = eval(func_kwargs_repr)
                func_sig = signature(func)
                func_params = list(func_sig.parameters.keys())
                facade_param = self.__server_facade_extension_param_name

                # Graph creation extensions that have the last arg named
                # self.__server_facade_extension_param_name are passed a
                # ExtensionServerFacade instance to allow them to query the
                # "server" in a safe way, if needed.
                if facade_param in func_params:
                    if func_params[-1] == facade_param:
                        func_kwargs[facade_param] = ExtensionServerFacade(self)
                    else:
                        raise CugraphServiceError(
                            f"{facade_param}, if specified, must be the last param."
                        )
                try:
                    return func(*func_args, **func_kwargs)
                except Exception:
                    # FIXME: raise a more detailed error
                    raise CugraphServiceError(
                        f"error running {func_name} : {traceback.format_exc()}"
                    )

        raise CugraphServiceError(f"extension {func_name} was not found")
