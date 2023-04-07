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

from cugraph_service_client.remote_graph_utils import import_optional, MissingModule

import numpy as np

from functools import wraps
from collections.abc import Sequence
import pickle
import ucp
import asyncio
import threading


from cugraph_service_client import defaults
from cugraph_service_client.remote_graph import RemoteGraph
from cugraph_service_client import extension_return_dtype_map
from cugraph_service_client.types import (
    ValueWrapper,
    GraphVertexEdgeID,
    UniformNeighborSampleResult,
)
from cugraph_service_client.cugraph_service_thrift import create_client

cp = import_optional("cupy")
cudf = import_optional("cudf")
pandas = import_optional("pandas")

cupy_installed = not isinstance(cp, MissingModule)
cudf_installed = not isinstance(cudf, MissingModule)
pandas_installed = not isinstance(pandas, MissingModule)


class RunAsyncioThread(threading.Thread):
    """
    This class provides a thread whose purpose is to start a new
    event loop and call the provided function.
    """

    def __init__(self, func, args, kwargs):
        """
        Parameters
        ----------
        func : function
            The function that will be run.
        *args : args
            The arguments to the given function.
        **kwargs : kwargs
            The keyword arguments to the given function.
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        super().__init__()

    def run(self):
        """
        Runs this thread's previously-provided function inside
        a new event loop.  Returns the result.

        Returns
        -------
        The returned object of the previously-provided function.
        """
        self.result = asyncio.run(self.func(*self.args, **self.kwargs))


def run_async(func, *args, **kwargs):
    """
    If no loop is running on the current thread,
    this method calls func using a new event
    loop using asyncio.run.  If a loop is running, this
    method starts a new thread, and calls func on a new
    event loop in the new thread.

    Parameters
    ----------
    func : function
        The function that will be run.
    *args : args
        The arguments to the given function.
    **kwargs : kwargs
        The keyword arguments to the given function.

    Returns
    -------
    The output of the given function.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        thread = RunAsyncioThread(func, args, kwargs)
        thread.start()
        thread.join()
        return thread.result
    else:
        return asyncio.run(func(*args, **kwargs))


class DeviceArrayAllocator:
    """
    This class is used to create a callable instance for allocating a cupy
    array on a specific device. It is constructed with a particular device
    number, and can be called repeatedly with the number of bytes to allocate,
    returning an array of the requested size on the device.
    """

    def __init__(self, device):
        self.device = device

    def __call__(self, nbytes):
        with cp.cuda.Device(self.device):
            a = cp.empty(nbytes, dtype="uint8")
        return a


class CugraphServiceClient:
    """
    Client object for cugraph_service, which defines the API that clients can
    use to access the cugraph_service server.
    """

    def __init__(
        self, host=defaults.host, port=defaults.port, results_port=defaults.results_port
    ):
        """
        Creates a connection to a cugraph_service server running on host/port.

        Parameters
        ----------
        host : string, defaults to 127.0.0.1
            Hostname where the cugraph_service server is running

        port : int, defaults to 9090
            Port number where the cugraph_service server is listening

        Returns
        -------
        CugraphServiceClient object

        Examples
        --------
        >>> from cugraph_service_client import CugraphServiceClient
        >>> client = CugraphServiceClient()
        """
        self.host = host
        self.port = port
        self.results_port = results_port
        self.__client = None

        # If True, do not automatically close a server connection upon
        # completion or error of a server API call. This requires the caller to
        # manually call close() when done.
        self.hold_open = False

    def __del__(self):
        self.close()

    def __server_connection(method):
        """
        Decorator for methods that require a connection to the server to be
        created prior to calling a server function, then closed upon completion
        or error. If self.hold_open is True, the automatic call to close() will
        not take place, allowing for multiple subsequent server calls to be
        made using the same connection. self.hold_open therefore requires the
        caller to manually call close() in order to allow other clients to
        connect.
        """

        @wraps(method)
        def wrapped_method(self, *args, **kwargs):
            self.open()
            try:
                ret_val = method(self, *args, **kwargs)
            finally:
                if not self.hold_open:
                    self.close()
            return ret_val

        return wrapped_method

    def open(self, call_timeout=900000):
        """
        Opens a connection to the server at self.host/self.port if one is not
        already established. close() must be called in order to allow other
        connections from other clients to be made.

        This call does nothing if a connection to the server is already open.

        Note: all APIs that access the server will call this method
        automatically, followed automatically by a call to close(), so calling
        this method should not be necessary. close() is not automatically
        called if self.hold_open is False.

        Parameters
        ----------
        call_timeout : int (default is 900000)
            Time in millisecods that calls to the server using this open
            connection must return by.

        Returns
        -------
        None

        Examples
        --------
        >>> from cugraph_service_client import CugraphServiceClient
        >>> client = CugraphServiceClient()
        >>> # Manually open a connection. The connection is held open and other
        >>> # clients cannot connect until a client API call completes or
        >>> # close() is manually called.
        >>> client.open()

        """
        if self.__client is None:
            self.__client = create_client(
                self.host, self.port, call_timeout=call_timeout
            )

    def close(self):
        """
        Closes a connection to the server if one has been established, allowing
        other clients to access the server. This method is called automatically
        for all APIs that access the server if self.hold_open is False.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Examples
        --------
        >>> from cugraph_service_client import CugraphServiceClient
        >>> client = CugraphServiceClient()
        >>> # Have the client hold open the connect automatically opened as
        >>> # part of a server API call until close() is called. This is
        >>> # normally not necessary and shown here for demonstration purposes.
        >>> client.hold_open = True
        >>> client.node2vec([0,1], 2)
        >>> # close the connection so other clients can connect
        >>> client.close()
        >>> # go back to automatic open/close mode (safer)
        >>> client.hold_open = False
        """
        if self.__client is not None:
            self.__client.close()
            self.__client = None

    ###########################################################################
    # Environment management
    @__server_connection
    def uptime(self):
        """
        Return the server uptime in seconds. This is often used as a "ping".

        Parameters
        ----------
        None

        Returns
        -------
        uptime : int
            The time in seconds the server has been running.

        Examples
        --------
        >>> from cugraph_service_client import CugraphServiceClient
        >>> client = CugraphServiceClient()
        >>> client.uptime()
        >>> 32
        """
        return self.__client.uptime()

    @__server_connection
    def get_server_info(self):
        """
        Return a dictionary of information about the server.

        Parameters
        ----------
        None

        Returns
        -------
        server_info : dict

            Dictionary containing environment and state information about the
            server.

        Examples
        --------
        >>> from cugraph_service_client import CugraphServiceClient
        >>> client = CugraphServiceClient()
        >>> client.get_server_info()
        >>> {'num_gpus': 2}
        """
        server_info = self.__client.get_server_info()
        # server_info is a dictionary of Value objects ("union" types returned
        # from the server), so convert them to simple py types.
        return dict((k, ValueWrapper(server_info[k]).get_py_obj()) for k in server_info)

    @__server_connection
    def load_graph_creation_extensions(self, extension_dir_path):
        """
        Loads the extensions for graph creation present in the directory
        specified by extension_dir_path.

        Parameters
        ----------
        extension_dir_path : string
            Path to the directory containing the extension files (.py source
            files). This directory must be readable by the server.

        Returns
        -------
        extension_modnames : list
            List of the module names loaded. These can be used in calls to
            unload_extension_module()

        Examples
        --------
        >>> from cugraph_service_client import CugraphServiceClient
        >>> client = CugraphServiceClient()
        >>> extension_modnames = client.load_graph_creation_extensions(
        ... "/some/server/side/directory")
        >>>
        """
        return self.__client.load_graph_creation_extensions(extension_dir_path)

    @__server_connection
    def load_extensions(self, extension_dir_or_mod_path):
        """
        Loads the extensions present in the directory (path on disk), or module or
        package path (as used in an import statement) specified by
        extension_dir_or_mod_path.

        Parameters
        ----------
        extension_dir_or_mod_path : string
            Path to the directory containing the extension files (.py source
            files), or an importable module or package path (eg. my.package or
            my.package.module). If a directory is specified it must be readable
            by the server, and if a module or package path is specified it must
            be importable by the server (ie. present in the sys.path of the
            running server).

        Returns
        -------
        extension_modnames : list
            List of the module names loaded as paths to files on disk. These can
            be used in calls to unload_extension_module()

        Examples
        --------
        >>> from cugraph_service_client import CugraphServiceClient
        >>> client = CugraphServiceClient()
        >>> extension_modnames = client.load_graph_creation_extensions(
        ... "/some/server/side/directory")
        >>> more_extension_modnames = client.load_graph_creation_extensions(
        ... "my_project.extensions.etl")
        """
        return self.__client.load_extensions(extension_dir_or_mod_path)

    @__server_connection
    def unload_extension_module(self, modname):
        """
        Removes all extensions contained in the modname module.

        Parameters
        ----------
        modname : string
            Name of the module to be unloaded. All extension functions contained in
            modname will no longer be callable.

        Returns
        -------
        None

        Examples
        --------
        >>> from cugraph_service_client import CugraphServiceClient
        >>> client = CugraphServiceClient()
        >>> ext_mod_name = client.load_graph_creation_extensions(
        ...                    "/some/server/side/directory")
        >>> client.unload_extension_module(ext_mod_name)
        >>>
        """
        return self.__client.unload_extension_module(modname)

    @__server_connection
    def call_graph_creation_extension(self, func_name, *func_args, **func_kwargs):
        """
        Calls a graph creation extension on the server that was previously
        loaded by a prior call to load_graph_creation_extensions(), then
        returns the graph ID of the graph created by the extension.

        Parameters
        ----------
        func_name : string
            The name of the server-side extension function loaded by a prior
            call to load_graph_creation_extensions(). All graph creation
            extension functions are expected to return a new graph.

        *func_args : string, int, list, dictionary (optional)
            The positional args to pass to func_name. Note that func_args are
            converted to their string representation using repr() on the
            client, then restored to python objects on the server using eval(),
            and therefore only objects that can be restored server-side with
            eval() are supported.

        **func_kwargs : string, int, list, dictionary
            The keyword args to pass to func_name. Note that func_kwargs are
            converted to their string representation using repr() on the
            client, then restored to python objects on the server using eval(),
            and therefore only objects that can be restored server-side with
            eval() are supported.

        Returns
        -------
        graph_id : int
            unique graph ID

        Examples
        --------
        >>> from cugraph_service_client import CugraphServiceClient
        >>> client = CugraphServiceClient()
        >>> # Load the extension file containing "my_complex_create_graph()"
        >>> client.load_graph_creation_extensions("/some/server/side/dir")
        >>> new_graph_id = client.call_graph_creation_extension(
        ... "my_complex_create_graph",
        ... "/path/to/csv/on/server/graph.csv",
        ... clean_data=True)
        >>>
        """
        func_args_repr = repr(func_args)
        func_kwargs_repr = repr(func_kwargs)
        return self.__client.call_graph_creation_extension(
            func_name, func_args_repr, func_kwargs_repr
        )

    @__server_connection
    def call_extension(
        self,
        func_name,
        *func_args,
        result_device=None,
        **func_kwargs,
    ):
        """
        Calls an extension on the server that was previously loaded by a prior
        call to load_extensions(), then returns the result returned by the
        extension.

        Parameters
        ----------
        func_name : string
            The name of the server-side extension function loaded by a prior
            call to load_graph_creation_extensions(). All graph creation
            extension functions are expected to return a new graph.

        *func_args : string, int, list, dictionary (optional)
            The positional args to pass to func_name. Note that func_args are
            converted to their string representation using repr() on the
            client, then restored to python objects on the server using eval(),
            and therefore only objects that can be restored server-side with
            eval() are supported.

        **func_kwargs : string, int, list, dictionary The keyword args to pass
            to func_name. func_kwargs are converted to their string
            representation using repr() on the client, then restored to python
            objects on the server using eval(), and therefore only objects that
            can be restored server-side with eval() are supported.

            result_device is reserved for use in specifying an optional GPU
            device ID to have the server transfer results to.

        result_device : int, default is None
            If specified, must be the integer ID of a GPU device to have the
            server transfer results to as one or more cupy ndarrays

        Returns
        -------
        result : python int, float, string, list
            The result returned by the extension

        Examples
        --------
        >>> from cugraph_service_client import CugraphServiceClient
        >>> client = CugraphServiceClient()
        >>> # Load the extension file containing "my_serverside_function()"
        >>> client.load_extensions("/some/server/side/dir")
        >>> result = client.call_extension(
        ... "my_serverside_function", 33, 22, "some_string")
        >>>
        """
        func_args_repr = repr(func_args)
        func_kwargs_repr = repr(func_kwargs)
        if result_device is not None:
            result_obj = asyncio.run(
                self.__call_extension_to_device(
                    func_name, func_args_repr, func_kwargs_repr, result_device
                )
            )
            # result_obj is a cupy array or tuple of cupy arrays on result_device
            return result_obj
        else:
            result_obj = self.__client.call_extension(
                func_name,
                func_args_repr,
                func_kwargs_repr,
                client_host=None,
                client_result_port=None,
            )
            # Convert the structure returned from the RPC call to a python type
            # FIXME: ValueWrapper ctor and get_py_obj are recursive and could be slow,
            # especially if Value is a list. Consider returning the Value obj as-is.
            return ValueWrapper(result_obj).get_py_obj()

    ###########################################################################
    # Graph management
    @__server_connection
    def create_graph(self):
        """
        Create a new graph associated with a new (non-default) unique graph ID,
        return the new graph ID.

        Parameters
        ----------
        None

        Returns
        -------
        graph_id : int
            unique graph ID

        Examples
        --------
        >>> from cugraph_service_client import CugraphServiceClient
        >>> client = CugraphServiceClient()
        >>> my_graph_id = client.create_graph()
        >>> # Load a CSV to the new graph
        >>> client.load_csv_as_edge_data(
        ... "edges.csv", ["int32", "int32", "float32"],
        ... vertex_col_names=["src", "dst"], graph_id=my_graph_id)
        >>>
        """
        return self.__client.create_graph()

    @__server_connection
    def delete_graph(self, graph_id):
        """
        Deletes the graph referenced by graph_id.

        Parameters
        ----------
        graph_id : int
            The graph ID to delete. If the ID passed is not valid on the
            server, CugraphServiceError is raised.

        Returns
        -------
        None

        Examples
        --------
        >>> from cugraph_service_client import CugraphServiceClient
        >>> client = CugraphServiceClient()
        >>> my_graph_id = client.create_graph()
        >>> # Load a CSV to the new graph
        >>> client.load_csv_as_edge_data(
        ... "edges.csv", ["int32", "int32", "float32"],
        ... vertex_col_names=["src", "dst"], graph_id=my_graph_id)
        >>> # Remove the graph instance on the server and reclaim the memory
        >>> client.delete_graph(my_graph_id)
        """
        return self.__client.delete_graph(graph_id)

    def graph(self):
        """
        Constructs a new RemoteGraph object wrapping a remote PropertyGraph.
        """
        return RemoteGraph(self, self.create_graph())

    @__server_connection
    def get_graph_ids(self):
        """
        Returns a list of all graph IDs the server is currently maintaining.

        Parameters
        ----------
        None

        Returns
        -------
        graph_id_list : list of unique int graph IDs

        Examples
        --------
        >>> from cugraph_service_client import CugraphServiceClient
        >>> client = CugraphServiceClient()
        >>> # This server already has graphs loaded from other sessions
        >>> client.get_graph_ids()
        [0, 26]
        >>>
        """
        return self.__client.get_graph_ids()

    @__server_connection
    def get_graph_info(self, keys=None, graph_id=defaults.graph_id):
        """
        Returns a dictionary containing meta-data about the graph referenced by
        graph_id (or the default graph if not specified).

        Parameters
        ----------
        graph_id : int, default is defaults.graph_id
            The graph ID to apply the properties in the CSV to. If not provided
            the default graph ID is used.

        Returns
        -------
        None

        Examples
        --------
        >>> from cugraph_service_client import CugraphServiceClient
        >>> client = CugraphServiceClient()
        >>> client.load_csv_as_vertex_data(
        ... "/server/path/to/vertex_data.csv",
        ... dtypes=["int32", "string", "int32"],
        ... vertex_col_name="vertex_id",
        ... header="infer")
        >>> client.get_graph_info()
        {'num_edges': 3, 'num_vertices': 4}
        """
        # Ensure keys is a list of strings when passing to RPC API
        if keys is None:
            keys = []
        elif isinstance(keys, str):
            keys = [keys]
        elif isinstance(keys, list):
            if False in [isinstance(k, str) for k in keys]:
                raise TypeError(f"keys must be a list of strings, got {keys}")
        else:
            raise TypeError(
                "keys must be a string or list of strings, got " f"{type(keys)}"
            )

        graph_info = self.__client.get_graph_info(keys, graph_id)

        # special case: if only one key was specified, return only the single
        # value
        if len(keys) == 1:
            return ValueWrapper(graph_info[keys[0]]).get_py_obj()

        # graph_info is a dictionary of Value objects ("union" types returned
        # from the graph), so convert them to simple py types.
        return dict((k, ValueWrapper(graph_info[k]).get_py_obj()) for k in graph_info)

    @__server_connection
    def load_csv_as_vertex_data(
        self,
        csv_file_name,
        dtypes,
        vertex_col_name,
        delimiter=" ",
        header=None,
        type_name="",
        property_columns=None,
        graph_id=defaults.graph_id,
        names=None,
    ):

        """
        Reads csv_file_name and applies it as vertex data to the graph
        identified as graph_id (or the default graph if not specified).

        Parameters
        ----------
        csv_file_name : string
            Path to CSV file on the server

        dtypes : list of strings
            Types for the columns in the CSV file

        vertex_col_name : string
            Name of the column to use as the vertex ID

        delimiter : string, default is " "
            Character that serves as the delimiter between columns in the CSV

        header : int, default is None
            Row number to use as the column names. Default behavior is to
            assume column names are explicitely provided (header=None).
            header="infer" if the column names are to be inferred. If no names
            are passed, header=0. See also cudf.read_csv

        type_name : string, default is ""
            The vertex property "type" the CSV data is describing. For
            instance, CSV data describing properties for "users" might pass
            type_name as "user". A vertex property type is optional.

        property_columns : list of strings, default is None
            The column names in the CSV to add as vertex properties. If None,
            all columns will be added as properties.

        graph_id : int, default is defaults.graph_id
            The graph ID to apply the properties in the CSV to. If not provided
            the default graph ID is used.

        names: list of strings, default is None
            The names to be used to reference the CSV columns, in lieu of a
            header.

        Returns
        -------
        None

        Examples
        --------
        >>> from cugraph_service_client import CugraphServiceClient
        >>> client = CugraphServiceClient()
        >>> client.load_csv_as_vertex_data(
        ... "/server/path/to/vertex_data.csv",
        ... dtypes=["int32", "string", "int32"],
        ... vertex_col_name="vertex_id",
        ... header="infer")
        >>>
        """
        # Map all int arg types that also have string options to ints
        # FIXME: check for invalid header arg values
        if header == "infer":
            header = -1
        elif header is None:
            header = -2
        return self.__client.load_csv_as_vertex_data(
            csv_file_name,
            delimiter,
            dtypes,
            header,
            vertex_col_name,
            type_name,
            property_columns or [],
            graph_id,
            names or [],
        )

    @__server_connection
    def load_csv_as_edge_data(
        self,
        csv_file_name,
        dtypes,
        vertex_col_names,
        delimiter=" ",
        header=None,
        type_name="",
        property_columns=None,
        edge_id_col_name=None,
        graph_id=defaults.graph_id,
        names=None,
    ):
        """
        Reads csv_file_name and applies it as edge data to the graph identified
        as graph_id (or the default graph if not specified).

        Parameters
        ----------
        csv_file_name : string
            Path to CSV file on the server

        dtypes : list of strings
            Types for the columns in the CSV file

        vertex_col_names : tuple of strings
            Names of the columns to use as the source and destination vertex
            IDs defining the edges

        delimiter : string, default is " "
            Character that serves as the delimiter between columns in the CSV

        header : int, default is None
            Row number to use as the column names. Default behavior is to
            assume column names are explicitely provided (header=None).
            header="infer" if the column names are to be inferred. If no names
            are passed, header=0. See also cudf.read_csv

        type_name : string, default is ""
            The edge property "type" the CSV data is describing. For instance,
            CSV data describing properties for "transactions" might pass
            type_name as "transaction". An edge property type is optional.

        property_columns : list of strings, default is None
            The column names in the CSV to add as edge properties. If None, all
            columns will be added as properties.

        edge_id_col_name : string, optional
            The column name that contains the values to be used as edge IDs.
            If unspecified, edge IDs will be automatically assigned.
            Currently, all edge data must be added with the same method: either
            with automatically generated IDs, or from user-provided edge IDs.

        graph_id : int, default is defaults.graph_id
            The graph ID to apply the properties in the CSV to. If not provided
            the default graph ID is used.

        names: list of strings, default is None
            The names to be used to reference the CSV columns, in lieu of a
            header.

        Returns
        -------
        None

        Examples
        --------
        >>> from cugraph_service_client import CugraphServiceClient
        >>> client = CugraphServiceClient()
        >>> client.load_csv_as_edge_data(
        ... "/server/path/to/edge_data.csv",
        ... dtypes=["int32", "int32", "string", "int32"],
        ... vertex_col_names=("src", "dst"),
        ... header="infer")
        >>>
        """
        # Map all int arg types that also have string options to ints
        # FIXME: check for invalid header arg values
        if header == "infer":
            header = -1
        elif header is None:
            header = -2
        return self.__client.load_csv_as_edge_data(
            csv_file_name,
            delimiter,
            dtypes,
            header,
            vertex_col_names,
            type_name,
            property_columns or [],
            graph_id,
            names or [],
            edge_id_col_name or "",
        )

    @__server_connection
    def get_edge_IDs_for_vertices(
        self, src_vert_IDs, dst_vert_IDs, graph_id=defaults.graph_id
    ):
        """ """
        # FIXME: finish docstring above
        # FIXME: add type checking
        return self.__client.get_edge_IDs_for_vertices(
            src_vert_IDs, dst_vert_IDs, graph_id
        )

    @__server_connection
    def renumber_vertices_by_type(
        self, prev_id_column=None, graph_id=defaults.graph_id
    ):
        """
        Renumbers the vertices in the graph referenced by graph id to be contiguous
        by vertex type.  Returns the start and end vertex id ranges for each type.
        """
        if prev_id_column is None:
            prev_id_column = ""
        return self.__client.renumber_vertices_by_type(prev_id_column, graph_id)

    @__server_connection
    def renumber_edges_by_type(self, prev_id_column=None, graph_id=defaults.graph_id):
        """
        Renumbers the edges in the graph referenced by graph id to be contiguous
        by edge type.  Returns the start and end edge id ranges for each type.
        """
        if prev_id_column is None:
            prev_id_column = ""
        return self.__client.renumber_edges_by_type(prev_id_column, graph_id)

    @__server_connection
    def extract_subgraph(
        self,
        create_using=None,
        selection=None,
        edge_weight_property="",
        default_edge_weight=1.0,
        check_multi_edges=True,
        renumber_graph=True,
        add_edge_data=True,
        graph_id=defaults.graph_id,
    ):
        """
        Return a graph ID for a subgraph of the graph referenced by graph_id
        that containing vertices and edges that match a selection.

        Parameters
        ----------
        create_using : string, default is None
            String describing the type of Graph object to create from the
            selected subgraph of vertices and edges. The default (None) results
            in a directed cugraph.MultiGraph object.

        selection : int, default is None
            A PropertySelection ID returned from one or more calls to
            select_vertices() and/or select_edges(), used for creating a Graph
            with only the selected properties. If not speciied the resulting
            Graph will have all properties. Note, this could result in a Graph
            with multiple edges, which may not be supported based on the value
            of create_using.

        edge_weight_property : string, default is ""
            The name of the property whose values will be used as weights on
            the returned Graph. If not specified, the returned Graph will be
            unweighted.

        default_edge_weight : float, default is 1.0
            The value to use when an edge property is specified but not present
            on an edge.

        check_multi_edges : bool (default is True)
            When True and create_using argument is given and not a MultiGraph,
            this will perform an expensive check to verify that the edges in
            the edge dataframe do not form a multigraph with duplicate edges.

        graph_id : int, default is defaults.graph_id
           The graph ID to extract the subgraph from. If the ID passed is not
           valid on the server, CugraphServiceError is raised.

        Returns
        -------
        A graph ID for a new Graph instance of the same type as create_using
        containing only the vertices and edges resulting from applying the
        selection to the set of vertex and edge property data.

        Examples
        --------
        >>>
        """
        # FIXME: finish docstring above

        # FIXME: convert defaults to type needed by the Thrift API. These will
        # be changing to different types.
        create_using = create_using or ""
        selection = selection or ""

        return self.__client.extract_subgraph(
            create_using,
            selection,
            edge_weight_property,
            default_edge_weight,
            check_multi_edges,
            renumber_graph,
            add_edge_data,
            graph_id,
        )

    @__server_connection
    def get_graph_vertex_data(
        self,
        id_or_ids=-1,
        null_replacement_value=0,
        property_keys=None,
        types=None,
        graph_id=defaults.graph_id,
    ):
        """
        Returns ...

        Parameters
        ----------
        id_or_ids : int or list of ints (default -1)

        null_replacement_value : number or string (default 0)

        property_keys : list of strings (default [])
            The keys (names) of properties to retrieve.  If omitted, returns
            the whole dataframe.

        types : list of strings (default [])
            The vertex types to include in the query.  If ommitted, returns
            properties for all types.

        graph_id : int, default is defaults.graph_id
           The graph ID to extract the subgraph from. If the ID passed is not
           valid on the server, CugraphServiceError is raised.

        Returns
        -------

        Examples
        --------
        >>>
        """
        # FIXME: finish docstring above

        vertex_edge_id_obj = self.__get_vertex_edge_id_obj(id_or_ids)
        null_replacement_value_obj = ValueWrapper(
            null_replacement_value, val_name="null_replacement_value"
        ).union

        ndarray_bytes = self.__client.get_graph_vertex_data(
            vertex_edge_id_obj,
            null_replacement_value_obj,
            property_keys or [],
            types or [],
            graph_id,
        )

        return pickle.loads(ndarray_bytes)

    @__server_connection
    def get_graph_edge_data(
        self,
        id_or_ids=-1,
        null_replacement_value=0,
        property_keys=None,
        types=None,
        graph_id=defaults.graph_id,
    ):
        """
        Returns ...

        Parameters
        ----------
        id_or_ids : int or list of ints (default -1)

        null_replacement_value : number or string (default 0)

        property_keys : list of strings (default [])
            The keys (names) of properties to retrieve.  If omitted, returns
            the whole dataframe.

        types : list of strings (default [])
            The types of edges to include in the query.  If ommitted, returns
            data for all edge types.

        graph_id : int, default is defaults.graph_id
           The graph ID to extract the subgraph from. If the ID passed is not
           valid on the server, CugraphServiceError is raised.

        Returns
        -------

        Examples
        --------
        >>>
        """
        # FIXME: finish docstring above

        vertex_edge_id_obj = self.__get_vertex_edge_id_obj(id_or_ids)
        null_replacement_value_obj = ValueWrapper(
            null_replacement_value, val_name="null_replacement_value"
        ).union

        ndarray_bytes = self.__client.get_graph_edge_data(
            vertex_edge_id_obj,
            null_replacement_value_obj,
            property_keys or [],
            types or [],
            graph_id,
        )

        return pickle.loads(ndarray_bytes)

    @__server_connection
    def is_vertex_property(self, property_key, graph_id=defaults.graph_id):
        """
        Returns True if the given property key is for a valid vertex property
        in the given graph, False otherwise.

        Parameters
        ----------
        property_key: string
            The key (name) of the vertex property to check
        graph_id: int
            The id of the graph of interest
        """
        return self.__client.is_vertex_property(property_key, graph_id)

    @__server_connection
    def is_edge_property(self, property_key, graph_id=defaults.graph_id):
        """
        Returns True if the given property key is for a valid vertex property
        in the given graph, false otherwise.e

        Parameters
        ----------
        property_key: string
            The key (name) of the vertex property to check
        graph_id: int
            The id of the graph of interest
        """
        return self.__client.is_edge_property(property_key, graph_id)

    @__server_connection
    def get_graph_vertex_property_names(self, graph_id=defaults.graph_id):
        """
        Returns a list of the vertex property names for the graph with
        the given graph id.

        Parameters
        ----------
        graph_id: int
            The id of the graph of interest
        """
        return self.__client.get_graph_vertex_property_names(graph_id)

    @__server_connection
    def get_graph_edge_property_names(self, graph_id=defaults.graph_id):
        """
        Returns a list of the edge property names for the graph with
        the given graph id.

        Parameters
        ----------
        graph_id: int
            The id of the graph of interest
        """
        return self.__client.get_graph_edge_property_names(graph_id)

    @__server_connection
    def get_graph_vertex_types(self, graph_id=defaults.graph_id):
        """
        Returns a list of the vertex type names for the graph with
        the given graph id.

        Parameters
        ----------
        graph_id: it
            The id of the graph of interest
        """
        return self.__client.get_graph_vertex_types(graph_id)

    @__server_connection
    def get_graph_edge_types(self, graph_id=defaults.graph_id):
        """
        Returns a list of the edge type names for the graph with
        the given graph id.

        Parameters
        ----------
        graph_id: int
            The id of the graph of interest
        """
        return self.__client.get_graph_edge_types(graph_id)

    @__server_connection
    def get_num_vertices(
        self, vertex_type=None, include_edge_data=True, graph_id=defaults.graph_id
    ):
        """
        Returns the number of vertices in the graph with the given
        graph id.

        Parameters
        ----------
        vertex_type: string
            The vertex type to count. If not defined, all types are counted.
        include_edge_data: bool
            Whether to include vertices added only as part of the edgelist.
        graph_id: int
            The id of the grpah of interest.
        """
        return self.__client.get_num_vertices(
            vertex_type or "", include_edge_data, graph_id
        )

    @__server_connection
    def get_num_edges(self, edge_type=None, graph_id=defaults.graph_id):
        """
        Returns the number of edges in the graph with the given
        graph id.

        Parameters
        ----------
        edge_type: string
            The edge type to count. If not defined, all types are counted.
        graph_id: int
            The id of the grpah of interest.
        """
        return self.__client.get_num_edges(edge_type or "", graph_id)

    ###########################################################################
    # Algos
    @__server_connection
    def batched_ego_graphs(self, seeds, radius=1, graph_id=defaults.graph_id):
        """
        Parameters
        ----------

        Returns
        -------

        Examples
        --------
        >>>
        """
        # FIXME: finish docstring above

        if not isinstance(seeds, list):
            seeds = [seeds]
        batched_ego_graphs_result = self.__client.batched_ego_graphs(
            seeds, radius, graph_id
        )

        return (
            batched_ego_graphs_result.src_verts,
            batched_ego_graphs_result.dst_verts,
            batched_ego_graphs_result.edge_weights,
            batched_ego_graphs_result.seeds_offsets,
        )

    @__server_connection
    def node2vec(self, start_vertices, max_depth, graph_id=defaults.graph_id):
        """
        Computes random walks for each node in 'start_vertices', under the
        node2vec sampling framework.

        Parameters
        ----------
        start_vertices: int or list or cudf.Series or cudf.DataFrame
            A single node or a list or a cudf.Series of nodes from which to run
            the random walks. In case of multi-column vertices it should be
            a cudf.DataFrame. Only supports int32 currently.

        max_depth: int
            The maximum depth of the random walks

        Returns
        -------

        Examples
        --------
        >>>
        """
        # FIXME: finish docstring above

        # start_vertices must be a list (cannot just be an iterable), and
        # assume return value is tuple of python lists on host.
        if not isinstance(start_vertices, list):
            start_vertices = [start_vertices]
        # FIXME: ensure list is a list of int32, since Thrift interface
        # specifies that?
        node2vec_result = self.__client.node2vec(start_vertices, max_depth, graph_id)
        return (
            node2vec_result.vertex_paths,
            node2vec_result.edge_weights,
            node2vec_result.path_sizes,
        )

    @__server_connection
    def uniform_neighbor_sample(
        self,
        start_list,
        fanout_vals,
        with_replacement=True,
        *,
        graph_id=defaults.graph_id,
        result_device=None,
    ):
        """
        Samples the graph and returns a UniformNeighborSampleResult instance.

        Parameters:
        start_list : list[int]

        fanout_vals : list[int]

        with_replacement : bool

        graph_id : int, default is defaults.graph_id

        result_device : int, default is None

        Returns
        -------
        result : UniformNeighborSampleResult
            Instance containing three CuPy device arrays.

            result.sources: CuPy array
                Contains the source vertices from the sampling result
            result.destinations: CuPy array
                Contains the destination vertices from the sampling result
            result.indices: CuPy array
                Contains the indices from the sampling result for path reconstruction
        """

        if result_device is not None:
            result_obj = run_async(
                self.__uniform_neighbor_sample_to_device,
                start_list,
                fanout_vals,
                with_replacement,
                graph_id,
                result_device,
            )

        else:
            result_obj = self.__client.uniform_neighbor_sample(
                start_list,
                fanout_vals,
                with_replacement,
                graph_id,
                client_host=None,
                client_result_port=None,
            )

        return result_obj

    @__server_connection
    def pagerank(self, graph_id=defaults.graph_id):
        """
        pagerank
        """
        raise NotImplementedError

    ###########################################################################
    # Test/Debug
    @__server_connection
    def _create_test_array(self, nbytes):
        """
        Creates an array of bytes (int8 values set to 1) on the server and
        returns an ID to use to reference the array in later test calls.

        The test array must be deleted on the server by calling
        _delete_test_array().
        """
        return self.__client.create_test_array(nbytes)

    @__server_connection
    def _delete_test_array(self, test_array_id):
        """
        Deletes the test array on the server identified by test_array_id.
        """
        self.__client.delete_test_array(test_array_id)

    @__server_connection
    def _receive_test_array(self, test_array_id, result_device=None):
        """
        Returns the array of bytes (int8 values set to 1) from the server,
        either to result_device or on the client host. The array returned must
        have been created by a prior call to create_test_array() which returned
        test_array_id.

        This can be used to verify transfer speeds from server to client are
        performing as expected.
        """
        if result_device is not None:
            return asyncio.run(
                self.__receive_test_array_to_device(test_array_id, result_device)
            )
        else:
            return self.__client.receive_test_array(test_array_id)

    @__server_connection
    def _get_graph_type(self, graph_id=defaults.graph_id):

        """
        Test/debug API for returning a string repr of the graph_id instance.
        """
        return self.__client.get_graph_type(graph_id)

    ###########################################################################
    # Private
    async def __receive_test_array_to_device(self, test_array_id, result_device):
        # Create an object to set results on in the "receiver" callback below.
        result_obj = type("Result", (), {})()
        allocator = DeviceArrayAllocator(result_device)

        async def receiver(endpoint):
            with cp.cuda.Device(result_device):
                result_obj.array = await endpoint.recv_obj(allocator=allocator)
                result_obj.array = result_obj.array.view("int8")

            await endpoint.close()
            listener.close()

        listener = ucp.create_listener(receiver, self.results_port)

        # This sends a one-way request to the server and returns
        # immediately. The server will create and send the array back to the
        # listener started above.
        self.__client.receive_test_array_to_device(
            test_array_id, self.host, self.results_port
        )

        while not listener.closed():
            await asyncio.sleep(0.05)

        return result_obj.array

    async def __uniform_neighbor_sample_to_device(
        self, start_list, fanout_vals, with_replacement, graph_id, result_device
    ):
        """
        Run uniform_neighbor_sample() with the args provided, but have the
        result send directly to the device specified by result_device.
        """
        # FIXME: check for valid device
        result_obj = UniformNeighborSampleResult()

        allocator = DeviceArrayAllocator(result_device)

        async def receiver(endpoint):
            with cp.cuda.Device(result_device):
                result_obj.sources = await endpoint.recv_obj(allocator=allocator)
                result_obj.sources = result_obj.sources.view("int32")
                result_obj.destinations = await endpoint.recv_obj(allocator=allocator)
                result_obj.destinations = result_obj.destinations.view("int32")
                result_obj.indices = await endpoint.recv_obj(allocator=allocator)
                result_obj.indices = result_obj.indices.view("float64")

            await endpoint.close()
            listener.close()

        listener = ucp.create_listener(receiver, self.results_port)

        # Use an excepthook to store an exception on the thread object if one is
        # raised in the thread.
        def excepthook(exc):
            if exc.thread is not None:
                exc.thread.exception = exc.exc_type(exc.exc_value)

        orig_excepthook = threading.excepthook
        threading.excepthook = excepthook

        thread = threading.Thread(
            target=self.__client.uniform_neighbor_sample,
            args=(
                start_list,
                fanout_vals,
                with_replacement,
                graph_id,
                self.host,
                self.results_port,
            ),
        )
        thread.start()

        # Poll the listener and the state of the thread. Close the listener if
        # the thread died and raise the stored exception.
        while not listener.closed():
            await asyncio.sleep(0.05)
            if not thread.is_alive():
                listener.close()
                threading.excepthook = orig_excepthook
                if hasattr(thread, "exception"):
                    raise thread.exception

        thread.join()
        return result_obj

    async def __call_extension_to_device(
        self, func_name, func_args_repr, func_kwargs_repr, result_device
    ):
        """
        Run the server-side extension func_name with the args/kwargs and have the
        result sent directly to the device specified by result_device.
        """
        # FIXME: there's probably a better way to do this, eg. create a class containing
        # both allocator and receiver that maintains results, devices, etc. that's
        # callable from the listener
        result = []

        # FIXME: check for valid device
        allocator = DeviceArrayAllocator(result_device)

        async def receiver(endpoint):
            # Format of data sent is assumed to be:
            # 1) a single array of length n describing the dtypes for the n arrays that
            #    follow
            # 2) n arrays
            with cp.cuda.Device(result_device):
                # First get the array describing the data
                # FIXME: meta_data doesn't need to be a cupy array
                dtype_meta_data = await endpoint.recv_obj(allocator=allocator)
                for dtype_enum in [int(i) for i in dtype_meta_data]:
                    # FIXME: safe to assume dtype_enum will always be valid?
                    dtype = extension_return_dtype_map[dtype_enum]
                    a = await endpoint.recv_obj(allocator=allocator)
                    result.append(a.view(dtype))

            await endpoint.close()
            listener.close()

        listener = ucp.create_listener(receiver, self.results_port)

        # Use an excepthook to store an exception on the thread object if one is
        # raised in the thread.
        def excepthook(exc):
            if exc.thread is not None:
                exc.thread.exception = exc.exc_type(exc.exc_value)

        orig_excepthook = threading.excepthook
        threading.excepthook = excepthook

        thread = threading.Thread(
            target=self.__client.call_extension,
            args=(
                func_name,
                func_args_repr,
                func_kwargs_repr,
                self.host,
                self.results_port,
            ),
        )
        thread.start()

        # Poll the listener and the state of the thread. Close the listener if
        # the thread died and raise the stored exception.
        while not listener.closed():
            await asyncio.sleep(0.05)
            if not thread.is_alive():
                listener.close()
                threading.excepthook = orig_excepthook
                if hasattr(thread, "exception"):
                    raise thread.exception

        thread.join()

        # special case, assume a list of len 1 should not be a list
        if len(result) == 1:
            result = result[0]
        return result

    @staticmethod
    def __get_vertex_edge_id_obj(id_or_ids):
        # Force np.ndarray
        if not isinstance(id_or_ids, (int, Sequence, np.ndarray)):
            if cupy_installed and isinstance(id_or_ids, cp.ndarray):
                id_or_ids = id_or_ids.get()
            elif cudf_installed and isinstance(id_or_ids, cudf.Series):
                id_or_ids = id_or_ids.values_host
            elif pandas_installed and isinstance(id_or_ids, pandas.Series):
                id_or_ids = id_or_ids.to_numpy()
            else:
                raise ValueError(
                    f"No available module for processing {type(id_or_ids)}"
                )

        if isinstance(id_or_ids, Sequence):
            vert_edge_id_obj = GraphVertexEdgeID(int64_ids=id_or_ids)
        elif isinstance(id_or_ids, np.ndarray):
            if id_or_ids.dtype == "int32":
                vert_edge_id_obj = GraphVertexEdgeID(int32_ids=id_or_ids)
            elif id_or_ids.dtype == "int64":
                vert_edge_id_obj = GraphVertexEdgeID(int64_ids=id_or_ids)
        else:
            vert_edge_id_obj = GraphVertexEdgeID(int64_id=id_or_ids)
        return vert_edge_id_obj
