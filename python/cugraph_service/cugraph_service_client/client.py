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

from functools import wraps
from collections.abc import Sequence
import pickle

from gaas_client import defaults
from gaas_client.types import ValueWrapper, GraphVertexEdgeID
from gaas_client.gaas_thrift import create_client


class GaasClient:
    """
    Client object for GaaS, which defines the API that clients can use to access
    the GaaS server.
    """
    def __init__(self, host=defaults.host, port=defaults.port):
        """
        Creates a connection to a GaaS server running on host/port.

        Parameters
        ----------
        host : string, defaults to 127.0.0.1
            Hostname where the GaaS server is running

        port : int, defaults to 9090
            Port number where the GaaS server is listening

        Returns
        -------
        GaasClient object

        Examples
        --------
        >>> from gaas_client import GaasClient
        >>> client = GaasClient()
        """
        self.host = host
        self.port = port
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
        not take place, allowing for multiple subsequent server calls to be made
        using the same connection. self.hold_open therefore requires the caller
        to manually call close() in order to allow other clients to connect.
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
        this method should not be necessary. close() is not automatically called
        if self.hold_open is False.

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
        >>> from gaas_client import GaasClient
        >>> client = GaasClient()
        >>> # Manually open a connection. The connection is held open and other
        >>> # clients cannot connect until a client API call completes or
        >>> # close() is manually called.
        >>> client.open()

        """
        if self.__client is None:
            self.__client = create_client(self.host, self.port,
                                          call_timeout=call_timeout)

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
        >>> from gaas_client import GaasClient
        >>> client = GaasClient()
        >>> # Have the client hold open the connect automatically opened as part
        >>> # of a server API call until close() is called. This is normally not
        >>> # necessary and shown here for demonstration purposes.
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

    ############################################################################
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
        >>> from gaas_client import GaasClient
        >>> client = GaasClient()
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
        >>> from gaas_client import GaasClient
        >>> client = GaasClient()
        >>> client.get_server_info()
        >>> {'num_gpus': 2}
        """
        server_info = self.__client.get_server_info()
        # server_info is a dictionary of Value objects ("union" types returned
        # from the server), so convert them to simple py types.
        return dict((k, ValueWrapper(server_info[k]).get_py_obj())
                    for k in server_info)

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
        num_files_read : int
            Number of extension files read in the extension_dir_path directory.

        Examples
        --------
        >>> from gaas_client import GaasClient
        >>> client = GaasClient()
        >>> num_files_read = client.load_graph_creation_extensions(
        ... "/some/server/side/directory")
        >>>
        """
        return self.__client.load_graph_creation_extensions(extension_dir_path)

    @__server_connection
    def unload_graph_creation_extensions(self):
        """
        Removes all extensions for graph creation previously loaded.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Examples
        --------
        >>> from gaas_client import GaasClient
        >>> client = GaasClient()
        >>> client.unload_graph_creation_extensions()
        >>>
        """
        return self.__client.unload_graph_creation_extensions()

    @__server_connection
    def call_graph_creation_extension(self, func_name,
                                      *func_args, **func_kwargs):
        """
        Calls a graph creation extension on the server that was previously
        loaded by a prior call to load_graph_creation_extensions(), then returns
        the graph ID of the graph created by the extension.

        Parameters
        ----------
        func_name : string
            The name of the server-side extension function loaded by a prior
            call to load_graph_creation_extensions(). All graph creation
            extension functions are expected to return a new graph.

        *func_args : string, int, list, dictionary (optional)
            The positional args to pass to func_name. Note that func_args are
            converted to their string representation using repr() on the client,
            then restored to python objects on the server using eval(), and
            therefore only objects that can be restored server-side with eval()
            are supported.

        **func_kwargs : string, int, list, dictionary
            The keyword args to pass to func_name. Note that func_kwargs are
            converted to their string representation using repr() on the client,
            then restored to python objects on the server using eval(), and
            therefore only objects that can be restored server-side with eval()
            are supported.

        Returns
        -------
        graph_id : int
            unique graph ID

        Examples
        --------
        >>> from gaas_client import GaasClient
        >>> client = GaasClient()
        >>> # Load the extension file containing "my_complex_create_graph()"
        >>> client.load_graph_creation_extensions("/some/server/side/directory")
        >>> new_graph_id = client.call_graph_creation_extension(
        ... "my_complex_create_graph",
        ... "/path/to/csv/on/server/graph.csv",
        ... clean_data=True)
        >>>
        """
        func_args_repr = repr(func_args)
        func_kwargs_repr = repr(func_kwargs)
        return self.__client.call_graph_creation_extension(
            func_name, func_args_repr, func_kwargs_repr)

    ############################################################################
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
        >>> from gaas_client import GaasClient
        >>> client = GaasClient()
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
            The graph ID to delete. If the ID passed is not valid on the server,
            GaaSError is raised.

        Returns
        -------
        None

        Examples
        --------
        >>> from gaas_client import GaasClient
        >>> client = GaasClient()
        >>> my_graph_id = client.create_graph()
        >>> # Load a CSV to the new graph
        >>> client.load_csv_as_edge_data(
        ... "edges.csv", ["int32", "int32", "float32"],
        ... vertex_col_names=["src", "dst"], graph_id=my_graph_id)
        >>> # Remove the graph instance on the server and reclaim the memory
        >>> client.delete_graph(my_graph_id)
        """
        return self.__client.delete_graph(graph_id)

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
        >>> from gaas_client import GaasClient
        >>> client = GaasClient()
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
            The graph ID to apply the properties in the CSV to. If not provided,
            the default graph ID is used.

        Returns
        -------
        None

        Examples
        --------
        >>> from gaas_client import GaasClient
        >>> client = GaasClient()
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
            raise TypeError("keys must be a string or list of strings, got "
                            f"{type(keys)}")

        graph_info = self.__client.get_graph_info(keys, graph_id)

        # special case: if only one key was specified, return only the single
        # value
        if len(keys) == 1:
            return ValueWrapper(graph_info[keys[0]]).get_py_obj()

        # graph_info is a dictionary of Value objects ("union" types returned
        # from the graph), so convert them to simple py types.
        return dict((k, ValueWrapper(graph_info[k]).get_py_obj())
                    for k in graph_info)

    @__server_connection
    def load_csv_as_vertex_data(self,
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
            Row number to use as the column names. Default behavior is to assume
            column names are explicitely provided (header=None). header="infer"
            if the column names are to be inferred. If no names are passed,
            header=0. See also cudf.read_csv

        type_name : string, default is ""
            The vertex property "type" the CSV data is describing. For instance,
            CSV data describing properties for "users" might pass type_name as
            "user". A vertex property type is optional.

        property_columns : list of strings, default is None
            The column names in the CSV to add as vertex properties. If None,
            all columns will be added as properties.

        graph_id : int, default is defaults.graph_id
            The graph ID to apply the properties in the CSV to. If not provided,
            the default graph ID is used.

        names: list of strings, default is None
            The names to be used to reference the CSV columns, in lieu of a
            header.

        Returns
        -------
        None

        Examples
        --------
        >>> from gaas_client import GaasClient
        >>> client = GaasClient()
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
        return self.__client.load_csv_as_vertex_data(csv_file_name,
                                                     delimiter,
                                                     dtypes,
                                                     header,
                                                     vertex_col_name,
                                                     type_name,
                                                     property_columns or [],
                                                     graph_id,
                                                     names or [])

    @__server_connection
    def load_csv_as_edge_data(self,
                              csv_file_name,
                              dtypes,
                              vertex_col_names,
                              delimiter=" ",
                              header=None,
                              type_name="",
                              property_columns=None,
                              graph_id=defaults.graph_id,
                              names=None
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
            Names of the columns to use as the source and destination vertex IDs
            defining the edges

        delimiter : string, default is " "
            Character that serves as the delimiter between columns in the CSV

        header : int, default is None
            Row number to use as the column names. Default behavior is to assume
            column names are explicitely provided (header=None). header="infer"
            if the column names are to be inferred. If no names are passed,
            header=0. See also cudf.read_csv

        type_name : string, default is ""
            The edge property "type" the CSV data is describing. For instance,
            CSV data describing properties for "transactions" might pass
            type_name as "transaction". An edge property type is optional.

        property_columns : list of strings, default is None
            The column names in the CSV to add as edge properties. If None, all
            columns will be added as properties.

        graph_id : int, default is defaults.graph_id
            The graph ID to apply the properties in the CSV to. If not provided,
            the default graph ID is used.

        names: list of strings, default is None
            The names to be used to reference the CSV columns, in lieu of a
            header.

        Returns
        -------
        None

        Examples
        --------
        >>> from gaas_client import GaasClient
        >>> client = GaasClient()
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
        return self.__client.load_csv_as_edge_data(csv_file_name,
                                                   delimiter,
                                                   dtypes,
                                                   header,
                                                   vertex_col_names,
                                                   type_name,
                                                   property_columns or [],
                                                   graph_id,
                                                   names or [])

    @__server_connection
    def get_edge_IDs_for_vertices(self, src_vert_IDs, dst_vert_IDs,
                                  graph_id=defaults.graph_id):
        """
        """
        # FIXME: finish docstring above
        # FIXME: add type checking
        return self.__client.get_edge_IDs_for_vertices(src_vert_IDs,
                                                       dst_vert_IDs,
                                                       graph_id)

    @__server_connection
    def extract_subgraph(self,
                         create_using=None,
                         selection=None,
                         edge_weight_property="",
                         default_edge_weight=1.0,
                         allow_multi_edges=False,
                         renumber_graph=True,
                         add_edge_data=True,
                         graph_id=defaults.graph_id
                         ):
        """
        Return a graph ID for a subgraph of the graph referenced by graph_id
        that containing vertices and edges that match a selection.

        Parameters
        ----------
        create_using : string, default is None
            String describing the type of Graph object to create from the
            selected subgraph of vertices and edges. The default (None) results
            in a cugraph.Graph object.

        selection : int, default is None
            A PropertySelection ID returned from one or more calls to
            select_vertices() and/or select_edges(), used for creating a Graph
            with only the selected properties. If not speciied the resulting
            Graph will have all properties. Note, this could result in a Graph
            with multiple edges, which may not be supported based on the value
            of create_using.

        edge_weight_property : string, default is ""
            The name of the property whose values will be used as weights on the
            returned Graph. If not specified, the returned Graph will be
            unweighted.

        default_edge_weight : float, default is 1.0
            The value to use when an edge property is specified but not present
            on an edge.

        allow_multi_edges : bool
            If True, multiple edges should be used to create the resulting
            Graph, otherwise multiple edges will be detected and an exception
            raised.

        graph_id : int, default is defaults.graph_id
           The graph ID to extract the subgraph from. If the ID passed is not
           valid on the server, GaaSError is raised.

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

        return self.__client.extract_subgraph(create_using,
                                              selection,
                                              edge_weight_property,
                                              default_edge_weight,
                                              allow_multi_edges,
                                              renumber_graph,
                                              add_edge_data,
                                              graph_id)

    @__server_connection
    def get_graph_vertex_data(self,
                              id_or_ids=-1,
                              null_replacement_value=0,
                              graph_id=defaults.graph_id,
                              property_keys=None
                              ):
        """
        Returns ...

        Parameters
        ----------
        id_or_ids : int or list of ints (default -1)

        null_replacement_value : number or string (default 0)

        graph_id : int, default is defaults.graph_id
           The graph ID to extract the subgraph from. If the ID passed is not
           valid on the server, GaaSError is raised.

        property_keys : list of strings (default [])
            The keys (names) of properties to retrieve.  If omitted, returns
            the whole dataframe.

        Returns
        -------

        Examples
        --------
        >>>
        """
        # FIXME: finish docstring above

        vertex_edge_id_obj = self.__get_vertex_edge_id_obj(id_or_ids)
        null_replacement_value_obj = ValueWrapper(
            null_replacement_value,
            val_name="null_replacement_value").union

        ndarray_bytes = \
            self.__client.get_graph_vertex_data(
                vertex_edge_id_obj,
                null_replacement_value_obj,
                graph_id,
                property_keys or []
            )

        return pickle.loads(ndarray_bytes)


    @__server_connection
    def get_graph_edge_data(self,
                            id_or_ids=-1,
                            null_replacement_value=0,
                            graph_id=defaults.graph_id,
                            property_keys=None
                            ):
        """
        Returns ...

        Parameters
        ----------
        id_or_ids : int or list of ints (default -1)

        null_replacement_value : number or string (default 0)

        graph_id : int, default is defaults.graph_id
           The graph ID to extract the subgraph from. If the ID passed is not
           valid on the server, GaaSError is raised.

        property_keys : list of strings (default [])
            The keys (names) of properties to retrieve.  If omitted, returns
            the whole dataframe.

        Returns
        -------

        Examples
        --------
        >>>
        """
        # FIXME: finish docstring above

        vertex_edge_id_obj = self.__get_vertex_edge_id_obj(id_or_ids)
        null_replacement_value_obj = ValueWrapper(
            null_replacement_value,
            val_name="null_replacement_value").union

        ndarray_bytes = \
            self.__client.get_graph_edge_data(
                vertex_edge_id_obj,
                null_replacement_value_obj,
                graph_id,
                property_keys or []
            )

        return pickle.loads(ndarray_bytes)

    @__server_connection
    def is_vertex_property(self, property_key, graph_id=defaults.graph_id):
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

    ############################################################################
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
        batched_ego_graphs_result = self.__client.batched_ego_graphs(seeds,
                                                                     radius,
                                                                     graph_id)

        # FIXME: ensure dtypes are correct for values returned from cugraph.batched_ego_graphs() in gaas_handler.py
        #return (numpy.frombuffer(batched_ego_graphs_result.src_verts, dtype="int32"),
        #        numpy.frombuffer(batched_ego_graphs_result.dst_verts, dtype="int32"),
        #        numpy.frombuffer(batched_ego_graphs_result.edge_weights, dtype="float64"),
        #        numpy.frombuffer(batched_ego_graphs_result.seeds_offsets, dtype="int64"))
        return (batched_ego_graphs_result.src_verts,
                batched_ego_graphs_result.dst_verts,
                batched_ego_graphs_result.edge_weights,
                batched_ego_graphs_result.seeds_offsets)

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

        # start_vertices must be a list (cannot just be an iterable), and assume
        # return value is tuple of python lists on host.
        if not isinstance(start_vertices, list):
            start_vertices = [start_vertices]
        # FIXME: ensure list is a list of int32, since Thrift interface
        # specifies that?
        node2vec_result = self.__client.node2vec(start_vertices,
                                                 max_depth,
                                                 graph_id)
        return (node2vec_result.vertex_paths,
                node2vec_result.edge_weights,
                node2vec_result.path_sizes)

    @__server_connection
    def uniform_neighbor_sample(self,
                                start_list,
                                fanout_vals,
                                with_replacement=True,
                                graph_id=defaults.graph_id):
        """
        Samples the graph and returns the graph id of the sampled
        graph.

        Parameters:
        start_list: list[int]

        fanout_vals: list[int]

        with_replacement: bool

        graph_id: int, default is defaults.graph_id

        Returns
        -------
        The graph id of the sampled graph.

        """

        return self.__client.uniform_neighbor_sample(
            start_list,
            fanout_vals,
            with_replacement,
            graph_id,
        )

    @__server_connection
    def pagerank(self, graph_id=defaults.graph_id):
        """
        pagerank
        """
        raise NotImplementedError


    ############################################################################
    # Test/Debug
    @__server_connection
    def _get_graph_type(self, graph_id=defaults.graph_id):
        """
        Test/debug API for returning a string repr of the graph_id instance.
        """
        return self.__client.get_graph_type(graph_id)


    ############################################################################
    # Private
    @staticmethod
    def __get_vertex_edge_id_obj(id_or_ids):
        # FIXME: do not assume all values are int32
        if isinstance(id_or_ids, Sequence):
            vert_edge_id_obj = GraphVertexEdgeID(int32_ids=id_or_ids)
        else:
            vert_edge_id_obj = GraphVertexEdgeID(int32_id=id_or_ids)
        return vert_edge_id_obj
