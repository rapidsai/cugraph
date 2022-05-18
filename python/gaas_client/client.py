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

from . import defaults
from .gaas_thrift import create_client


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

    def open(self):
        """
        Opens a connection to the server at self.host/self.port if one is not
        already established. close() must be called in order to allow other
        connections from other clients to be made.

        Note: all APIs that access the server will call this method
        automatically, followed automatically by a call to close(), so calling
        this method should not be necessary. close() is not automatically called
        if self.hold_open is False.

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
        >>> # Manually open a connection. The connection is held open and other
        >>> # clients cannot connect until a client API call completes or
        >>> # close() is manually called.
        >>> client.open()
        """
        if self.__client is None:
            self.__client = create_client(self.host, self.port)

    def close(self):
        """Closes a connection to the server if one has been established, allowing
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
        """Calls a graph creation extension on the server that was previously
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
    def load_csv_as_vertex_data(self,
                                csv_file_name,
                                dtypes,
                                vertex_col_name,
                                delimiter=" ",
                                header=None,
                                type_name="",
                                property_columns=None,
                                graph_id=defaults.graph_id,
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
                                                     graph_id)

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
                                                   graph_id)

    @__server_connection
    def get_num_edges(self, graph_id=defaults.graph_id):
        """
        Returns the number of edges for the graph identified as graph_id (or the
        default graph if not specified).

        Parameters
        ----------
        graph_id : int, default is defaults.graph_id
            The graph ID to query. If the ID passed is not valid on the server,
            GaaSError is raised.

        Returns
        -------
        num_edges : int
            The number of edges in graph_id

        Examples
        --------
        >>> from gaas_client import GaasClient
        >>> client = GaasClient()
        >>> # This server already has graphs loaded from other sessions
        >>> client.get_num_edges()
        10000
        """
        return self.__client.get_num_edges(graph_id)

    @__server_connection
    def extract_subgraph(self,
                         create_using=None,
                         selection=None,
                         edge_weight_property="",
                         default_edge_weight=1.0,
                         allow_multi_edges=False,
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
        # FIXME: convert defaults to type needed by the Thrift API. These will
        # be changing to different types.
        create_using = create_using or ""
        selection = selection or ""

        return self.__client.extract_subgraph(create_using,
                                              selection,
                                              edge_weight_property,
                                              default_edge_weight,
                                              allow_multi_edges,
                                              graph_id)


    ############################################################################
    # Algos
    @__server_connection
    def node2vec(self, start_vertices, max_depth, graph_id=defaults.graph_id):
        """
        Computes random walks for each node in 'start_vertices', under the
        node2vec sampling framework.

        Parameters
        ----------
        G : cuGraph.Graph or networkx.Graph
            The graph can be either directed (DiGraph) or undirected (Graph).
            Weights in the graph are ignored.

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
        # start_vertices must be a list (cannot just be an iterable), and assume
        # return value is tuple of python lists on host.
        if not isinstance(start_vertices, list):
            start_vertices = [start_vertices]
        # FIXME: ensure list is a list of int32, since Thrift interface
        # specifies that?
        node2vec_result = self.__client.node2vec(start_vertices,
                                                 max_depth,
                                                 graph_id)
        # Hide the generated Thrift result type for node2vec and instead return
        # a tuple of lists)
        return (node2vec_result.vertex_paths,
                node2vec_result.edge_weights,
                node2vec_result.path_sizes)

    @__server_connection
    def pagerank(self, graph_id=defaults.graph_id):
        """
        pagerank
        """
        raise NotImplementedError
