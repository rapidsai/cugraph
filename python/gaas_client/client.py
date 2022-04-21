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
    """
    def __init__(self, host=defaults.host, port=defaults.port):
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
        connections from other clients to be made. All APIs that access the
        server will call this method automatically, followed by a call to
        close(). close() is not automatically called if self.hold_open is False.
        """
        if self.__client is None:
            self.__client = create_client(self.host, self.port)

    def close(self):
        """
        Closes a connection to the server if one has been established, allowing
        other clients to access the server. This method is called automatically
        for all APIs that access the server if self.hold_open is False.
        """
        if self.__client is not None:
            self.__client.close()
            self.__client = None

    ############################################################################
    # Graph management
    @__server_connection
    def create_graph(self):
        """
        Create a new graph associated with a new (non-default) unique graph ID,
        return the new graph ID.
        """
        return self.__client.create_graph()

    @__server_connection
    def delete_graph(self, graph_id):
        """
        Deletes the graph referenced by graph_id.
        """
        return self.__client.delete_graph(graph_id)

    @__server_connection
    def get_graph_ids(self):
        """
        Returns a list of all graph IDs the server is currently maintaining.
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
        """
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
        """
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
        Extract a subgraph, return a new graph ID.
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
        """
        raise NotImplementedError
