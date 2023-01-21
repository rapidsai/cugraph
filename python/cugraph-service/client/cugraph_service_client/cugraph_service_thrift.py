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

import io

import thriftpy2
from thriftpy2.rpc import make_client
from thriftpy2.protocol import TBinaryProtocolFactory
from thriftpy2.server import TSimpleServer
from thriftpy2.thrift import TProcessor
from thriftpy2.transport import (
    TBufferedTransportFactory,
    TServerSocket,
    TTransportException,
)


# This is the Thrift input file as a string rather than a separate file. This
# allows the Thrift input to be contained within the module that's responsible
# for all Thrift-specific details rather than a separate .thrift file.
#
# thriftpy2 (https://github.com/Thriftpy/thriftpy2) is being used here instead
# of Apache Thrift since it offers an easier-to-use API exclusively for Python
# which is still compatible with servers/cleints using Apache Thrift (Apache
# Thrift can be used from a variety of different languages) while offering
# approximately the same performance.
#
# See the Apache Thrift tutorial for Python for examples:
# https://thrift.apache.org/tutorial/py.html
cugraph_thrift_spec = """
# FIXME: consider additional, more fine-grained exceptions
exception CugraphServiceError {
  1:string message
}

struct BatchedEgoGraphsResult {
  1:list<i32> src_verts
  2:list<i32> dst_verts
  3:list<double> edge_weights
  4:list<i32> seeds_offsets
}

struct Node2vecResult {
  1:list<i32> vertex_paths
  2:list<double> edge_weights
  3:list<i32> path_sizes
}

# FIXME: uniform_neighbor_sample may need to return indices as ints
# See: https://github.com/rapidsai/cugraph/issues/2654
struct UniformNeighborSampleResult {
  1:list<i32> sources
  2:list<i32> destinations
  3:list<double> indices
}

union GraphVertexEdgeID {
  1:i32 int32_id
  2:i64 int64_id
  3:list<i32> int32_ids
  4:list<i64> int64_ids
}

union Value {
  1:i32 int32_value
  2:i64 int64_value
  3:string string_value
  4:bool bool_value
  5:double double_value
  6:list<Value> list_value
}

union Offsets {
  1:list<string> type
  2:list<i32> start
  3:list<i32> stop
}

service CugraphService {

  ##############################################################################
  # Environment management
  i32 uptime()

  map<string, Value> get_server_info() throws (1:CugraphServiceError e),

  list<string> load_graph_creation_extensions(1:string extension_dir_path
                                              ) throws (1:CugraphServiceError e),

  list<string> load_extensions(1:string extension_dir_path
                              ) throws (1:CugraphServiceError e),

  void unload_extension_module(1:string modname) throws (1:CugraphServiceError e),

  i32 call_graph_creation_extension(1:string func_name,
                                    2:string func_args_repr,
                                    3:string func_kwargs_repr
                                    ) throws (1:CugraphServiceError e),

  Value call_extension(1:string func_name,
                       2:string func_args_repr,
                       3:string func_kwargs_repr
                       4:string result_host,
                       5:i16 result_port
                       ) throws (1:CugraphServiceError e),

  ##############################################################################
  # Graph management
  i32 create_graph() throws(1:CugraphServiceError e),

  void delete_graph(1:i32 graph_id) throws (1:CugraphServiceError e),

  list<i32> get_graph_ids() throws(1:CugraphServiceError e),

  map<string, Value> get_graph_info(1:list<string> keys,
                                    2:i32 graph_id
                                    ) throws(1:CugraphServiceError e),

  void load_csv_as_vertex_data(1:string csv_file_name,
                               2:string delimiter,
                               3:list<string> dtypes,
                               4:i32 header,
                               5:string vertex_col_name,
                               6:string type_name,
                               7:list<string> property_columns,
                               8:i32 graph_id,
                               9:list<string> names
                               ) throws (1:CugraphServiceError e),

  void load_csv_as_edge_data(1:string csv_file_name,
                             2:string delimiter,
                             3:list<string> dtypes,
                             4:i32 header,
                             5:list<string> vertex_col_names,
                             6:string type_name,
                             7:list<string> property_columns,
                             8:i32 graph_id,
                             9:list<string> names,
                             10:string edge_id_col_name
                             ) throws (1:CugraphServiceError e),

  list<i32> get_edge_IDs_for_vertices(1:list<i32> src_vert_IDs,
                                      2:list<i32> dst_vert_IDs,
                                      3:i32 graph_id
                             ) throws (1:CugraphServiceError e),

  Offsets
  renumber_vertices_by_type(1:string prev_id_column,
                            2:i32 graph_id
                            ) throws (1:CugraphServiceError e),

  Offsets
  renumber_edges_by_type(1:string prev_id_column,
                         2:i32 graph_id
                         ) throws (1:CugraphServiceError e),

  i32 extract_subgraph(1:string create_using,
                       2:string selection,
                       3:string edge_weight_property,
                       4:double default_edge_weight,
                       5:bool check_multi_edges,
                       6:bool renumber_graph,
                       7:bool add_edge_data,
                       8:i32 graph_id
                       ) throws (1:CugraphServiceError e),

  binary get_graph_vertex_data(1:GraphVertexEdgeID vertex_id,
                               2:Value null_replacement_value,
                               3:list<string> property_keys,
                               4:list<string> types,
                               5:i32 graph_id
                               ) throws (1:CugraphServiceError e),

  binary get_graph_edge_data(1:GraphVertexEdgeID edge_id,
                             2:Value null_replacement_value,
                             3:list<string> property_keys,
                             4:list<string> types,
                             5:i32 graph_id,
                             ) throws (1:CugraphServiceError e),

  bool is_vertex_property(1:string property_key,
                          2:i32 graph_id) throws (1:CugraphServiceError e),

  bool is_edge_property(1:string property_key,
                        2:i32 graph_id) throws (1:CugraphServiceError e),

  list<string> get_graph_vertex_property_names(1:i32 graph_id)
               throws (1:CugraphServiceError e),

  list<string> get_graph_edge_property_names(1:i32 graph_id)
               throws (1:CugraphServiceError e),

  list<string> get_graph_vertex_types(1:i32 graph_id)
               throws (1:CugraphServiceError e),

  list<string> get_graph_edge_types(1:i32 graph_id)
               throws (1:CugraphServiceError e),

  i64 get_num_vertices(1:string vertex_type,
                       2:bool include_edge_data,
                       3:i32 graph_id) throws (1:CugraphServiceError e),

  i64 get_num_edges(1:string edge_type,
                    2:i32 graph_id) throws (1:CugraphServiceError e),
  ##############################################################################
  # Algos
  BatchedEgoGraphsResult
  batched_ego_graphs(1:list<i32> seeds,
                     2:i32 radius,
                     3:i32 graph_id
                     ) throws (1:CugraphServiceError e),

  Node2vecResult
  node2vec(1:list<i32> start_vertices,
           2:i32 max_depth,
           3:i32 graph_id
           ) throws (1:CugraphServiceError e),

  UniformNeighborSampleResult
  uniform_neighbor_sample(1:list<i32> start_list,
                          2:list<i32> fanout_vals,
                          3:bool with_replacement,
                          4:i32 graph_id,
                          5:string result_host,
                          6:i16 result_port
                          ) throws (1:CugraphServiceError e),

  ##############################################################################
  # Test/Debug
  i32 create_test_array(1:i64 nbytes
                        ) throws (1:CugraphServiceError e),

  void delete_test_array(1:i32 test_array_id) throws (1:CugraphServiceError e),

  list<byte> receive_test_array(1:i32 test_array_id
                                ) throws (1:CugraphServiceError e),

  oneway void receive_test_array_to_device(1:i32 test_array_id,
                                           2:string result_host,
                                           3:i16 result_port
                                           ) throws (1:CugraphServiceError e),

  string get_graph_type(1:i32 graph_id) throws(1:CugraphServiceError e),
}
"""

# Load the cugraph Thrift specification on import. Syntax errors and other
# problems will be apparent immediately on import, and it allows any other
# module to import this and access the various types defined in the Thrift
# specification without being exposed to the thriftpy2 API.
spec = thriftpy2.load_fp(io.StringIO(cugraph_thrift_spec), module_name="cugraph_thrift")


def create_server(handler, host, port, client_timeout=90000):
    """
    Return a server object configured to listen on host/port and use the
    handler object to handle calls from clients. The handler object must have
    an interface compatible with the CugraphService service defined in the
    Thrift specification.

    Note: This function is defined here in order to allow it to have easy
    access to the Thrift spec loaded here on import, and to keep all thriftpy2
    calls in this module. However, this function is likely only called from the
    cugraph_service_server package which depends on the code in this package.
    """
    proto_factory = TBinaryProtocolFactory()
    trans_factory = TBufferedTransportFactory()
    client_timeout = client_timeout

    processor = TProcessor(spec.CugraphService, handler)
    server_socket = TServerSocket(host=host, port=port, client_timeout=client_timeout)
    server = TSimpleServer(
        processor,
        server_socket,
        iprot_factory=proto_factory,
        itrans_factory=trans_factory,
    )
    return server


def create_client(host, port, call_timeout=90000):
    """
    Return a client object that will make calls on a server listening on
    host/port.

    The call_timeout value defaults to 90 seconds, and is used for setting the
    timeout for server API calls when using the client created here - if a call
    does not return in call_timeout milliseconds, an exception is raised.
    """
    try:
        return make_client(
            spec.CugraphService, host=host, port=port, timeout=call_timeout
        )
    except TTransportException:
        # Raise a CugraphServiceError in order to completely encapsulate all
        # Thrift details in this module. If this was not done, callers of this
        # function would have to import thriftpy2 in order to catch the
        # TTransportException, which then leaks thriftpy2.
        #
        # NOTE: normally the CugraphServiceError exception is imported from the
        # cugraph_service_client.exceptions module, but since
        # cugraph_service_client.exceptions.CugraphServiceError is actually
        # defined from the spec in this module, just use it directly from spec.
        #
        # FIXME: may need to have additional thrift exception handlers
        # FIXME: this exception being raised could use more detail
        raise spec.CugraphServiceError(
            "could not create a client session with a cugraph_service server"
        )
