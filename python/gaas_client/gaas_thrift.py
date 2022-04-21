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

import io

import thriftpy2
from thriftpy2.rpc import make_server, make_client


gaas_thrift_spec = """
# FIXME: consider additional, more fine-grained exceptions
exception GaasError {
  1:string message
}

struct Node2vecResult {
  1:list<i32> vertex_paths
  2:list<double> edge_weights
  3:list<i32> path_sizes
}

service GaasService {

  i32 create_graph() throws(1:GaasError e),

  void delete_graph(1:i32 graph_id) throws (1:GaasError e),

  list<i32> get_graph_ids() throws(1:GaasError e),

  void load_csv_as_vertex_data(1:string csv_file_name,
                               2:string delimiter,
                               3:list<string> dtypes,
                               4:i32 header,
                               5:string vertex_col_name,
                               6:string type_name,
                               7:list<string> property_columns,
                               8:i32 graph_id
                               ) throws (1:GaasError e),

  void load_csv_as_edge_data(1:string csv_file_name,
                             2:string delimiter,
                             3:list<string> dtypes,
                             4:i32 header,
                             5:list<string> vertex_col_names,
                             6:string type_name,
                             7:list<string> property_columns,
                             8:i32 graph_id
                             ) throws (1:GaasError e),

  i32 get_num_edges(1:i32 graph_id) throws(1:GaasError e),

  Node2vecResult
  node2vec(1:list<i32> start_vertices,
           2:i32 max_depth,
           3:i32 graph_id
           ) throws (1:GaasError e),

  i32 extract_subgraph(1:string create_using,
                       2:string selection,
                       3:string edge_weight_property,
                       4:double default_edge_weight,
                       5:bool allow_multi_edges,
                       6:i32 graph_id
                       ) throws (1:GaasError e),
}
"""

spec = thriftpy2.load_fp(io.StringIO(gaas_thrift_spec),
                         module_name="gaas_thrift")

def create_server(handler, host, port):
    """
    """
    return make_server(spec.GaasService, handler, host, port)

def create_client(host, port):
    """
    """
    return make_client(spec.GaasService, host=host, port=port)
