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

from cugraph_service_client import CugraphServiceClient

# Use the location of this file for finding various data files
this_dir = Path(__file__).parent

# Use the defaults for host and port (localhost, 9090)
# Assume the server is running and using the same defaults!
client = CugraphServiceClient()

# Remove any graphs from a previous session!
for gid in client.get_graph_ids():
    client.delete_graph(gid)

# Add vertex and edge data to the default graph instance (the default graph
# does not require a graph ID to access) The file names specified must be
# visible to the server.

client.load_csv_as_vertex_data(
    (this_dir / "vertex_data.csv").absolute().as_posix(),
    dtypes=["int32", "string", "int32"],
    vertex_col_name="vertex_id",
    header="infer",
)
client.load_csv_as_edge_data(
    (this_dir / "edge_data.csv").absolute().as_posix(),
    dtypes=["int32", "int32", "string", "int32"],
    vertex_col_names=("src", "dst"),
    header="infer",
)

# Verify the number of edges
assert client.get_num_edges() == 10000

# Run sampling and get a path, need to extract a subgraph first
extracted_gid = client.extract_subgraph(allow_multi_edges=True)
start_vertices = 11
max_depth = 2
(vertex_paths, edge_weights, path_sizes) = client.node2vec(
    start_vertices, max_depth, extracted_gid
)

# Create another graph on the server
graph2 = client.create_graph()

# Verify that both the default and new graph are present on the server
assert len(client.get_graph_ids()) == 3

# Add edge data to the new graph
client.load_csv_as_vertex_data(
    (this_dir / "vertex_data.csv").absolute().as_posix(),
    dtypes=["int32", "string", "int32"],
    vertex_col_name="vertex_id",
    header="infer",
    graph_id=graph2,
)

# Remove the new graph from the server and verify
client.delete_graph(graph2)
assert len(client.get_graph_ids()) == 2
