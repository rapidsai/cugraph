# Copyright (c) 2022, NVIDIA CORPORATION.
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

import time

from cugraph_funcs import generate_edgelist

# Call RMAT
edgelist_df = generate_edgelist(scale=23, edgefactor=16)
srcs = edgelist_df["src"]
dsts = edgelist_df["dst"]
weights = edgelist_df["weight"]
weights = weights.astype("float32")

print(f"num edges: {len(weights)}")
print()

########
import cugraph

st = time.time()
G2 = cugraph.Graph(directed=True)
print(f"cugraph Graph create time:      {time.time()-st}")
G2.from_cudf_edgelist(edgelist_df, source="src", destination="dst",
                      edge_attr="weight", renumber=True)
st = time.time()
result = cugraph.pagerank(G2, alpha=0.85, tol=1.0e-6, max_iter=500)
print(f"cugraph time:      {time.time()-st}")

########
import pylibcugraph

resource_handle = pylibcugraph.experimental.ResourceHandle()
graph_props = pylibcugraph.experimental.GraphProperties(
    is_symmetric=False, is_multigraph=False)
st = time.time()
G = pylibcugraph.experimental.SGGraph(
    resource_handle, graph_props, srcs, dsts, weights,
    store_transposed=True, renumber=True, do_expensive_check=False)
print(f"pylibcugraph Graph create time: {time.time()-st}")
st = time.time()
(vertices, pageranks) = pylibcugraph.experimental.pagerank(
    resource_handle, G, None, alpha=0.85, epsilon=1.0e-6, max_iterations=500,
    has_initial_guess=False, do_expensive_check=True)
print(f"pylibcugraph time: {time.time()-st} (expensive check)")
st = time.time()
(vertices, pageranks) = pylibcugraph.experimental.pagerank(
    resource_handle, G, None, alpha=0.85, epsilon=1.0e-6, max_iterations=500,
    has_initial_guess=False, do_expensive_check=False)
print(f"pylibcugraph time: {time.time()-st}")

########
print()
vert_to_check = 4800348
p = result['pagerank'][result['vertex'] == vert_to_check]
print(f"cugraph pagerank for vert:      {vert_to_check}: {p.iloc[0]}")

host_verts = vertices.tolist()
index = host_verts.index(vert_to_check)
print(f"pylibcugraph pagerank for vert: {vert_to_check}: {pageranks[index]}")

vert_to_check = 268434647
p = result['pagerank'][result['vertex'] == vert_to_check]
print(f"cugraph pagerank for vert:      {vert_to_check}: {p.iloc[0]}")

index = host_verts.index(vert_to_check)
print(f"pylibcugraph pagerank for vert: {vert_to_check}: {pageranks[index]}")
