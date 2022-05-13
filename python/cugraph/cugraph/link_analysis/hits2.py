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
#

from cugraph.utilities import (ensure_cugraph_obj_for_nx,
                               df_score_to_dictionary,
                               )
from pylibcugraph import (ResourceHandle,
                          GraphProperties,
                          SGGraph,
                          hits as pylibcugraph_hits
                          )
import cudf

def hits2(G, max_iter=100, tol=1.0e-5, nstart=None, normalized=True):
    # Input validation

    # Setup G and other inputs

    G, isNx = ensure_cugraph_obj_for_nx(G)

    srcs = G.edgelist.edgelist_df['src']
    dsts = G.edgelist.edgelist_df['dst']
    # edge weights are not used for this algorithm
    weights = G.edgelist.edgelist_df['src'] * 0.0
    #breakpoint()

    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_multigraph=G.is_multigraph())
    store_transposed = False
    renumber = False
    do_expensive_check = False
    initial_hubs_guess_vertices = None
    initial_hubs_guess_values = None

    if nstart is not None:
        initial_hubs_guess_vertices = nstart['vertex']
        initial_hubs_guess_values = nstart['values']

    sg = SGGraph(resource_handle, graph_props, srcs, dsts, weights,
                 store_transposed, renumber, do_expensive_check)
    
    vertices, hubs, authorities = pylibcugraph_hits(resource_handle, sg, tol,
                                                    max_iter,
                                                    initial_hubs_guess_vertices,
                                                    initial_hubs_guess_values,
                                                    normalized,
                                                    do_expensive_check)
    results = cudf.DataFrame()
    results["vertex"] = cudf.Series(vertices)
    results["hubs"] = cudf.Series(hubs)
    results["authorities"] = cudf.Series(authorities)

    if isNx is True:
        d1 = df_score_to_dictionary(results[["vertex", "hubs"]], "hubs")
        d2 = df_score_to_dictionary(results[["vertex", "authorities"]], "authorities")
        results = (d1, d2) 
    
    if G.renumbered:
        results = G.unrenumber(results, "vertex")

    return results
