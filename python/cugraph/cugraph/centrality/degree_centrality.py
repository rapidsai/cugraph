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


from cugraph.utilities import (ensure_cugraph_obj_for_nx,
                               df_score_to_dictionary,
                               )
import cudf


def degree_centrality(G):
    """
    Computes the degrees of each vertex of the input graph.

    Parameters
    ----------
    G : cuGraph.Graph or networkx.Graph
    """
    G, isNx = ensure_cugraph_obj_for_nx(G)
    df = G.degree()
    df.rename("degree", "degree_centrality")
    return df
