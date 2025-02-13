# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

# If libcugraph was installed as a wheel, we must request it to load the library
# symbols. Otherwise, we assume that the library was installed in a system path that ld
# can find.
try:
    import libcugraph
except ModuleNotFoundError:
    pass
else:
    libcugraph.load_library()
    del libcugraph

from cugraph.community import (
    ecg,
    induced_subgraph,
    ktruss_subgraph,
    k_truss,
    louvain,
    leiden,
    spectralBalancedCutClustering,
    spectralModularityMaximizationClustering,
    analyzeClustering_modularity,
    analyzeClustering_edge_cut,
    analyzeClustering_ratio_cut,
    subgraph,
    triangle_count,
    ego_graph,
    batched_ego_graphs,
)

from cugraph.structure import (
    Graph,
    MultiGraph,
    BiPartiteGraph,
    from_edgelist,
    from_cudf_edgelist,
    from_pandas_edgelist,
    to_pandas_edgelist,
    from_pandas_adjacency,
    to_pandas_adjacency,
    from_numpy_array,
    to_numpy_array,
    from_numpy_matrix,
    to_numpy_matrix,
    from_adjlist,
    hypergraph,
    symmetrize,
    symmetrize_df,
    symmetrize_ddf,
    is_weighted,
    is_directed,
    is_multigraph,
    is_bipartite,
    is_multipartite,
)

from cugraph.centrality import (
    betweenness_centrality,
    edge_betweenness_centrality,
    katz_centrality,
    degree_centrality,
    eigenvector_centrality,
)

from cugraph.cores import core_number, k_core

from cugraph.components import (
    connected_components,
    weakly_connected_components,
    strongly_connected_components,
)

from cugraph.link_analysis import pagerank, hits

from cugraph.link_prediction import (
    jaccard,
    jaccard_coefficient,
    all_pairs_jaccard,
    overlap,
    overlap_coefficient,
    all_pairs_overlap,
    sorensen,
    sorensen_coefficient,
    all_pairs_sorensen,
    cosine,
    cosine_coefficient,
    all_pairs_cosine,
)

from cugraph.traversal import (
    bfs,
    bfs_edges,
    sssp,
    shortest_path,
    filter_unreachable,
    shortest_path_length,
    concurrent_bfs,
    multi_source_bfs,
)

from cugraph.tree import minimum_spanning_tree, maximum_spanning_tree

from cugraph.utilities import utils

from cugraph.linear_assignment import hungarian, dense_hungarian
from cugraph.layout import force_atlas2

from cugraph.sampling import (
    random_walks,
    uniform_random_walks,
    biased_random_walks,
    node2vec_random_walks,
    rw_path,
    node2vec,
    uniform_neighbor_sample,
)


from cugraph import experimental

from cugraph import gnn

from cugraph import exceptions

from cugraph._version import __git_commit__, __version__
