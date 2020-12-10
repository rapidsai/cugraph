# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

from cugraph.community import (
    ecg,
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
    triangles,
)

from cugraph.structure import (
    Graph,
    DiGraph,
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
)

from cugraph.centrality import (
    betweenness_centrality,
    edge_betweenness_centrality,
    katz_centrality,
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
    overlap,
    overlap_coefficient,
    jaccard_w,
    overlap_w,
)

from cugraph.traversal import (
    bfs,
    bfs_edges,
    sssp,
    shortest_path,
    filter_unreachable,
    shortest_path_length
)

from cugraph.tree import minimum_spanning_tree, maximum_spanning_tree

from cugraph.utilities import utils

from cugraph.bsp.traversal import bfs_df_pregel

from cugraph.proto.components import strong_connected_component
from cugraph.proto.structure import find_bicliques

from cugraph.linear_assignment import hungarian
from cugraph.layout import force_atlas2
from cugraph.raft import raft_include_test
from cugraph.comms import comms

# Versioneer
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
