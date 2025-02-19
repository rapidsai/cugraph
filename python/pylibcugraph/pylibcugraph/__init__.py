# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

from pylibcugraph.components._connectivity import (
    strongly_connected_components,
)

from pylibcugraph.graphs import SGGraph, MGGraph

from pylibcugraph.resource_handle import ResourceHandle

from pylibcugraph.graph_properties import GraphProperties

from pylibcugraph.edge_id_lookup_table import EdgeIdLookupTable

from pylibcugraph.eigenvector_centrality import eigenvector_centrality

from pylibcugraph.katz_centrality import katz_centrality

from pylibcugraph.pagerank import pagerank

from pylibcugraph.personalized_pagerank import personalized_pagerank

from pylibcugraph.sssp import sssp

from pylibcugraph.hits import hits

from pylibcugraph.node2vec import node2vec

from pylibcugraph.node2vec_random_walks import node2vec_random_walks

from pylibcugraph.bfs import bfs

from pylibcugraph.uniform_neighbor_sample import uniform_neighbor_sample

from pylibcugraph.biased_neighbor_sample import biased_neighbor_sample

from pylibcugraph.homogeneous_uniform_neighbor_sample import (
    homogeneous_uniform_neighbor_sample,
)
from pylibcugraph.homogeneous_biased_neighbor_sample import (
    homogeneous_biased_neighbor_sample,
)
from pylibcugraph.heterogeneous_uniform_neighbor_sample import (
    heterogeneous_uniform_neighbor_sample,
)
from pylibcugraph.heterogeneous_biased_neighbor_sample import (
    heterogeneous_biased_neighbor_sample,
)

from pylibcugraph.negative_sampling import negative_sampling

from pylibcugraph.core_number import core_number

from pylibcugraph.k_core import k_core

from pylibcugraph.two_hop_neighbors import get_two_hop_neighbors

from pylibcugraph.louvain import louvain

from pylibcugraph.triangle_count import triangle_count

from pylibcugraph.egonet import ego_graph

from pylibcugraph.weakly_connected_components import weakly_connected_components

from pylibcugraph.uniform_random_walks import uniform_random_walks

from pylibcugraph.biased_random_walks import biased_random_walks

from pylibcugraph.betweenness_centrality import betweenness_centrality

from pylibcugraph.induced_subgraph import induced_subgraph

from pylibcugraph.ecg import ecg

from pylibcugraph.balanced_cut_clustering import balanced_cut_clustering

from pylibcugraph.spectral_modularity_maximization import (
    spectral_modularity_maximization,
)

from pylibcugraph.analyze_clustering_modularity import analyze_clustering_modularity

from pylibcugraph.analyze_clustering_edge_cut import analyze_clustering_edge_cut

from pylibcugraph.analyze_clustering_ratio_cut import analyze_clustering_ratio_cut

from pylibcugraph.random import CuGraphRandomState

from pylibcugraph.leiden import leiden

from pylibcugraph.select_random_vertices import select_random_vertices

from pylibcugraph.edge_betweenness_centrality import edge_betweenness_centrality

from pylibcugraph.generate_rmat_edgelist import generate_rmat_edgelist

from pylibcugraph.generate_rmat_edgelists import generate_rmat_edgelists

from pylibcugraph.replicate_edgelist import replicate_edgelist

from pylibcugraph.k_truss_subgraph import k_truss_subgraph

from pylibcugraph.jaccard_coefficients import jaccard_coefficients

from pylibcugraph.overlap_coefficients import overlap_coefficients

from pylibcugraph.sorensen_coefficients import sorensen_coefficients

from pylibcugraph.cosine_coefficients import cosine_coefficients

from pylibcugraph.all_pairs_jaccard_coefficients import all_pairs_jaccard_coefficients

from pylibcugraph.all_pairs_overlap_coefficients import all_pairs_overlap_coefficients

from pylibcugraph.all_pairs_sorensen_coefficients import all_pairs_sorensen_coefficients

from pylibcugraph.all_pairs_cosine_coefficients import all_pairs_cosine_coefficients

from pylibcugraph.degrees import in_degrees, out_degrees, degrees

from pylibcugraph.decompress_to_edgelist import decompress_to_edgelist

from pylibcugraph.renumber_arbitrary_edgelist import renumber_arbitrary_edgelist


from pylibcugraph import exceptions

from pylibcugraph._version import __git_commit__, __version__
