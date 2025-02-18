# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

from dask import config

from .link_analysis.pagerank import pagerank
from .link_analysis.hits import hits
from .traversal.bfs import bfs
from .traversal.sssp import sssp
from .common.read_utils import get_chunksize
from .common.read_utils import get_n_workers
from .community.louvain import louvain
from .community.triangle_count import triangle_count
from .community.egonet import ego_graph
from .community.induced_subgraph import induced_subgraph
from .community.ktruss_subgraph import ktruss_subgraph
from .centrality.katz_centrality import katz_centrality
from .components.connectivity import weakly_connected_components
from .sampling.uniform_neighbor_sample import uniform_neighbor_sample
from .sampling.random_walks import random_walks
from .sampling.uniform_random_walks import uniform_random_walks
from .sampling.biased_random_walks import biased_random_walks
from .sampling.node2vec_random_walks import node2vec_random_walks
from .centrality.eigenvector_centrality import eigenvector_centrality
from .cores.core_number import core_number
from .centrality.betweenness_centrality import betweenness_centrality
from .centrality.betweenness_centrality import edge_betweenness_centrality
from .cores.k_core import k_core
from .link_prediction.jaccard import jaccard
from .link_prediction.jaccard import all_pairs_jaccard
from .link_prediction.sorensen import sorensen
from .link_prediction.sorensen import all_pairs_sorensen
from .link_prediction.overlap import overlap
from .link_prediction.overlap import all_pairs_overlap
from .link_prediction.cosine import cosine
from .link_prediction.cosine import all_pairs_cosine
from .community.leiden import leiden
from .community.ecg import ecg

# Avoid "p2p" shuffling in dask for now
config.set({"dataframe.shuffle.method": "tasks"})
