# Copyright (c) 2023, NVIDIA CORPORATION.
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

from cugraph.testing.utils import (
    RAPIDS_DATASET_ROOT_DIR_PATH,
    RAPIDS_DATASET_ROOT_DIR,
)
from cugraph.testing.resultset import (
    Resultset,
    load_resultset,
    get_resultset,
    results_dir_path,
)
from cugraph.datasets import (
    cyber,
    dining_prefs,
    dolphins,
    karate,
    karate_disjoint,
    polbooks,
    netscience,
    small_line,
    small_tree,
    email_Eu_core,
    toy_graph,
    toy_graph_undirected,
    soc_livejournal,
    cit_patents,
    europe_osm,
    hollywood,
    # twitter,
)

#
# Moved Dataset Batches
#

UNDIRECTED_DATASETS = [karate, dolphins]
SMALL_DATASETS = [karate, dolphins, polbooks]
WEIGHTED_DATASETS = [
    dining_prefs,
    dolphins,
    karate,
    karate_disjoint,
    netscience,
    polbooks,
    small_line,
    small_tree,
]
ALL_DATASETS = [
    dining_prefs,
    dolphins,
    karate,
    karate_disjoint,
    polbooks,
    netscience,
    small_line,
    small_tree,
    email_Eu_core,
    toy_graph,
    toy_graph_undirected,
]
DEFAULT_DATASETS = [dolphins, netscience, karate_disjoint]
BENCHMARKING_DATASETS = [soc_livejournal, cit_patents, europe_osm, hollywood]
