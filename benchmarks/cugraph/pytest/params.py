# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

import pytest

from cugraph.testing import utils
from pylibcugraph.testing.utils import gen_fixture_params


# FIXME: omitting soc-twitter-2010.csv due to OOM error on some workstations.
undirected_datasets = [
    pytest.param(Path(utils.RAPIDS_DATASET_ROOT_DIR)/"karate.csv",
                 marks=[pytest.mark.tiny, pytest.mark.undirected]),
    pytest.param(Path(utils.RAPIDS_DATASET_ROOT_DIR)/"csv/undirected/hollywood.csv",
                 marks=[pytest.mark.small, pytest.mark.undirected]),
    pytest.param(Path(utils.RAPIDS_DATASET_ROOT_DIR)/"csv/undirected/europe_osm.csv",
                 marks=[pytest.mark.undirected]),
    # pytest.param("../datasets/csv/undirected/soc-twitter-2010.csv",
    #              marks=[pytest.mark.undirected]),
]

directed_datasets = [
    pytest.param(Path(utils.RAPIDS_DATASET_ROOT_DIR)/"csv/directed/cit-Patents.csv",
                 marks=[pytest.mark.small, pytest.mark.directed]),
    pytest.param(Path(
        utils.RAPIDS_DATASET_ROOT_DIR)/"csv/directed/soc-LiveJournal1.csv",
                 marks=[pytest.mark.directed]),
]

managed_memory = [
    pytest.param(True,
                 marks=[pytest.mark.managedmem_on]),
    pytest.param(False,
                 marks=[pytest.mark.managedmem_off]),
]

pool_allocator = [
    pytest.param(True,
                 marks=[pytest.mark.poolallocator_on]),
    pytest.param(False,
                 marks=[pytest.mark.poolallocator_off]),
]

sg = pytest.param(
    "SG",
    marks=[pytest.mark.sg],
    id="gpu_config=SG",
)
snmg = pytest.param(
    "SNMG",
    marks=[pytest.mark.snmg, pytest.mark.mg],
    id="gpu_config=SNMG",
)
mnmg = pytest.param(
    "MNMG",
    marks=[pytest.mark.mnmg, pytest.mark.mg],
    id="gpu_config=MNMG",
)
graph = pytest.param(
    "Graph",
    marks=[pytest.mark.local],
    id="type=Graph",
)
remotegraph = pytest.param(
    "RemoteGraph",
    marks=[pytest.mark.remote],
    id="type=RemoteGraph",
)
karate = pytest.param(
    "karate",
    id="dataset=karate",
)
small_low_degree_rmat = pytest.param(
    {"scale": 16, "edgefactor": 4},
    id="dataset=rmat_16_4",
)
small_high_degree_rmat = pytest.param(
    {"scale": 16, "edgefactor": 32},
    id="dataset=rmat_16_32",
)
large_low_degree_rmat = pytest.param(
    {"scale": 23, "edgefactor": 4},
    id="dataset=rmat_23_4",
)
large_high_degree_rmat = pytest.param(
    {"scale": 23, "edgefactor": 32},
    id="dataset=rmat_23_32",
)
huge_low_degree_rmat = pytest.param(
    {"scale": 30, "edgefactor": 4},
    id="dataset=rmat_30_4",
)
huge_high_degree_rmat = pytest.param(
    {"scale": 30, "edgefactor": 32},
    id="dataset=rmat_30_32",
)
large_start_list = pytest.param(
    "LARGE",
    marks=[pytest.mark.start_list_large],
    id="start_list_len=LARGE",
)
small_start_list = pytest.param(
    "SMALL",
    marks=[pytest.mark.start_list_small],
    id="start_list_len=SMALL",
)
large_fanout_list = pytest.param(
    "LARGE",
    marks=[pytest.mark.fanout_list_large],
    id="fanout_list_len=LARGE",
)
small_fanout_list = pytest.param(
    "SMALL",
    marks=[pytest.mark.fanout_list_small],
    id="fanout_list_len=SMALL",
)

# Define/generate the combinations to run.
graph_obj_fixture_params = gen_fixture_params(
    (graph, sg, karate),
    (graph, sg, small_low_degree_rmat),
    (graph, sg, small_high_degree_rmat),
    (graph, snmg, large_low_degree_rmat),
    (graph, snmg, large_high_degree_rmat),
)

remote_graph_obj_fixture_params = gen_fixture_params(
    (remotegraph, sg, karate),
    (remotegraph, sg, small_low_degree_rmat),
    (remotegraph, sg, small_high_degree_rmat),
    (remotegraph, snmg, large_low_degree_rmat),
    (remotegraph, snmg, large_high_degree_rmat),
    (remotegraph, mnmg, huge_low_degree_rmat),
    (remotegraph, mnmg, huge_high_degree_rmat),
)
