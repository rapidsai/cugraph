# Copyright (c) 2020, NVIDIA CORPORATION.
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

import pytest

from cugraph.tests.utils import genFixtureParamsProduct


# FIXME: write and use mechanism described here for specifying datasets:
#        https://docs.rapids.ai/maintainers/datasets
# FIXME: rlr: soc-twitter-2010.csv crashes with OOM error on my RTX-8000
UNDIRECTED_DATASETS = [
    pytest.param("../datasets/karate.csv",
                 marks=[pytest.mark.tiny, pytest.mark.undirected]),
    pytest.param("../datasets/csv/undirected/hollywood.csv",
                 marks=[pytest.mark.small, pytest.mark.undirected]),
    pytest.param("../datasets/csv/undirected/europe_osm.csv",
                 marks=[pytest.mark.undirected]),
    # pytest.param("../datasets/csv/undirected/soc-twitter-2010.csv",
    #              marks=[pytest.mark.undirected]),
]
DIRECTED_DATASETS = [
    pytest.param("../datasets/csv/directed/cit-Patents.csv",
                 marks=[pytest.mark.small, pytest.mark.directed]),
    pytest.param("../datasets/csv/directed/soc-LiveJournal1.csv",
                 marks=[pytest.mark.directed]),
]

MANAGED_MEMORY = [
    pytest.param(True,
                 marks=[pytest.mark.managedmem_on]),
    pytest.param(False,
                 marks=[pytest.mark.managedmem_off]),
]

POOL_ALLOCATOR = [
    pytest.param(True,
                 marks=[pytest.mark.poolallocator_on]),
    pytest.param(False,
                 marks=[pytest.mark.poolallocator_off]),
]

FIXTURE_PARAMS = genFixtureParamsProduct(
    (DIRECTED_DATASETS + UNDIRECTED_DATASETS, "ds"),
    (MANAGED_MEMORY, "mm"),
    (POOL_ALLOCATOR, "pa"))
