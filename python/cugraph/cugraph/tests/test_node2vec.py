# Copyright (c) 2022, NVIDIA CORPORATION.:
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

import gc
# import random

import pytest
# from cudf.testing import assert_series_equal

from cugraph.tests import utils
# import cugraph


# =============================================================================
# Parameters
# =============================================================================
DIRECTED_GRAPH_OPTIONS = [False, True]
WEIGHTED_GRAPH_OPTIONS = [False, True]
DATASETS = [pytest.param(d) for d in utils.DATASETS]
DATASETS_SMALL = [pytest.param(d) for d in utils.DATASETS_SMALL]


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


@pytest.mark.parametrize("graph_file", utils.DATASETS_SMALL)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_node2vec_coalesced():
    assert 1 == 2


@pytest.mark.parametrize("graph_file", utils.DATASETS_SMALL)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
def test_node2vec_padded():
    assert 1 == 2
