# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
import time
import pytest
import numpy as np

import cugraph
from cugraph.experimental.datasets import DATASETS

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx


print("Networkx version : {} ".format(nx.__version__))

SOURCES = [1]


@pytest.mark.parametrize("graph_file", DATASETS)
@pytest.mark.parametrize("source", SOURCES)
def test_filter_unreachable(graph_file, source):
    G = graph_file.get_graph(create_using=cugraph.Graph(directed=True))
    cu_M = G.view_edge_list()

    print("sources size = " + str(len(cu_M)))
    print("destinations size = " + str(len(cu_M)))

    print("cugraph Solving... ")
    t1 = time.time()

    df = cugraph.sssp(G, source)

    t2 = time.time() - t1
    print("Time : " + str(t2))

    reachable_df = cugraph.filter_unreachable(df)

    if np.issubdtype(df["distance"].dtype, np.integer):
        inf = np.iinfo(reachable_df["distance"].dtype).max
        assert len(reachable_df.query("distance == @inf")) == 0
    elif np.issubdtype(df["distance"].dtype, np.inexact):
        inf = np.finfo(reachable_df["distance"].dtype).max  # noqa: F841
        assert len(reachable_df.query("distance == @inf")) == 0

    assert len(reachable_df) != 0
