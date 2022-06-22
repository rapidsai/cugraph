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

import gc

import pytest
import cugraph
import cudf
# from cugraph.testing import utils

from cugraph.experimental.datasets import (karate,
                                           SMALL_DATASETS)


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# import warnings
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=DeprecationWarning)
#     import networkx as nx


@pytest.mark.parametrize("dataset", SMALL_DATASETS)
def test_getters(dataset):
    # Getting the graph does not need to depend on get_edgelist
    M = dataset.get_edgelist(fetch=True)

    G = dataset.get_graph(fetch=True)

    assert M is not None
    assert G is not None


# Test the number of nodes and edges
def test_karate_nodes():
    graph_file = 'datasets/karate-data.csv'
    G_a = karate.get_graph(fetch=True)

    df = cudf.read_csv(
            graph_file,
            delimiter="\t",
            dtype=["int32", "int32"],
            header=None,
        )
    G_b = cugraph.Graph(directed=True)
    G_b.from_cudf_edgelist(df, source="src",
                           destination="dst", renumber=False)

    assert G_a.number_of_nodes() == G_b.number_of_nodes()
