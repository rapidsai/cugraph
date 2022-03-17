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


# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import pytest
from cugraph.tests import utils

MAX_ITERATIONS = [500]
TOLERANCE = [1.0e-06]
ALPHA = [0.85]
PERSONALIZATION_PERC = [0]
HAS_GUESS = [0]


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED_WEIGHTS)
def test_with_noparams(graph_file):

    import cugraph.compat.nx as nx

    M = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight",
        create_using=nx.DiGraph()
    )
    pr = nx.pagerank(Gnx)
    print(type(Gnx))
    assert type(pr) == dict

    M = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight",
        create_using=nx.DiGraph()
    )
    pr = nx.pagerank(Gnx)
    print(type(Gnx))
    assert type(pr) == dict
