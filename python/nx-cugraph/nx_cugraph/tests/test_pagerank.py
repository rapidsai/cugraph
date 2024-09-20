# Copyright (c) 2024, NVIDIA CORPORATION.
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
import networkx as nx
import pandas as pd
from pytest import approx


def test_pagerank_multigraph():
    """
    Ensures correct differences between pagerank results for Graphs
    vs. MultiGraphs generated using from_pandas_edgelist()
    """
    df = pd.DataFrame({"source": [0, 1, 1, 1, 1, 1, 1, 2],
                       "target": [1, 2, 2, 2, 2, 2, 2, 3]})
    expected_pr_for_G = nx.pagerank(nx.from_pandas_edgelist(df))
    expected_pr_for_MultiG = nx.pagerank(
        nx.from_pandas_edgelist(df, create_using=nx.MultiGraph))

    G = nx.from_pandas_edgelist(df, backend="cugraph")
    actual_pr_for_G = nx.pagerank(G, backend="cugraph")

    MultiG = nx.from_pandas_edgelist(df, create_using=nx.MultiGraph, backend="cugraph")
    actual_pr_for_MultiG = nx.pagerank(MultiG, backend="cugraph")

    assert actual_pr_for_G == approx(expected_pr_for_G)
    assert actual_pr_for_MultiG == approx(expected_pr_for_MultiG)
