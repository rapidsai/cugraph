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

import gc

import pytest

import cugraph
import cudf
from cugraph.testing import utils
from cugraph.experimental.datasets import karate
import numpy as np


def test_bfs_paths():
    with pytest.raises(ValueError) as ErrorMsg:
        gc.collect()
        G = karate.get_graph()

        # run BFS starting at vertex 17
        df = cugraph.bfs(G, 16)

        # Get the path to vertex 1
        p_df = cugraph.utils.get_traversed_path(df, 0)

        assert len(p_df) == 3

        # Get path to vertex 0 - which is not in graph
        p_df = cugraph.utils.get_traversed_path(df, 100)

        assert "not in the result set" in str(ErrorMsg)


def test_bfs_paths_array():
    with pytest.raises(ValueError) as ErrorMsg:
        gc.collect()
        G = karate.get_graph()

        # run BFS starting at vertex 17
        df = cugraph.bfs(G, 16)

        # Get the path to vertex 1
        answer = cugraph.utils.get_traversed_path_list(df, 0)

        assert len(answer) == 3

        # Get path to vertex 0 - which is not in graph
        answer = cugraph.utils.get_traversed_path_list(df, 100)

        assert "not in the result set" in str(ErrorMsg)


@pytest.mark.parametrize("graph_file", utils.DATASETS)
@pytest.mark.skip(reason="Skipping large tests")
def test_get_traversed_cost(graph_file):
    cu_M = utils.read_csv_file(graph_file)

    noise = cudf.Series(np.random.randint(10, size=(cu_M.shape[0])))
    cu_M["info"] = cu_M["2"] + noise

    G = cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="info")

    # run SSSP starting at vertex 17
    df = cugraph.sssp(G, 16)

    answer = cugraph.utilities.path_retrieval.get_traversed_cost(
        df, 16, cu_M["0"], cu_M["1"], cu_M["info"]
    )

    df = df.sort_values(by="vertex").reset_index()
    answer = answer.sort_values(by="vertex").reset_index()

    assert df.shape[0] == answer.shape[0]
    assert np.allclose(df["distance"], answer["info"])
