# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
from cugraph.tests import utils
from pathlib import PurePath


def test_bfs_paths():
    with pytest.raises(ValueError) as ErrorMsg:
        gc.collect()

        graph_file = PurePath(utils.RAPIDS_DATASET_ROOT_DIR)/"karate.csv"

        cu_M = utils.read_csv_file(graph_file)

        G = cugraph.Graph()
        G.from_cudf_edgelist(cu_M, source='0', destination='1', edge_attr='2')

        # run BFS starting at vertex 17
        df = cugraph.bfs(G,  16)

        # Get the path to vertex 1
        p_df = cugraph.utils.get_traversed_path(df, 0)

        assert len(p_df) == 3

        # Get path to vertex 0 - which is not in graph
        p_df = cugraph.utils.get_traversed_path(df, 100)

        assert "not in the result set" in str(ErrorMsg)


def test_bfs_paths_array():
    with pytest.raises(ValueError) as ErrorMsg:
        gc.collect()

        graph_file = PurePath(utils.RAPIDS_DATASET_ROOT_DIR)/"karate.csv"

        cu_M = utils.read_csv_file(graph_file)

        G = cugraph.Graph()
        G.from_cudf_edgelist(cu_M, source='0', destination='1', edge_attr='2')

        # run BFS starting at vertex 17
        df = cugraph.bfs(G,  16)

        # Get the path to vertex 1
        answer = cugraph.utils.get_traversed_path_list(df, 0)

        assert len(answer) == 3

        # Get path to vertex 0 - which is not in graph
        answer = cugraph.utils.get_traversed_path_list(df, 100)

        assert "not in the result set" in str(ErrorMsg)
