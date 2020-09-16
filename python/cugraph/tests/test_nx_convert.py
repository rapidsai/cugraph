# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import cudf
import cugraph
from cugraph.tests import utils

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx


# Test
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_nx_convert(graph_file):
    gc.collect()

    # read data and create a Nx Graph
    nx_df = utils.read_csv_for_nx(graph_file)
    nxG = nx.from_pandas_edgelist(nx_df, "0", "1")

    cuG = cugraph.utilities.convert_from_nx(nxG)

    assert nxG.number_of_nodes() == cuG.number_of_nodes()
    assert nxG.number_of_edges() == cuG.number_of_edges()
    