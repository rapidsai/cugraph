# Copyright (c) 2019, NVIDIA CORPORATION.
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

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx


print("Networkx version : {} ".format(nx.__version__))


def calc_core_number(graph_file):
    M = utils.read_csv_file(graph_file)
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(M, source="0", destination="1")

    cn = cugraph.core_number(G)

    NM = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        NM, source="0", target="1", create_using=nx.Graph()
    )
    nc = nx.core_number(Gnx)
    pdf = [nc[k] for k in sorted(nc.keys())]
    cn["nx_core_number"] = pdf
    cn = cn.rename({"core_number": "cu_core_number"})
    return cn


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_core_number(graph_file):
    gc.collect()

    cn = calc_core_number(graph_file)

    assert cn["cu_core_number"].equals(cn["nx_core_number"])
