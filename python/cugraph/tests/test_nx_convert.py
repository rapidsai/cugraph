# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
from cugraph.utilities.utils import is_device_version_less_than

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx


def _compare_graphs(nxG, cuG, has_wt=True):
    assert nxG.number_of_nodes() == cuG.number_of_nodes()
    assert nxG.number_of_edges() == cuG.number_of_edges()

    cu_df = cuG.view_edge_list().to_pandas()
    if has_wt is True:
        cu_df = cu_df.drop(columns=["weights"])
    cu_df = cu_df.sort_values(by=["src", "dst"]).reset_index(drop=True)

    nx_df = nx.to_pandas_edgelist(nxG)
    if has_wt is True:
        nx_df = nx_df.drop(columns=["weight"])
    nx_df = nx_df.rename(columns={"source": "src", "target": "dst"})
    nx_df = nx_df.astype('int32')
    nx_df = nx_df.sort_values(by=["src", "dst"]).reset_index(drop=True)

    assert cu_df.to_dict() == nx_df.to_dict()


@pytest.mark.skipif(
    is_device_version_less_than((7, 0)), reason="Not supported on Pascal"
)
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_networkx_compatibility(graph_file):
    # test to make sure cuGraph and Nx build similar Graphs

    gc.collect()

    # Read in the graph
    M = utils.read_csv_for_nx(graph_file, read_weights_in_sp=True)

    # create a NetworkX DiGraph
    nxG = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight",
        create_using=nx.DiGraph()
    )

    # create a cuGraph DiGraph
    gdf = cudf.from_pandas(M)
    gdf = gdf.rename(columns={"weight": "weights"})
    cuG = cugraph.from_cudf_edgelist(
        gdf,
        source="0",
        destination="1",
        edge_attr="weights",
        create_using=cugraph.DiGraph,
    )

    _compare_graphs(nxG, cuG)


@pytest.mark.skipif(
    is_device_version_less_than((7, 0)), reason="Not supported on Pascal"
)
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_nx_convert(graph_file):
    gc.collect()

    # read data and create a Nx Graph
    nx_df = utils.read_csv_for_nx(graph_file)
    nxG = nx.from_pandas_edgelist(nx_df, "0", "1", create_using=nx.DiGraph)

    cuG = cugraph.utilities.convert_from_nx(nxG)

    _compare_graphs(nxG, cuG, has_wt=False)


@pytest.mark.skipif(
    is_device_version_less_than((7, 0)), reason="Not supported on Pascal"
)
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_nx_convert_multicol(graph_file):
    gc.collect()

    # read data and create a Nx Graph
    nx_df = utils.read_csv_for_nx(graph_file)

    G = nx.DiGraph()

    for row in nx_df.iterrows():
        G.add_edge(
            row[1]["0"], row[1]["1"], count=[row[1]["0"], row[1]["1"]]
        )

    nxG = nx.from_pandas_edgelist(nx_df, "0", "1")

    cuG = cugraph.utilities.convert_from_nx(nxG)

    assert nxG.number_of_nodes() == cuG.number_of_nodes()
    assert nxG.number_of_edges() == cuG.number_of_edges()
