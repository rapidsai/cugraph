# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
import numpy as np
import networkx as nx

import cugraph
from cugraph.testing import utils


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_to_from_pandas(graph_file):
    # Read in the graph
    M = utils.read_csv_for_nx(graph_file, read_weights_in_sp=True)

    # create a NetworkX DiGraph and convert to pandas adjacency
    nxG = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.DiGraph
    )
    nx_pdf = nx.to_pandas_adjacency(nxG)
    nx_pdf = nx_pdf[sorted(nx_pdf.columns)]
    nx_pdf.sort_index(inplace=True)

    # create a cugraph Directed Graph and convert to pandas adjacency
    cuG = cugraph.from_pandas_edgelist(
        M,
        source="0",
        destination="1",
        edge_attr="weight",
        create_using=cugraph.Graph(directed=True),
    )

    cu_pdf = cugraph.to_pandas_adjacency(cuG)
    cu_pdf = cu_pdf[sorted(cu_pdf.columns)]
    cu_pdf.sort_index(inplace=True)

    # Compare pandas adjacency list
    assert nx_pdf.equals(cu_pdf)

    # Convert pandas adjacency list to graph
    new_nxG = nx.from_pandas_adjacency(nx_pdf, create_using=nx.DiGraph)
    new_cuG = cugraph.from_pandas_adjacency(
        cu_pdf, create_using=cugraph.Graph(directed=True)
    )

    # Compare pandas edgelist
    exp_pdf = nx.to_pandas_edgelist(new_nxG)
    res_pdf = cugraph.to_pandas_edgelist(new_cuG)

    exp_pdf = exp_pdf.rename(columns={"source": "src", "target": "dst"})

    exp_pdf = exp_pdf.sort_values(by=["src", "dst"]).reset_index(drop=True)
    res_pdf = res_pdf.sort_values(by=["src", "dst"]).reset_index(drop=True)
    res_pdf = res_pdf[["src", "dst", "weight"]]

    assert exp_pdf.equals(res_pdf)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_from_to_numpy(graph_file):
    # Read in the graph
    M = utils.read_csv_for_nx(graph_file, read_weights_in_sp=True)

    # create NetworkX and cugraph Directed Graph
    nxG = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.DiGraph
    )

    cuG = cugraph.from_pandas_edgelist(
        M,
        source="0",
        destination="1",
        edge_attr="weight",
        create_using=cugraph.Graph(directed=True),
    )

    # convert graphs to numpy array
    nparray_nx = nx.to_numpy_array(nxG, nodelist=cuG.nodes().values_host)
    nparray_cu = cugraph.to_numpy_array(cuG)
    npmatrix_nx = nx.to_numpy_array(nxG, nodelist=cuG.nodes().values_host)
    npmatrix_cu = cugraph.to_numpy_array(cuG)

    # Compare arrays and matrices
    assert np.array_equal(nparray_nx, nparray_cu)
    assert np.array_equal(np.asarray(npmatrix_nx), np.asarray(npmatrix_cu))

    # Create graphs from numpy array
    new_nxG = nx.from_numpy_array(nparray_nx, create_using=nx.DiGraph)
    new_cuG = cugraph.from_numpy_array(
        nparray_cu, create_using=cugraph.Graph(directed=True)
    )

    # Assert graphs are same
    exp_pdf = nx.to_pandas_edgelist(new_nxG)
    res_pdf = cugraph.to_pandas_edgelist(new_cuG)

    exp_pdf = exp_pdf.rename(columns={"source": "src", "target": "dst"})

    exp_pdf = exp_pdf.sort_values(by=["src", "dst"]).reset_index(drop=True)
    res_pdf = res_pdf.sort_values(by=["src", "dst"]).reset_index(drop=True)
    res_pdf = res_pdf[["src", "dst", "weight"]]

    assert exp_pdf.equals(res_pdf)

    # Create graphs from numpy matrix
    new_nxG = nx.from_numpy_array(npmatrix_nx, create_using=nx.DiGraph)
    new_cuG = cugraph.from_numpy_array(
        npmatrix_cu, create_using=cugraph.Graph(directed=True)
    )

    # Assert graphs are same
    exp_pdf = nx.to_pandas_edgelist(new_nxG)
    res_pdf = cugraph.to_pandas_edgelist(new_cuG)

    exp_pdf = exp_pdf.rename(columns={"source": "src", "target": "dst"})

    exp_pdf = exp_pdf.sort_values(by=["src", "dst"]).reset_index(drop=True)
    res_pdf = res_pdf.sort_values(by=["src", "dst"]).reset_index(drop=True)
    res_pdf = res_pdf[["src", "dst", "weight"]]

    assert exp_pdf.equals(res_pdf)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_from_edgelist(graph_file):
    """
    Compare the resulting Graph objs from cugraph.from_edgelist() calls of both
    a cudf and pandas DataFrame and ensure the results are equal.
    """
    df = utils.read_csv_file(graph_file)
    pdf = utils.read_csv_for_nx(graph_file)

    G1 = cugraph.from_edgelist(df, source="0", destination="1")
    G2 = cugraph.from_edgelist(pdf, source="0", destination="1")

    assert G1.EdgeList == G2.EdgeList


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_from_adjlist(graph_file):
    """
    Compare the resulting Graph objs from cugraph.from_adjlist() calls of both
    a cudf and pandas DataFrame and ensure the results are equal.
    """
    G = utils.generate_cugraph_graph_from_file(graph_file, directed=True)
    (cu_offsets, cu_indices, cu_vals) = G.view_adj_list()

    pd_offsets = cu_offsets.to_pandas()
    pd_indices = cu_indices.to_pandas()
    if cu_vals is not None:
        pd_vals = cu_vals.to_pandas()
    else:
        pd_vals = None

    # FIXME: should mixing types be allowed?
    with pytest.raises(TypeError):
        G1 = cugraph.from_adjlist(cu_offsets, pd_indices)
    with pytest.raises(TypeError):
        G1 = cugraph.from_adjlist(cu_offsets, cu_indices, cu_vals, create_using=33)

    G1 = cugraph.from_adjlist(
        cu_offsets, cu_indices, cu_vals, create_using=cugraph.Graph(directed=True)
    )
    G2 = cugraph.from_adjlist(
        pd_offsets, pd_indices, pd_vals, create_using=cugraph.Graph(directed=True)
    )

    assert G1.AdjList == G2.AdjList
