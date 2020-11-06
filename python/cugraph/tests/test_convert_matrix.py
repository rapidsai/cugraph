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
import cugraph
from cugraph.tests import utils
import numpy as np

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_to_from_pandas(graph_file):
    gc.collect()

    # Read in the graph
    M = utils.read_csv_for_nx(graph_file, read_weights_in_sp=True)

    # create a NetworkX DiGraph and convert to pandas adjacency
    nxG = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight",
        create_using=nx.DiGraph
    )
    nx_pdf = nx.to_pandas_adjacency(nxG)
    nx_pdf = nx_pdf[sorted(nx_pdf.columns)]
    nx_pdf.sort_index(inplace=True)

    # create a cugraph DiGraph and convert to pandas adjacency
    cuG = cugraph.from_pandas_edgelist(
        M, source="0", destination="1", edge_attr="weight",
        create_using=cugraph.DiGraph
    )

    cu_pdf = cugraph.to_pandas_adjacency(cuG)
    cu_pdf = cu_pdf[sorted(cu_pdf.columns)]
    cu_pdf.sort_index(inplace=True)

    # Compare pandas adjacency list
    assert nx_pdf.equals(cu_pdf)

    # Convert pandas adjacency list to graph
    new_nxG = nx.from_pandas_adjacency(nx_pdf, create_using=nx.DiGraph)
    new_cuG = cugraph.from_pandas_adjacency(cu_pdf,
                                            create_using=cugraph.DiGraph)

    # Compare pandas edgelist
    exp_pdf = nx.to_pandas_edgelist(new_nxG)
    res_pdf = cugraph.to_pandas_edgelist(new_cuG)

    exp_pdf = exp_pdf.rename(columns={"source": "src", "target": "dst",
                                      "weight": "weights"})

    exp_pdf = exp_pdf.sort_values(by=["src", "dst"]).reset_index(drop=True)
    res_pdf = res_pdf.sort_values(by=["src", "dst"]).reset_index(drop=True)
    res_pdf = res_pdf[['src', 'dst', 'weights']]

    assert exp_pdf.equals(res_pdf)


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_from_to_numpy(graph_file):
    gc.collect()

    # Read in the graph
    M = utils.read_csv_for_nx(graph_file, read_weights_in_sp=True)

    # create NetworkX and cugraph DiGraph
    nxG = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight",
        create_using=nx.DiGraph
    )

    cuG = cugraph.from_pandas_edgelist(
        M, source="0", destination="1", edge_attr="weight",
        create_using=cugraph.DiGraph
    )

    # convert graphs to numpy array
    nparray_nx = nx.to_numpy_array(nxG,
                                   nodelist=cuG.nodes().values_host)
    nparray_cu = cugraph.to_numpy_array(cuG)
    npmatrix_nx = nx.to_numpy_matrix(nxG,
                                     nodelist=cuG.nodes().values_host)
    npmatrix_cu = cugraph.to_numpy_matrix(cuG)

    # Compare arrays and matrices
    assert np.array_equal(nparray_nx, nparray_cu)
    assert np.array_equal(np.asarray(npmatrix_nx),
                          np.asarray(npmatrix_cu))

    # Create graphs from numpy array
    new_nxG = nx.from_numpy_array(nparray_nx,
                                  create_using=nx.DiGraph)
    new_cuG = cugraph.from_numpy_array(nparray_cu,
                                       create_using=cugraph.DiGraph)

    # Assert graphs are same
    exp_pdf = nx.to_pandas_edgelist(new_nxG)
    res_pdf = cugraph.to_pandas_edgelist(new_cuG)

    exp_pdf = exp_pdf.rename(columns={"source": "src", "target": "dst",
                                      "weight": "weights"})

    exp_pdf = exp_pdf.sort_values(by=["src", "dst"]).reset_index(drop=True)
    res_pdf = res_pdf.sort_values(by=["src", "dst"]).reset_index(drop=True)
    res_pdf = res_pdf[['src', 'dst', 'weights']]

    assert exp_pdf.equals(res_pdf)

    # Create graphs from numpy matrix
    new_nxG = nx.from_numpy_matrix(npmatrix_nx,
                                   create_using=nx.DiGraph)
    new_cuG = cugraph.from_numpy_matrix(npmatrix_cu,
                                        create_using=cugraph.DiGraph)

    # Assert graphs are same
    exp_pdf = nx.to_pandas_edgelist(new_nxG)
    res_pdf = cugraph.to_pandas_edgelist(new_cuG)

    exp_pdf = exp_pdf.rename(columns={"source": "src", "target": "dst",
                                      "weight": "weights"})

    exp_pdf = exp_pdf.sort_values(by=["src", "dst"]).reset_index(drop=True)
    res_pdf = res_pdf.sort_values(by=["src", "dst"]).reset_index(drop=True)
    res_pdf = res_pdf[['src', 'dst', 'weights']]

    assert exp_pdf.equals(res_pdf)
