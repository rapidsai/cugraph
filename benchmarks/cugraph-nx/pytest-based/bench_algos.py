# Copyright (c) 2023, NVIDIA CORPORATION.
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

import os
import time

import networkx as nx
import pandas as pd
import pytest
from pylibcugraph.testing import gen_fixture_params_product

from cugraph_benchmarking.params import (
    directed_datasets,
    undirected_datasets,
)


################################################################################
# FIXME: add a directed attr to Dataset. For now, manually add an attr.
for param in directed_datasets:
    param.values[0].directed = True
for param in undirected_datasets:
    param.values[0].directed = False

backend_params = ["cugraph", None]
dataset_params = directed_datasets + undirected_datasets
#dataset_params = [directed_datasets[0]]
bench_inputs_params = gen_fixture_params_product(
    (backend_params, "backend"),
    (dataset_params, "ds"),
)

from networkx.classes import backends
backends.plugin_name = "cugraph"
orig_dispatch = backends._dispatch
test_dispatch = backends.test_override_dispatch
dispatch_decorator = orig_dispatch

@pytest.fixture(scope="module", params=bench_inputs_params)
def graph_obj_for_dispatch(request):
    global dispatch_decorator
    (backend, dataset) = request.param[:2]
    if backend == "cugraph":
        dispatch_decorator = test_dispatch
    elif backend is None:
        dispatch_decorator = orig_dispatch
    else:
        raise ValueError(f"got unsupported {backend=}")

    create_using = nx.DiGraph if dataset.directed else nx.Graph

    #pandas_edgelist = dataset.get_edgelist().to_pandas()
    header = None
    if isinstance(dataset.metadata["header"], int):
        header = dataset.metadata["header"]

    names=dataset.metadata["col_names"]
    dtypes=dataset.metadata["col_types"]

    print(f"reading csv {dataset.get_path()}...", flush=True, end="")
    st = time.time()
    pandas_edgelist = pd.read_csv(dataset.get_path(),
                                  delimiter=dataset.metadata["delim"],
                                  names=names,
                                  dtype=dict(zip(names, dtypes)),
                                  header=header,
                                  )
    print(f"done in {time.time() - st} seconds", flush=True)

    print("creating graph from pandas edgelist...", flush=True, end="")
    st=time.time()
    G = nx.from_pandas_edgelist(pandas_edgelist,
                                source=names[0],
                                target=names[1],
                                create_using=create_using)
    print(f"done in {time.time() - st} seconds", flush=True)

    yield G

    dispatch_decorator = orig_dispatch


################################################################################
@pytest.mark.parametrize("normalized", [True, False], ids=lambda norm: f"{norm=}")
def bench_betweenness_centrality(benchmark, graph_obj_for_dispatch, normalized):
    backends._registered_algorithms = {}
    benchmark(dispatch_decorator(nx.betweenness_centrality),
              graph_obj_for_dispatch, weight=None, normalized=normalized)


@pytest.mark.parametrize("normalized", [True, False], ids=lambda norm: f"{norm=}")
def bench_edge_betweenness_centrality(benchmark, graph_obj_for_dispatch, normalized):
    backends._registered_algorithms = {}
    benchmark(dispatch_decorator(nx.edge_betweenness_centrality),
              graph_obj_for_dispatch, weight=None, normalized=normalized)
