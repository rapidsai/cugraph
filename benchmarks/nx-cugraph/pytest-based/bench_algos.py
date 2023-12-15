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

import random

import networkx as nx
import pandas as pd
import pytest
from cugraph import datasets
import nx_cugraph as nxcg

# Attempt to import the NetworkX dispatching module, which is only needed when
# testing with NX <3.2 in order to dynamically switch backends. NX >=3.2 allows
# the backend to be specified directly in the API call.
try:
    from networkx.classes import backends  # NX <3.2
except ImportError:
    backends = None


################################################################################
# Fixtures and params

# See https://pytest-benchmark.readthedocs.io/en/latest/glossary.html for how
# these variables are used.
rounds = 1
iterations = 1
warmup_rounds = 1

dataset_param_values = [
    pytest.param(datasets.karate, marks=[pytest.mark.small, pytest.mark.undirected]),
    pytest.param(datasets.netscience, marks=[pytest.mark.small, pytest.mark.directed]),
    pytest.param(
        datasets.email_Eu_core, marks=[pytest.mark.small, pytest.mark.directed]
    ),
    pytest.param(
        datasets.cit_patents, marks=[pytest.mark.medium, pytest.mark.directed]
    ),
    pytest.param(
        datasets.hollywood, marks=[pytest.mark.medium, pytest.mark.undirected]
    ),
    pytest.param(
        datasets.soc_livejournal, marks=[pytest.mark.medium, pytest.mark.directed]
    ),
    pytest.param(
        datasets.europe_osm, marks=[pytest.mark.large, pytest.mark.undirected]
    ),
]

backend_param_values = ["cugraph", "cugraph-preconverted", None]


def setup_module(module):
    """
    Trivial conversion call to force various one-time CUDA initialization
    operations to happen outside of benchmarks.
    """
    G = nx.karate_club_graph()
    nxcg.from_networkx(G)


# Test IDs are generated using the lambda assigned to the ids arg to provide an
# easier-to-read name. This is especially helpful for Dataset objs (see
# https://docs.pytest.org/en/stable/reference/reference.html#pytest-fixture)
@pytest.fixture(
    scope="module", params=dataset_param_values, ids=lambda ds: f"ds={str(ds)}"
)
def graph_obj(request):
    """
    Returns a NX Graph or DiGraph obj from the dataset instance parameter.
    """
    dataset = request.param
    return nx_graph_from_dataset(dataset)


@pytest.fixture(
    scope="module",
    params=backend_param_values,
    ids=lambda backend: f"backend={backend}",
)
def backend(request):
    """
    Returns the backend name to use. This is done as a fixture for consistency
    and simplicity when creating benchmarks (no need to mark the benchmark as
    parametrized).
    """
    return request.param


################################################################################
# Helpers
def nx_graph_from_dataset(dataset_obj):
    """
    Read the dataset specified by the dataset_obj and create and return a
    nx.Graph or nx.DiGraph instance based on the dataset is_directed metadata.
    """
    create_using = nx.DiGraph if dataset_obj.metadata["is_directed"] else nx.Graph
    names = dataset_obj.metadata["col_names"]
    dtypes = dataset_obj.metadata["col_types"]
    if isinstance(dataset_obj.metadata["header"], int):
        header = dataset_obj.metadata["header"]
    else:
        header = None

    pandas_edgelist = pd.read_csv(
        dataset_obj.get_path(),
        delimiter=dataset_obj.metadata["delim"],
        names=names,
        dtype=dict(zip(names, dtypes)),
        header=header,
    )
    G = nx.from_pandas_edgelist(
        pandas_edgelist, source=names[0], target=names[1], create_using=create_using
    )
    return G


def get_legacy_backend_wrapper(backend_name):
    """
    Returns a callable that wraps an algo function with either the default
    dispatch decorator, or the "testing" decorator which unconditionally
    dispatches.
    This is only supported for NetworkX <3.2
    """
    backends.plugin_name = "cugraph"
    orig_dispatch = backends._dispatch
    testing_dispatch = backends.test_override_dispatch

    # Testing with the networkx <3.2 dispatch mechanism is based on decorating
    # networkx APIs. The decorator is either one that only uses a backend if
    # the input graph type is for that backend (the default decorator), or the
    # "testing" decorator, which unconditionally converts a graph type to the
    # type needed by the backend then calls the backend. If the cugraph backend
    # is specified, create a callable that decorates the benchmarked function
    # with the testing decorator.
    #
    # Because both the default and testing decorators assume they are only
    # applied once and do bookkeeping to ensure algos are not registered
    # multiple times, the callable also clears bookkeeping so the decorators
    # can be reapplied multiple times. This is obviously a hack and networkx
    # >=3.2 makes this use case properly supported.
    backends._registered_algorithms = {}
    if backend_name == "cugraph":
        wrap_callable_for_dispatch = testing_dispatch(*args, **kwargs)
    else:
        wrap_callable_for_dispatch = orig_dispatch(*args, **kwargs)

    return wrap_callable_for_dispatch


def get_backend_wrapper(backend_name):
    """
    Returns a callable that wraps an algo function in order to set the
    "backend" kwarg on it.
    This is only supported for NetworkX >= 3.2
    """

    def wrap_callable_for_dispatch(func):
        def wrapper(*args, **kwargs):
            kwargs["backend"] = backend_name
            return func(*args, **kwargs)

        return wrapper

    return wrap_callable_for_dispatch


@pytest.fixture(
    scope="module",
    params=backend_param_values,
    ids=lambda backend: f"backend={backend}",
)
def backend_wrapper(request):
    """
    Returns a callable that takes a function algo and wraps it in another
    function that calls the algo using the appropriate backend.

    For example: if the backend to test is "cugraph", this will return a
    function that calls nx.pagerank(..., backend='cugraph')
    """
    backend_name = request.param
    actual_backend_name = backend_name

    # Special case: cugraph-preconverted may be specified as a backend but this
    # name is reserved to indicate a cugraph backend is to be used with a
    # preconverted graph obj (rather than having the backend do the
    # conversion).
    if backend_name == "cugraph-preconverted":
        actual_backend_name = "cugraph"

    # NX <3.2 does not support the backends= kwarg, so the backend must be
    # enabled differently
    if backends is not None:
        wrapper = get_legacy_backend_wrapper(actual_backend_name)
    else:
        wrapper = get_backend_wrapper(actual_backend_name)

    wrapper.backend_name = backend_name
    return wrapper


def get_graph_obj_for_benchmark(graph_obj, backend_wrapper):
    """
    Given a Graph object and a backend name, return a converted Graph or the
    original Graph object based on the backend to use.

    This is used since some backend names are actually used as descriptions for
    combinations of backends and converted/non-converted graphs.
    """
    G = graph_obj
    if backend_wrapper.backend_name == "cugraph-preconverted":
        G = nxcg.from_networkx(G)
    return G


################################################################################
# Benchmarks
# normalized_param_values = [True, False]
# k_param_values = [10, 100]
normalized_param_values = [True]
k_param_values = [10]


@pytest.mark.parametrize(
    "normalized", normalized_param_values, ids=lambda norm: f"{norm=}"
)
@pytest.mark.parametrize("k", k_param_values, ids=lambda k: f"{k=}")
def bench_betweenness_centrality(benchmark, graph_obj, backend_wrapper, normalized, k):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.betweenness_centrality),
        args=(G,),
        kwargs=dict(
            weight=None,
            normalized=normalized,
            k=k,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


@pytest.mark.parametrize(
    "normalized", normalized_param_values, ids=lambda norm: f"{norm=}"
)
@pytest.mark.parametrize("k", k_param_values, ids=lambda k: f"{k=}")
def bench_edge_betweenness_centrality(
    benchmark, graph_obj, backend_wrapper, normalized, k
):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.edge_betweenness_centrality),
        args=(G,),
        kwargs=dict(
            weight=None,
            normalized=normalized,
            k=k,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


def bench_louvain_communities(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    # The cugraph backend for louvain_communities only supports undirected graphs
    if G.is_directed():
        G = G.to_undirected()
    result = benchmark.pedantic(
        target=backend_wrapper(nx.community.louvain_communities),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is list


def bench_degree_centrality(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.degree_centrality),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


def bench_eigenvector_centrality(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.eigenvector_centrality),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


@pytest.mark.parametrize(
    "normalized", normalized_param_values, ids=lambda norm: f"{norm=}"
)
def bench_hits(benchmark, graph_obj, backend_wrapper, normalized):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.hits),
        args=(G,),
        kwargs=dict(
            normalized=normalized,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is tuple
    assert len(result) == 2
    assert type(result[0]) is dict
    assert type(result[1]) is dict


def bench_in_degree_centrality(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.in_degree_centrality),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


@pytest.mark.parametrize(
    "normalized", normalized_param_values, ids=lambda norm: f"{norm=}"
)
def bench_katz_centrality(benchmark, graph_obj, backend_wrapper, normalized):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.katz_centrality),
        args=(G,),
        kwargs=dict(
            normalized=normalized,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


def bench_k_truss(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    # DiGraphs are not supported
    if G.is_directed():
        G = G.to_undirected()
    result = benchmark.pedantic(
        target=backend_wrapper(nx.k_truss),
        args=(G,),
        kwargs=dict(
            k=2,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    # Check that this at least appears to be some kind of NX-like Graph
    assert hasattr(result, "has_node")


def bench_out_degree_centrality(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.out_degree_centrality),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


def bench_pagerank(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.pagerank),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


def bench_single_source_shortest_path_length(benchmark, graph_obj, backend_wrapper):
    # Use the node with the highest degree
    degrees = graph_obj.degree()  # list of tuples of (node, degree)
    node = max(degrees, key=lambda t: t[1])[0]
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)

    result = benchmark.pedantic(
        target=backend_wrapper(nx.single_source_shortest_path_length),
        args=(G,),
        kwargs=dict(
            source=node,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict
    # print(f"\n{node=}\n{result=}")


def bench_single_target_shortest_path_length(benchmark, graph_obj, backend_wrapper):
    # Use the node with the highest degree
    degrees = graph_obj.degree()  # list of tuples of (node, degree)
    node = max(degrees, key=lambda t: t[1])[0]
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)

    # Ensure the benchmark time include computation of each result from the
    # returned generator
    def run_generator(*args, **kwargs):
        func = backend_wrapper(nx.single_target_shortest_path_length)
        results = func(*args, **kwargs)
        return list(results)

    result = benchmark.pedantic(
        target=run_generator,
        args=(G,),
        kwargs=dict(
            target=node,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is list
