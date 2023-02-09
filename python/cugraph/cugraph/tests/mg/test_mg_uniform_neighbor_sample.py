# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
import random
import os

import pytest
import cudf
import dask_cudf
from pylibcugraph.testing.utils import gen_fixture_params_product

import cugraph.dask as dcg
import cugraph

from cugraph.dask import uniform_neighbor_sample
from cugraph.experimental.datasets import DATASETS_UNDIRECTED, email_Eu_core, small_tree

# If the rapids-pytest-benchmark plugin is installed, the "gpubenchmark"
# fixture will be available automatically. Check that this fixture is available
# by trying to import rapids_pytest_benchmark, and if that fails, set
# "gpubenchmark" to the standard "benchmark" fixture provided by
# pytest-benchmark.
try:
    import rapids_pytest_benchmark  # noqa: F401
except ImportError:
    import pytest_benchmark

    gpubenchmark = pytest_benchmark.plugin.benchmark

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# =============================================================================
# Pytest fixtures
# =============================================================================
IS_DIRECTED = [True, False]

datasets = DATASETS_UNDIRECTED + [email_Eu_core]

fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    (IS_DIRECTED, "directed"),
    ([False, True], "with_replacement"),
    (["int32", "float32"], "indices_type"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(
        zip(
            ("graph_file", "directed", "with_replacement", "indices_type"),
            request.param,
        )
    )

    indices_type = parameters["indices_type"]

    input_data_path = parameters["graph_file"].get_path()
    directed = parameters["directed"]

    chunksize = dcg.get_chunksize(input_data_path)
    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", indices_type],
    )

    dg = cugraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(
        ddf,
        source="src",
        destination="dst",
        edge_attr="value",
        store_transposed=False,
        legacy_renum_only=True,
    )

    parameters["MGGraph"] = dg

    # sample k vertices from the cuGraph graph
    k = random.randint(1, 3)
    srcs = dg.input_df["src"]
    dsts = dg.input_df["dst"]

    vertices = dask_cudf.concat([srcs, dsts]).drop_duplicates().compute()
    start_list = vertices.sample(k).astype("int32")

    # Generate a random fanout_vals list of length random(1, k)
    fanout_vals = [random.randint(1, k) for _ in range(random.randint(1, k))]

    # These prints are for debugging purposes since the vertices and the
    # fanout_vals are randomly sampled/chosen
    print("\nstart_list: \n", start_list)
    print("fanout_vals: ", fanout_vals)

    parameters["start_list"] = start_list
    parameters["fanout_vals"] = fanout_vals

    return parameters


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.cugraph_ops
def test_mg_uniform_neighbor_sample_simple(dask_client, input_combo):

    dg = input_combo["MGGraph"]

    input_df = dg.input_df
    result_nbr = uniform_neighbor_sample(
        dg,
        input_combo["start_list"],
        input_combo["fanout_vals"],
        input_combo["with_replacement"],
    )

    # multi edges are dropped to easily verify that each edge in the
    # results is present in the input dataframe
    result_nbr = result_nbr.drop_duplicates()

    # FIXME: The indices are not included in the comparison because garbage
    # value are intermittently retuned. This observation is observed when
    # passing float weights
    join = result_nbr.merge(
        input_df, left_on=[*result_nbr.columns[:2]], right_on=[*input_df.columns[:2]]
    )

    if len(result_nbr) != len(join):
        join2 = input_df.merge(
            result_nbr,
            how="right",
            left_on=[*input_df.columns],
            right_on=[*result_nbr.columns],
        )
        # The left part of the datasets shows which edge is missing from the
        # right part where the left and right part are respectively the
        # uniform-neighbor-sample results and the input dataframe.
        difference = (
            join2.sort_values([*result_nbr.columns])
            .compute()
            .to_pandas()
            .query("src.isnull()", engine="python")
        )

        invalid_edge = difference[difference.columns[:3]]
        raise Exception(
            f"\nThe edges below from uniform-neighbor-sample "
            f"are invalid\n {invalid_edge}"
        )

    # Ensure the right indices type is returned
    assert result_nbr["indices"].dtype == input_combo["indices_type"]

    sampled_vertex_result = (
        dask_cudf.concat([result_nbr["sources"], result_nbr["destinations"]])
        .drop_duplicates()
        .compute()
        .reset_index(drop=True)
    )

    sampled_vertex_result = sampled_vertex_result.to_pandas()
    start_list = input_combo["start_list"].to_pandas()

    if not set(start_list).issubset(set(sampled_vertex_result)):
        missing_vertex = set(start_list) - set(sampled_vertex_result)
        missing_vertex = list(missing_vertex)
        # compute the out-degree of the missing vertices
        out_degree = dg.out_degree(missing_vertex)

        out_degree = out_degree[out_degree.degree != 0]
        # If the missing vertices have outgoing edges, return an error
        if len(out_degree) != 0:
            missing_vertex = out_degree["vertex"].compute().to_pandas().to_list()
            raise Exception(
                f"vertex {missing_vertex} is missing from "
                f"uniform neighbor sampling results"
            )


@pytest.mark.cugraph_ops
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_mg_uniform_neighbor_sample_tree(dask_client, directed):

    input_data_path = small_tree.get_path()
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    G = cugraph.Graph(directed=directed)
    G.from_dask_cudf_edgelist(
        ddf, "src", "dst", "value", store_transposed=False, legacy_renum_only=True
    )

    # TODO: Incomplete, include more testing for tree graph as well as
    # for larger graphs
    start_list = cudf.Series([0, 0], dtype="int32")
    fanout_vals = [4, 1, 3]
    with_replacement = True
    result_nbr = uniform_neighbor_sample(G, start_list, fanout_vals, with_replacement)

    result_nbr = result_nbr.drop_duplicates()

    # input_df != ddf if 'directed = False' because ddf will be symmetrized
    # internally.
    input_df = G.input_df
    join = result_nbr.merge(
        input_df, left_on=[*result_nbr.columns[:2]], right_on=[*input_df.columns[:2]]
    )

    assert len(join) == len(result_nbr)
    # Since the validity of results have (probably) been tested at both the C++
    # and C layers, simply test that the python interface and conversions were
    # done correctly.
    assert result_nbr["sources"].dtype == "int32"
    assert result_nbr["destinations"].dtype == "int32"
    assert result_nbr["indices"].dtype == "float32"

    result_nbr_vertices = (
        dask_cudf.concat([result_nbr["sources"], result_nbr["destinations"]])
        .drop_duplicates()
        .compute()
        .reset_index(drop=True)
    )

    result_nbr_vertices = result_nbr_vertices.to_pandas()
    start_list = start_list.to_pandas()

    # The vertices in start_list must be a subsets of the vertices
    # in the result
    assert set(start_list).issubset(set(result_nbr_vertices))


@pytest.mark.cugraph_ops
def test_mg_uniform_neighbor_sample_unweighted(dask_client):
    df = cudf.DataFrame(
        {
            "src": cudf.Series([0, 1, 2, 2, 0, 1, 4, 4], dtype="int32"),
            "dst": cudf.Series([3, 2, 1, 4, 1, 3, 1, 2], dtype="int32"),
        }
    )

    df = dask_cudf.from_cudf(df, npartitions=2)

    G = cugraph.Graph()
    G.from_dask_cudf_edgelist(
        df, source="src", destination="dst", legacy_renum_only=True
    )

    start_list = cudf.Series([0], dtype="int32")
    fanout_vals = [-1]
    with_replacement = True

    sampling_results = uniform_neighbor_sample(
        G, start_list, fanout_vals, with_replacement
    )

    expected_src = [0, 0]
    actual_src = sampling_results.sources
    actual_src = actual_src.compute().to_arrow().to_pylist()
    assert sorted(actual_src) == sorted(expected_src)

    expected_dst = [3, 1]
    actual_dst = sampling_results.destinations
    actual_dst = actual_dst.compute().to_arrow().to_pylist()
    assert sorted(actual_dst) == sorted(expected_dst)


@pytest.mark.cugraph_ops
def test_mg_uniform_neighbor_sample_ensure_no_duplicates(dask_client):
    # See issue #2760
    # This ensures that the starts are properly distributed

    df = cudf.DataFrame({"src": [6, 6, 6, 6], "dst": [7, 9, 10, 11]})
    df = df.astype("int32")

    dask_df = dask_cudf.from_cudf(df, npartitions=2)

    mg_G = cugraph.MultiGraph(directed=True)
    mg_G.from_dask_cudf_edgelist(
        dask_df, source="src", destination="dst", renumber=True, legacy_renum_only=True
    )

    output_df = cugraph.dask.uniform_neighbor_sample(
        mg_G,
        cudf.Series([6]).astype("int32"),
        fanout_vals=[3],
        with_replacement=False,
    )

    assert len(output_df.compute()) == 3


@pytest.mark.cugraph_ops
def test_uniform_neighbor_sample_edge_properties():
    edgelist_df = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "src": [0, 1, 2, 3, 4, 3, 4, 2, 0, 1, 0, 2],
                "dst": [1, 2, 4, 2, 3, 4, 1, 1, 2, 3, 4, 4],
                "eid": cudf.Series(
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype="int64"
                ),
                "etp": cudf.Series([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0], dtype="int32"),
                "w": [0.0, 0.1, 0.2, 3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10, 0.11],
            }
        ),
        npartitions=2,
    )

    start_df = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "seed": cudf.Series([0, 4], dtype="int64"),
                "batch": cudf.Series([0, 1], dtype="int32"),
            }
        ),
        npartitions=2,
    )

    G = cugraph.MultiGraph(directed=True)
    G.from_dask_cudf_edgelist(
        edgelist_df,
        source="src",
        destination="dst",
        edge_attr=["w", "eid", "etp"],
        legacy_renum_only=True,
    )

    sampling_results = uniform_neighbor_sample(
        G,
        start_list=start_df["seed"],
        fanout_vals=[2, 2],
        with_replacement=False,
        with_edge_properties=True,
        batch_id_list=start_df["batch"],
    ).compute()

    sampling_results = cugraph.dask.uniform_neighbor_sample(
        G,
        start_list=start_df["seed"].compute(),
        fanout_vals=[2, 2],
        with_replacement=False,
        with_edge_properties=True,
        batch_id_list=start_df["batch"].compute(),
    ).compute()

    print("original edgelist:")
    print(edgelist_df.compute())

    print("sampling result:")
    print(sampling_results)

    mdf = cudf.merge(
        sampling_results, edgelist_df.compute(), left_on="edge_id", right_on="eid"
    )
    assert (mdf.w == mdf.weight).all()
    assert (mdf.etp == mdf.edge_type).all()
    assert (mdf.src == mdf.sources).all()
    assert (mdf.dst == mdf.destinations).all()

    assert sorted(sampling_results["hop_id"].values_host.tolist()) == [0] * (2 * 2) + [
        1
    ] * (2 * 2 * 2)
    # FIXME test the batch id values once that is fixed in C++


def test_uniform_neighbor_sample_edge_properties_self_loops():
    df = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "src": [0, 1, 2],
                "dst": [0, 1, 2],
                "eid": [2, 4, 6],
                "etp": cudf.Series([1, 1, 2], dtype="int32"),
                "w": [0.0, 0.1, 0.2],
            }
        ),
        npartitions=2,
    )

    G = cugraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(
        df,
        source="src",
        destination="dst",
        edge_attr=["w", "eid", "etp"],
        legacy_renum_only=True,
    )

    sampling_results = cugraph.dask.uniform_neighbor_sample(
        G,
        start_list=dask_cudf.from_cudf(cudf.Series([0, 1, 2]), npartitions=2),
        batch_id_list=dask_cudf.from_cudf(
            cudf.Series([1, 1, 1], dtype="int32"), npartitions=2
        ),
        fanout_vals=[2, 2],
        with_replacement=False,
        with_edge_properties=True,
    ).compute()

    assert sorted(sampling_results.sources.values_host.tolist()) == [0, 0, 1, 1, 2, 2]
    assert sorted(sampling_results.destinations.values_host.tolist()) == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    assert sorted(sampling_results.weight.values_host.tolist()) == [
        0.0,
        0.0,
        0.1,
        0.1,
        0.2,
        0.2,
    ]
    assert sorted(sampling_results.edge_id.values_host.tolist()) == [2, 2, 4, 4, 6, 6]
    assert sorted(sampling_results.edge_type.values_host.tolist()) == [1, 1, 1, 1, 2, 2]
    assert sorted(sampling_results.batch_id.values_host.tolist()) == [1, 1, 1, 1, 1, 1]
    assert sorted(sampling_results.hop_id.values_host.tolist()) == [0, 0, 0, 1, 1, 1]


@pytest.mark.parametrize("with_replacement", [True, False])
@pytest.mark.skipif(
    int(os.getenv("DASK_NUM_WORKERS", 2)) < 2, reason="too few workers to test"
)
def test_uniform_neighbor_edge_properties_sample_small_start_list(with_replacement):
    df = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "src": [0, 1, 2],
                "dst": [0, 1, 2],
                "eid": [2, 4, 6],
                "etp": cudf.Series([1, 1, 2], dtype="int32"),
                "w": [0.0, 0.1, 0.2],
            }
        ),
        npartitions=2,
    )

    G = cugraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(
        df,
        source="src",
        destination="dst",
        edge_attr=["w", "eid", "etp"],
        legacy_renum_only=True,
    )

    cugraph.dask.uniform_neighbor_sample(
        G,
        start_list=cudf.Series([0]),
        fanout_vals=[10, 25],
        with_replacement=with_replacement,
        with_edge_properties=True,
        batch_id_list=cudf.Series([10], dtype="int32"),
    )


def test_uniform_neighbor_sample_without_dask_inputs():
    df = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "src": [0, 1, 2],
                "dst": [0, 1, 2],
                "eid": [2, 4, 6],
                "etp": cudf.Series([1, 1, 2], dtype="int32"),
                "w": [0.0, 0.1, 0.2],
            }
        ),
        npartitions=2,
    )

    G = cugraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(
        df,
        source="src",
        destination="dst",
        edge_attr=["w", "eid", "etp"],
        legacy_renum_only=True,
    )

    sampling_results = cugraph.dask.uniform_neighbor_sample(
        G,
        start_list=cudf.Series([0, 1, 2]),
        batch_id_list=cudf.Series([1, 1, 1], dtype="int32"),
        fanout_vals=[2, 2],
        with_replacement=False,
        with_edge_properties=True,
    ).compute()

    assert sorted(sampling_results.sources.values_host.tolist()) == [0, 0, 1, 1, 2, 2]
    assert sorted(sampling_results.destinations.values_host.tolist()) == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    assert sorted(sampling_results.weight.values_host.tolist()) == [
        0.0,
        0.0,
        0.1,
        0.1,
        0.2,
        0.2,
    ]
    assert sorted(sampling_results.edge_id.values_host.tolist()) == [2, 2, 4, 4, 6, 6]
    assert sorted(sampling_results.edge_type.values_host.tolist()) == [1, 1, 1, 1, 2, 2]
    assert sorted(sampling_results.batch_id.values_host.tolist()) == [1, 1, 1, 1, 1, 1]
    assert sorted(sampling_results.hop_id.values_host.tolist()) == [0, 0, 0, 1, 1, 1]


# =============================================================================
# Benchmarks
# =============================================================================


@pytest.mark.slow
@pytest.mark.parametrize("n_samples", [1_000, 5_000, 10_000])
def bench_uniform_neigbour_sample_email_eu_core(gpubenchmark, dask_client, n_samples):
    input_data_path = email_Eu_core.get_path()
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "int32"],
    )

    dg = cugraph.Graph(directed=False)
    dg.from_dask_cudf_edgelist(
        ddf,
        source="src",
        destination="dst",
        edge_attr="value",
        store_transposed=False,
        legacy_renum_only=True,
    )
    # Partition the dataframe to add in chunks
    srcs = dg.input_df["src"]
    start_list = srcs[:n_samples].compute()

    def func():
        _ = cugraph.dask.uniform_neighbor_sample(dg, start_list, [10])
        del _

    gpubenchmark(func)
