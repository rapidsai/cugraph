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
import cupy
import cudf
import dask_cudf
from pylibcugraph.testing.utils import gen_fixture_params_product
from cugraph.dask.common.mg_utils import is_single_gpu

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
@pytest.mark.mg
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


@pytest.mark.mg
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
    G.from_dask_cudf_edgelist(ddf, "src", "dst", "value", store_transposed=False)

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


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="FIXME: MG test fails on single-GPU")
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
    G.from_dask_cudf_edgelist(df, source="src", destination="dst")

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


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="FIXME: MG test fails on single-GPU")
@pytest.mark.cugraph_ops
def test_mg_uniform_neighbor_sample_ensure_no_duplicates(dask_client):
    # See issue #2760
    # This ensures that the starts are properly distributed

    df = cudf.DataFrame({"src": [6, 6, 6, 6], "dst": [7, 9, 10, 11]})
    df = df.astype("int32")

    dask_df = dask_cudf.from_cudf(df, npartitions=2)

    mg_G = cugraph.MultiGraph(directed=True)
    mg_G.from_dask_cudf_edgelist(
        dask_df, source="src", destination="dst", renumber=True
    )

    output_df = cugraph.dask.uniform_neighbor_sample(
        mg_G,
        cudf.Series([6]).astype("int32"),
        fanout_vals=[3],
        with_replacement=False,
    )

    assert len(output_df.compute()) == 3


@pytest.mark.mg
@pytest.mark.cugraph_ops
@pytest.mark.parametrize("return_offsets", [True, False])
def test_uniform_neighbor_sample_edge_properties(dask_client, return_offsets):
    n_workers = len(dask_client.scheduler_info()["workers"])
    if n_workers <= 1:
        pytest.skip("Test only valid for MG environments")
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

    G = cugraph.MultiGraph(directed=True)
    G.from_dask_cudf_edgelist(
        edgelist_df,
        source="src",
        destination="dst",
        edge_attr=["w", "eid", "etp"],
    )

    sampling_results = cugraph.dask.uniform_neighbor_sample(
        G,
        start_list=cudf.DataFrame(
            {
                "start": cudf.Series([0, 4], dtype="int64"),
                "batch": cudf.Series([0, 1], dtype="int32"),
            }
        ),
        fanout_vals=[-1, -1],
        with_replacement=False,
        with_edge_properties=True,
        with_batch_ids=True,
        keep_batches_together=True,
        min_batch_id=0,
        max_batch_id=1,
        return_offsets=return_offsets,
    )

    if return_offsets:
        sampling_results, sampling_offsets = sampling_results

        batches_found = {0: 0, 1: 0}
        for i in range(n_workers):
            dfp = sampling_results.get_partition(i).compute()
            if len(dfp) > 0:
                offsets_p = sampling_offsets.get_partition(i).compute()
                assert len(offsets_p) > 0

                if offsets_p.batch_id.iloc[0] == 1:
                    batches_found[1] += 1

                    assert offsets_p.batch_id.values_host.tolist() == [1]
                    assert offsets_p.offsets.values_host.tolist() == [0]

                    assert sorted(dfp.sources.values_host.tolist()) == (
                        [1, 1, 3, 3, 4, 4]
                    )
                    assert sorted(dfp.destinations.values_host.tolist()) == (
                        [1, 2, 2, 3, 3, 4]
                    )
                elif offsets_p.batch_id.iloc[0] == 0:
                    batches_found[0] += 1

                    assert offsets_p.batch_id.values_host.tolist() == [0]
                    assert offsets_p.offsets.values_host.tolist() == [0]

                    assert sorted(dfp.sources.values_host.tolist()) == (
                        [0, 0, 0, 1, 1, 2, 2, 2, 4, 4]
                    )
                    assert sorted(dfp.destinations.values_host.tolist()) == (
                        [1, 1, 1, 2, 2, 3, 3, 4, 4, 4]
                    )

    mdf = cudf.merge(
        sampling_results.compute(),
        edgelist_df.compute(),
        left_on="edge_id",
        right_on="eid",
    )
    assert (mdf.w == mdf.weight).all()
    assert (mdf.etp == mdf.edge_type).all()
    assert (mdf.src == mdf.sources).all()
    assert (mdf.dst == mdf.destinations).all()

    assert sorted(sampling_results.compute()["hop_id"].values_host.tolist()) == [
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]


@pytest.mark.mg
def test_uniform_neighbor_sample_edge_properties_self_loops(dask_client):
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
    )

    sampling_results = cugraph.dask.uniform_neighbor_sample(
        G,
        start_list=dask_cudf.from_cudf(
            cudf.DataFrame(
                {
                    "start": cudf.Series([0, 1, 2], dtype="int64"),
                    "batch": cudf.Series([1, 1, 1], dtype="int32"),
                }
            ),
            npartitions=2,
        ),
        fanout_vals=[2, 2],
        with_replacement=False,
        with_edge_properties=True,
        with_batch_ids=True,
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


@pytest.mark.mg
def test_uniform_neighbor_sample_hop_id_order():
    df = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "src": [0, 1, 2, 3, 3, 6],
                "dst": [2, 3, 4, 5, 6, 7],
            }
        ),
        npartitions=2,
    )

    G = cugraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(df, source="src", destination="dst")

    sampling_results = cugraph.dask.uniform_neighbor_sample(
        G,
        cudf.Series([0, 1], dtype="int64"),
        fanout_vals=[2, 2, 2],
        with_replacement=False,
        with_edge_properties=True,
    )

    for p in range(sampling_results.npartitions):
        sampling_results_p = sampling_results.get_partition(p).compute()
        assert (
            sorted(sampling_results_p.hop_id.values_host.tolist())
            == sampling_results_p.hop_id.values_host.tolist()
        )


@pytest.mark.mg
def test_uniform_neighbor_sample_hop_id_order_multi_batch():
    df = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "src": [0, 1, 2, 3, 3, 6],
                "dst": [2, 3, 4, 5, 6, 7],
            }
        ),
        npartitions=2,
    )

    G = cugraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(df, source="src", destination="dst")

    sampling_results = cugraph.dask.uniform_neighbor_sample(
        G,
        dask_cudf.from_cudf(
            cudf.DataFrame(
                {
                    "start": cudf.Series([0, 1], dtype="int64"),
                    "batch": cudf.Series([0, 1], dtype="int32"),
                }
            ),
            npartitions=2,
        ),
        fanout_vals=[2, 2, 2],
        with_replacement=False,
        with_edge_properties=True,
        with_batch_ids=True,
    )

    for p in range(sampling_results.npartitions):
        sampling_results_p = sampling_results.get_partition(p)
        if len(sampling_results_p) > 0:
            for b in range(2):
                sampling_results_pb = sampling_results_p[
                    sampling_results_p.batch_id == b
                ].compute()
                assert (
                    sorted(sampling_results_pb.hop_id.values_host.tolist())
                    == sampling_results_pb.hop_id.values_host.tolist()
                )


@pytest.mark.mg
@pytest.mark.parametrize("with_replacement", [True, False])
@pytest.mark.skipif(
    len(os.getenv("DASK_WORKER_DEVICES", "0").split(",")) < 2,
    reason="too few workers to test",
)
def test_uniform_neighbor_edge_properties_sample_small_start_list(
    dask_client, with_replacement
):
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
    )

    cugraph.dask.uniform_neighbor_sample(
        G,
        start_list=dask_cudf.from_cudf(
            cudf.Series(
                {
                    "start": cudf.Series([0]),
                    "batch": cudf.Series([10], dtype="int32"),
                }
            ),
            npartitions=1,
        ),
        fanout_vals=[10, 25],
        with_replacement=with_replacement,
        with_edge_properties=True,
        with_batch_ids=True,
    )


@pytest.mark.mg
def test_uniform_neighbor_sample_without_dask_inputs(dask_client):
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
    )

    sampling_results = cugraph.dask.uniform_neighbor_sample(
        G,
        start_list=cudf.DataFrame(
            {
                "start": cudf.Series([0, 1, 2]),
                "batch": cudf.Series([1, 1, 1], dtype="int32"),
            }
        ),
        fanout_vals=[2, 2],
        with_replacement=False,
        with_edge_properties=True,
        with_batch_ids=True,
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


@pytest.mark.mg
@pytest.mark.parametrize("dataset", datasets)
@pytest.mark.parametrize("input_df", [cudf.DataFrame, dask_cudf.DataFrame])
@pytest.mark.parametrize("max_batches", [2, 8, 16, 32])
def test_uniform_neighbor_sample_batched(dask_client, dataset, input_df, max_batches):
    num_workers = len(dask_client.scheduler_info()["workers"])

    df = dataset.get_edgelist()
    df["eid"] = cupy.arange(len(df), dtype=df["src"].dtype)
    df["etp"] = cupy.zeros_like(df["eid"].to_cupy())
    ddf = dask_cudf.from_cudf(df, npartitions=num_workers)

    G = cugraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(
        ddf,
        source="src",
        destination="dst",
        edge_attr=["wgt", "eid", "etp"],
        legacy_renum_only=True,
    )

    input_vertices = dask_cudf.concat([df.src, df.dst]).unique().compute()
    assert isinstance(input_vertices, cudf.Series)

    input_vertices.name = "start"
    input_vertices.index = cupy.random.permutation(len(input_vertices))
    input_vertices = input_vertices.to_frame().reset_index(drop=True)

    input_vertices["batch"] = cudf.Series(
        cupy.random.randint(0, max_batches, len(input_vertices)), dtype="int32"
    )

    if input_df == dask_cudf.DataFrame:
        input_vertices = dask_cudf.from_cudf(input_vertices, npartitions=num_workers)

    sampling_results = cugraph.dask.uniform_neighbor_sample(
        G,
        start_list=input_vertices,
        fanout_vals=[5, 5],
        with_replacement=False,
        with_edge_properties=True,
        with_batch_ids=True,
    )

    for batch_id in range(max_batches):
        output_starts_per_batch = (
            sampling_results[
                (sampling_results.batch_id == batch_id) & (sampling_results.hop_id == 0)
            ]
            .sources.nunique()
            .compute()
        )

        input_starts_per_batch = len(input_vertices[input_vertices.batch == batch_id])

        # Should be <= to account for starts without outgoing edges
        assert output_starts_per_batch <= input_starts_per_batch


@pytest.mark.mg
def test_uniform_neighbor_sample_exclude_sources_basic(dask_client):
    df = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "src": [0, 4, 1, 2, 3, 5, 4, 1, 0],
                "dst": [1, 1, 2, 4, 3, 1, 5, 0, 2],
                "eid": [9, 8, 7, 6, 5, 4, 3, 2, 1],
            }
        ),
        npartitions=1,
    )

    G = cugraph.MultiGraph(directed=True)
    G.from_dask_cudf_edgelist(df, source="src", destination="dst", edge_id="eid")

    sampling_results = (
        cugraph.dask.uniform_neighbor_sample(
            G,
            cudf.DataFrame(
                {
                    "seed": cudf.Series([0, 4, 1], dtype="int64"),
                    "batch": cudf.Series([1, 1, 1], dtype="int32"),
                }
            ),
            [2, 3, 3],
            with_replacement=False,
            with_edge_properties=True,
            with_batch_ids=True,
            random_state=62,
            prior_sources_behavior="exclude",
        )
        .sort_values(by="hop_id")
        .compute()
    )

    expected_hop_0 = [1, 2, 1, 5, 2, 0]
    assert sorted(
        sampling_results[sampling_results.hop_id == 0].destinations.values_host.tolist()
    ) == sorted(expected_hop_0)

    next_sources = set(
        sampling_results[sampling_results.hop_id > 0].sources.values_host.tolist()
    )
    for v in [0, 4, 1]:
        assert v not in next_sources

    next_sources = set(
        sampling_results[sampling_results.hop_id > 1].sources.values_host.tolist()
    )
    for v in sampling_results[
        sampling_results.hop_id == 1
    ].sources.values_host.tolist():
        assert v not in next_sources


@pytest.mark.mg
def test_uniform_neighbor_sample_exclude_sources_email_eu_core(dask_client):
    el = dask_cudf.from_cudf(email_Eu_core.get_edgelist(), npartitions=8)

    G = cugraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(el, source="src", destination="dst")

    seeds = G.select_random_vertices(62, int(0.001 * len(el)))

    sampling_results = cugraph.dask.uniform_neighbor_sample(
        G,
        seeds,
        [5, 4, 3, 2, 1],
        with_replacement=False,
        with_edge_properties=True,
        with_batch_ids=False,
        prior_sources_behavior="exclude",
    ).compute()

    for hop in range(5):
        current_sources = set(
            sampling_results[
                sampling_results.hop_id == hop
            ].sources.values_host.tolist()
        )
        future_sources = set(
            sampling_results[sampling_results.hop_id > hop].sources.values_host.tolist()
        )

        for s in current_sources:
            assert s not in future_sources


@pytest.mark.mg
def test_uniform_neighbor_sample_carry_over_sources_basic(dask_client):
    df = dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "src": [0, 4, 1, 2, 3, 5, 4, 1, 0, 6],
                "dst": [1, 1, 2, 4, 6, 1, 5, 0, 2, 2],
                "eid": [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            }
        ),
        npartitions=4,
    )

    G = cugraph.MultiGraph(directed=True)
    G.from_dask_cudf_edgelist(df, source="src", destination="dst", edge_id="eid")

    sampling_results = (
        cugraph.dask.uniform_neighbor_sample(
            G,
            cudf.DataFrame(
                {
                    "seed": cudf.Series([0, 4, 3], dtype="int64"),
                    "batch": cudf.Series([1, 1, 1], dtype="int32"),
                }
            ),
            [2, 3, 3],
            with_replacement=False,
            with_edge_properties=True,
            with_batch_ids=True,
            random_state=62,
            prior_sources_behavior="carryover",
        )
        .sort_values(by="hop_id")[["sources", "destinations", "hop_id"]]
        .compute()
    )

    assert (
        len(
            sampling_results[
                (sampling_results.hop_id == 2) & (sampling_results.sources == 6)
            ]
        )
        == 2
    )

    for hop in range(2):
        sources_current_hop = set(
            sampling_results[
                sampling_results.hop_id == hop
            ].sources.values_host.tolist()
        )
        sources_next_hop = set(
            sampling_results[
                sampling_results.hop_id == (hop + 1)
            ].sources.values_host.tolist()
        )

        for s in sources_current_hop:
            assert s in sources_next_hop


@pytest.mark.mg
def test_uniform_neighbor_sample_carry_over_sources_email_eu_core(dask_client):
    el = dask_cudf.from_cudf(email_Eu_core.get_edgelist(), npartitions=8)

    G = cugraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(el, source="src", destination="dst")

    seeds = G.select_random_vertices(62, int(0.001 * len(el)))

    sampling_results = cugraph.dask.uniform_neighbor_sample(
        G,
        seeds,
        [5, 4, 3, 2, 1],
        with_replacement=False,
        with_edge_properties=True,
        with_batch_ids=False,
        prior_sources_behavior="carryover",
    ).compute()

    for hop in range(4):
        sources_current_hop = set(
            sampling_results[
                sampling_results.hop_id == hop
            ].sources.values_host.tolist()
        )
        sources_next_hop = set(
            sampling_results[
                sampling_results.hop_id == (hop + 1)
            ].sources.values_host.tolist()
        )

        for s in sources_current_hop:
            assert s in sources_next_hop


@pytest.mark.mg
def test_uniform_neighbor_sample_deduplicate_sources_email_eu_core(dask_client):
    el = dask_cudf.from_cudf(email_Eu_core.get_edgelist(), npartitions=8)

    G = cugraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(el, source="src", destination="dst")

    seeds = G.select_random_vertices(62, int(0.001 * len(el)))

    sampling_results = cugraph.dask.uniform_neighbor_sample(
        G,
        seeds,
        [5, 4, 3, 2, 1],
        with_replacement=False,
        with_edge_properties=True,
        with_batch_ids=False,
        deduplicate_sources=True,
    ).compute()

    for hop in range(5):
        counts_current_hop = (
            sampling_results[sampling_results.hop_id == hop]
            .sources.value_counts()
            .values_host.tolist()
        )
        for c in counts_current_hop:
            assert c <= 5 - hop


@pytest.mark.mg
@pytest.mark.parametrize("hops", [[5], [5, 5], [5, 5, 5]])
@pytest.mark.tags("runme")
def test_uniform_neighbor_sample_renumber(dask_client, hops):
    # FIXME This test is not very good because there is a lot of
    # non-deterministic behavior that still exists despite passing
    # a random seed. Right now, there are tests in cuGraph-DGL and
    # cuGraph-PyG that provide better coverage, but a better test
    # should eventually be written to augment or replace this one.

    el = dask_cudf.from_cudf(email_Eu_core.get_edgelist(), npartitions=4)

    G = cugraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(el, source="src", destination="dst")

    seeds = G.select_random_vertices(62, int(0.0001 * len(el)))

    sampling_results_renumbered, renumber_map = cugraph.dask.uniform_neighbor_sample(
        G,
        seeds,
        hops,
        with_replacement=False,
        with_edge_properties=True,
        with_batch_ids=False,
        deduplicate_sources=True,
        renumber=True,
        random_state=62,
        keep_batches_together=True,
        min_batch_id=0,
        max_batch_id=0,
    )
    sampling_results_renumbered = sampling_results_renumbered.compute()
    renumber_map = renumber_map.compute()

    sources_hop_0 = sampling_results_renumbered[
        sampling_results_renumbered.hop_id == 0
    ].sources

    assert (renumber_map.batch_id == 0).all()
    assert (
        renumber_map.map.nunique()
        == cudf.concat(
            [sources_hop_0, sampling_results_renumbered.destinations]
        ).nunique()
    )


# =============================================================================
# Benchmarks
# =============================================================================


@pytest.mark.mg
@pytest.mark.slow
@pytest.mark.parametrize("n_samples", [1_000, 5_000, 10_000])
def bench_uniform_neighbor_sample_email_eu_core(gpubenchmark, dask_client, n_samples):
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
    )
    # Partition the dataframe to add in chunks
    srcs = dg.input_df["src"]
    start_list = srcs[:n_samples].compute()

    def func():
        _ = cugraph.dask.uniform_neighbor_sample(dg, start_list, [10])
        del _

    gpubenchmark(func)
