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

import pytest
import cudf
from pylibcugraph.testing.utils import gen_fixture_params_product

import cugraph
from cugraph import uniform_neighbor_sample
from cugraph.experimental.datasets import DATASETS_UNDIRECTED, email_Eu_core, small_tree


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
    print("data path:", input_data_path)
    directed = parameters["directed"]

    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", indices_type],
    )

    G = cugraph.Graph(directed=directed)
    G.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="value")

    parameters["Graph"] = G

    # sample k vertices from the cuGraph graph
    k = random.randint(1, 3)
    srcs = G.view_edge_list()["src"]
    dsts = G.view_edge_list()["dst"]

    vertices = cudf.concat([srcs, dsts]).drop_duplicates()

    start_list = vertices.sample(k).astype("int32")

    # Generate a random fanout_vals list of length random(1, k)
    fanout_vals = [random.randint(1, k) for _ in range(random.randint(1, k))]

    # These prints are for debugging purposes since the vertices and
    # the fanout_vals are randomly sampled/chosen
    print("\nstart_list: \n", start_list)
    print("fanout_vals: ", fanout_vals)

    parameters["start_list"] = start_list
    parameters["fanout_vals"] = fanout_vals

    return parameters


@pytest.fixture(scope="module")
def simple_unweighted_input_expected_output(request):
    """
    Fixture for providing the input for a uniform_neighbor_sample test using a
    small/simple unweighted graph and the corresponding expected output.
    """
    test_data = {}

    df = cudf.DataFrame(
        {"src": [0, 1, 2, 2, 0, 1, 4, 4], "dst": [3, 2, 1, 4, 1, 3, 1, 2]}
    )

    G = cugraph.Graph()
    G.from_cudf_edgelist(df, source="src", destination="dst")
    test_data["Graph"] = G
    test_data["start_list"] = cudf.Series([0], dtype="int32")
    test_data["fanout_vals"] = [-1]
    test_data["with_replacement"] = True

    test_data["expected_src"] = [0, 0]
    test_data["expected_dst"] = [3, 1]

    return test_data


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.sg
@pytest.mark.cugraph_ops
def test_uniform_neighbor_sample_simple(input_combo):

    G = input_combo["Graph"]

    #
    # Make sure the old C++ renumbering was skipped because:
    #    1) Pylibcugraph already does renumbering
    #    2) Uniform neighborhood sampling allows int32 weights
    #       which are not supported by the C++ renumbering
    # This should be 'True' only for string vertices and multi columns vertices
    #

    assert G.renumbered is False
    # Retrieve the input dataframe.
    # FIXME: in simpleGraph and simpleDistributedGraph, G.edgelist.edgelist_df
    # should be 'None' if the datasets was never renumbered
    input_df = G.edgelist.edgelist_df

    result_nbr = uniform_neighbor_sample(
        G,
        input_combo["start_list"],
        input_combo["fanout_vals"],
        input_combo["with_replacement"],
    )

    print(input_df)
    print(result_nbr)

    # multi edges are dropped to easily verify that each edge in the
    # results is present in the input dataframe
    result_nbr = result_nbr.drop_duplicates()

    # FIXME: The indices are not included in the comparison because garbage
    # value are intermittently retuned. This observation is observed
    # when passing float weights
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
        cudf.concat([result_nbr["sources"], result_nbr["destinations"]])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    sampled_vertex_result = sampled_vertex_result.to_pandas()
    start_list = input_combo["start_list"].to_pandas()

    if not set(start_list).issubset(set(sampled_vertex_result)):
        missing_vertex = set(start_list) - set(sampled_vertex_result)
        missing_vertex = list(missing_vertex)
        # compute the out-degree of the missing vertices
        out_degree = G.out_degree(missing_vertex)
        out_degree = out_degree[out_degree.degree != 0]
        # If the missing vertices have outgoing edges, return an error
        if len(out_degree) != 0:
            missing_vertex = out_degree["vertex"].to_pandas().to_list()
            raise Exception(
                f"vertex {missing_vertex} is missing from "
                f"uniform neighbor sampling results"
            )


@pytest.mark.sg
@pytest.mark.cugraph_ops
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_uniform_neighbor_sample_tree(directed):

    input_data_path = small_tree.get_path()

    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    G = cugraph.Graph(directed=directed)
    G.from_cudf_edgelist(df, "src", "dst", "value")

    #
    # Make sure the old C++ renumbering was skipped because:
    #    1) Pylibcugraph already does renumbering
    #    2) Uniform neighborhood sampling allows int32 weights
    #       which are not supported by the C++ renumbering
    # This should be 'True' only for string vertices and multi columns vertices
    #

    assert G.renumbered is False

    # Retrieve the input dataframe.
    # input_df != df if 'directed = False' because df will be symmetrized
    # internally.
    input_df = G.edgelist.edgelist_df

    # TODO: Incomplete, include more testing for tree graph as well as
    # for larger graphs
    start_list = cudf.Series([0, 0], dtype="int32")
    fanout_vals = [4, 1, 3]
    with_replacement = True
    result_nbr = uniform_neighbor_sample(G, start_list, fanout_vals, with_replacement)

    result_nbr = result_nbr.drop_duplicates()

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
        cudf.concat([result_nbr["sources"], result_nbr["destinations"]])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    assert set(start_list.to_pandas()).issubset(set(result_nbr_vertices.to_pandas()))


@pytest.mark.sg
@pytest.mark.cugraph_ops
def test_uniform_neighbor_sample_unweighted(simple_unweighted_input_expected_output):
    test_data = simple_unweighted_input_expected_output

    sampling_results = uniform_neighbor_sample(
        test_data["Graph"],
        test_data["start_list"].astype("int64"),
        test_data["fanout_vals"],
        test_data["with_replacement"],
    )

    actual_src = sampling_results.sources
    actual_src = actual_src.to_arrow().to_pylist()
    assert sorted(actual_src) == sorted(test_data["expected_src"])

    actual_dst = sampling_results.destinations
    actual_dst = actual_dst.to_arrow().to_pylist()
    assert sorted(actual_dst) == sorted(test_data["expected_dst"])


@pytest.mark.sg
@pytest.mark.cugraph_ops
@pytest.mark.parametrize("return_offsets", [True, False])
def test_uniform_neighbor_sample_edge_properties(return_offsets):
    edgelist_df = cudf.DataFrame(
        {
            "src": cudf.Series([0, 1, 2, 3, 4, 3, 4, 2, 0, 1, 0, 2], dtype="int32"),
            "dst": cudf.Series([1, 2, 4, 2, 3, 4, 1, 1, 2, 3, 4, 4], dtype="int32"),
            "eid": cudf.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype="int32"),
            "etp": cudf.Series([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0], dtype="int32"),
            "w": [0.0, 0.1, 0.2, 3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10, 0.11],
        }
    )

    start_df = cudf.DataFrame(
        {
            "seed": cudf.Series([0, 4], dtype="int32"),
            "batch": cudf.Series([0, 1], dtype="int32"),
        }
    )

    G = cugraph.MultiGraph(directed=True)
    G.from_cudf_edgelist(
        edgelist_df,
        source="src",
        destination="dst",
        edge_attr=["w", "eid", "etp"],
    )

    sampling_results = uniform_neighbor_sample(
        G,
        start_list=start_df,
        fanout_vals=[2, 2],
        with_replacement=False,
        with_edge_properties=True,
        with_batch_ids=True,
        return_offsets=return_offsets,
    )
    if return_offsets:
        sampling_results, sampling_offsets = sampling_results

    edgelist_df.set_index("eid")
    assert (
        edgelist_df.loc[sampling_results.edge_id]["w"].values_host.tolist()
        == sampling_results["weight"].values_host.tolist()
    )
    assert (
        edgelist_df.loc[sampling_results.edge_id]["etp"].values_host.tolist()
        == sampling_results["edge_type"].values_host.tolist()
    )
    assert (
        edgelist_df.loc[sampling_results.edge_id]["src"].values_host.tolist()
        == sampling_results["sources"].values_host.tolist()
    )
    assert (
        edgelist_df.loc[sampling_results.edge_id]["dst"].values_host.tolist()
        == sampling_results["destinations"].values_host.tolist()
    )

    assert sampling_results["hop_id"].values_host.tolist() == ([0, 0, 1, 1, 1, 1] * 2)

    if return_offsets:
        assert sampling_offsets["batch_id"].values_host.tolist() == [0, 1]
        assert sampling_offsets["offsets"].values_host.tolist() == [0, 6]
    else:
        assert sampling_results["batch_id"].values_host.tolist() == ([0] * 6 + [1] * 6)


@pytest.mark.sg
def test_uniform_neighbor_sample_edge_properties_self_loops():
    df = cudf.DataFrame(
        {
            "src": [0, 1, 2],
            "dst": [0, 1, 2],
            "eid": [2, 4, 6],
            "etp": cudf.Series([1, 1, 2], dtype="int32"),
            "w": [0.0, 0.1, 0.2],
        }
    )

    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(
        df,
        source="src",
        destination="dst",
        edge_attr=["w", "eid", "etp"],
    )

    sampling_results = cugraph.uniform_neighbor_sample(
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
        random_state=80,
    )

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


@pytest.mark.sg
def test_uniform_neighbor_sample_hop_id_order():
    df = cudf.DataFrame(
        {
            "src": [0, 1, 2, 3, 3, 6],
            "dst": [2, 3, 4, 5, 6, 7],
        }
    )

    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(df, source="src", destination="dst")

    sampling_results = cugraph.uniform_neighbor_sample(
        G,
        cudf.Series([0, 1], dtype="int64"),
        fanout_vals=[2, 2, 2],
        with_replacement=False,
        with_edge_properties=True,
    )

    assert (
        sorted(sampling_results.hop_id.values_host.tolist())
        == sampling_results.hop_id.values_host.tolist()
    )


@pytest.mark.sg
def test_uniform_neighbor_sample_hop_id_order_multi_batch():
    df = cudf.DataFrame(
        {
            "src": [0, 1, 2, 3, 3, 6],
            "dst": [2, 3, 4, 5, 6, 7],
        }
    )

    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(df, source="src", destination="dst")

    sampling_results = cugraph.uniform_neighbor_sample(
        G,
        start_list=cudf.DataFrame(
            {
                "start": cudf.Series([0, 1], dtype="int64"),
                "batch": cudf.Series([0, 1], dtype="int32"),
            }
        ),
        fanout_vals=[2, 2, 2],
        with_replacement=False,
        with_edge_properties=True,
        with_batch_ids=True,
    )

    for b in range(2):
        assert (
            sorted(
                sampling_results[
                    sampling_results.batch_id == b
                ].hop_id.values_host.tolist()
            )
            == sampling_results[
                sampling_results.batch_id == b
            ].hop_id.values_host.tolist()
        )


@pytest.mark.sg
def test_uniform_neighbor_sample_empty_start_list():
    df = cudf.DataFrame(
        {
            "src": [0, 1, 2],
            "dst": [0, 1, 2],
            "eid": [2, 4, 6],
            "etp": cudf.Series([1, 1, 2], dtype="int32"),
            "w": [0.0, 0.1, 0.2],
        }
    )

    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(
        df,
        source="src",
        destination="dst",
        edge_attr=["w", "eid", "etp"],
    )

    sampling_results = cugraph.uniform_neighbor_sample(
        G,
        start_list=cudf.DataFrame(
            {
                "start_list": cudf.Series(dtype="int64"),
                "batch_id_list": cudf.Series(dtype="int32"),
            }
        ),
        fanout_vals=[2, 2],
        with_replacement=False,
        with_edge_properties=True,
        with_batch_ids=True,
        random_state=32,
    )

    assert sampling_results.empty


@pytest.mark.sg
def test_uniform_neighbor_sample_exclude_sources_basic():
    df = cudf.DataFrame(
        {
            "src": [0, 4, 1, 2, 3, 5, 4, 1, 0],
            "dst": [1, 1, 2, 4, 3, 1, 5, 0, 2],
            "eid": [9, 8, 7, 6, 5, 4, 3, 2, 1],
        }
    )

    G = cugraph.MultiGraph(directed=True)
    G.from_cudf_edgelist(df, source="src", destination="dst", edge_id="eid")

    sampling_results = cugraph.uniform_neighbor_sample(
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
    ).sort_values(by="hop_id")

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


@pytest.mark.sg
def test_uniform_neighbor_sample_exclude_sources_email_eu_core():
    el = email_Eu_core.get_edgelist()

    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(el, source="src", destination="dst")

    seeds = G.select_random_vertices(62, int(0.001 * len(el)))

    sampling_results = cugraph.uniform_neighbor_sample(
        G,
        seeds,
        [5, 4, 3, 2, 1],
        with_replacement=False,
        with_edge_properties=True,
        with_batch_ids=False,
        prior_sources_behavior="exclude",
    )

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


@pytest.mark.sg
def test_uniform_neighbor_sample_carry_over_sources_basic():
    df = cudf.DataFrame(
        {
            "src": [0, 4, 1, 2, 3, 5, 4, 1, 0, 6],
            "dst": [1, 1, 2, 4, 6, 1, 5, 0, 2, 2],
            "eid": [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        }
    )

    G = cugraph.MultiGraph(directed=True)
    G.from_cudf_edgelist(df, source="src", destination="dst", edge_id="eid")

    sampling_results = cugraph.uniform_neighbor_sample(
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
    ).sort_values(by="hop_id")[["sources", "destinations", "hop_id"]]

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


@pytest.mark.sg
def test_uniform_neighbor_sample_carry_over_sources_email_eu_core():
    el = email_Eu_core.get_edgelist()

    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(el, source="src", destination="dst")

    seeds = G.select_random_vertices(62, int(0.001 * len(el)))

    sampling_results = cugraph.uniform_neighbor_sample(
        G,
        seeds,
        [5, 4, 3, 2, 1],
        with_replacement=False,
        with_edge_properties=True,
        with_batch_ids=False,
        prior_sources_behavior="carryover",
    )

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


@pytest.mark.sg
def test_uniform_neighbor_sample_deduplicate_sources_email_eu_core():
    el = email_Eu_core.get_edgelist()

    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(el, source="src", destination="dst")

    seeds = G.select_random_vertices(62, int(0.001 * len(el)))

    sampling_results = cugraph.uniform_neighbor_sample(
        G,
        seeds,
        [5, 4, 3, 2, 1],
        with_replacement=False,
        with_edge_properties=True,
        with_batch_ids=False,
        deduplicate_sources=True,
    )

    for hop in range(5):
        counts_current_hop = (
            sampling_results[sampling_results.hop_id == hop]
            .sources.value_counts()
            .values_host.tolist()
        )
        for c in counts_current_hop:
            assert c <= 5 - hop


@pytest.mark.sg
@pytest.mark.parametrize("hops", [[5], [5, 5], [5, 5, 5]])
def test_uniform_neighbor_sample_renumber(hops):
    el = email_Eu_core.get_edgelist()

    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(el, source="src", destination="dst")

    seeds = G.select_random_vertices(62, int(0.0001 * len(el)))

    sampling_results_unrenumbered = cugraph.uniform_neighbor_sample(
        G,
        seeds,
        hops,
        with_replacement=False,
        with_edge_properties=True,
        with_batch_ids=False,
        deduplicate_sources=True,
        renumber=False,
        random_state=62,
    )

    sampling_results_renumbered, renumber_map = cugraph.uniform_neighbor_sample(
        G,
        seeds,
        hops,
        with_replacement=False,
        with_edge_properties=True,
        with_batch_ids=False,
        deduplicate_sources=True,
        renumber=True,
        random_state=62,
    )

    sources_hop_0 = sampling_results_unrenumbered[
        sampling_results_unrenumbered.hop_id == 0
    ].sources
    for hop in range(len(hops)):
        destinations_hop = sampling_results_unrenumbered[
            sampling_results_unrenumbered.hop_id <= hop
        ].destinations
        expected_renumber_map = cudf.concat([sources_hop_0, destinations_hop]).unique()

        assert sorted(expected_renumber_map.values_host.tolist()) == sorted(
            renumber_map.map[0 : len(expected_renumber_map)].values_host.tolist()
        )
    assert (renumber_map.batch_id == 0).all()


@pytest.mark.sg
@pytest.mark.skip(reason="needs to be written!")
def test_multi_client_sampling():
    # See gist for example test to write
    # https://gist.github.com/VibhuJawa/1b705427f7a0c5a2a4f58e0a3e71ef21
    raise NotImplementedError
