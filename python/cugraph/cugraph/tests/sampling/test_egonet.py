# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest
import networkx as nx

import cudf
import cugraph
from cugraph.testing import utils, DEFAULT_DATASETS


print("Networkx version : {} ".format(nx.__version__))

SEEDS = [0, 5, 13]
RADIUS = [1, 2, 3]


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("radius", RADIUS)
def test_ego_graph(graph_file, seed, radius):
    gc.collect()

    # Read dataset
    dataset_path = graph_file.get_path()

    # cugraph
    G = utils.generate_cugraph_graph_from_file(dataset_path, edgevals=True)
    assert G is not None

    ego_cugraph = cugraph.ego_graph(G, seed, radius=radius)
    assert ego_cugraph is not None

    # networkx
    df = utils.read_csv_for_nx(dataset_path, read_weights_in_sp=True)
    Gnx = nx.from_pandas_edgelist(
        df, create_using=nx.Graph(), source="0", target="1", edge_attr="weight"
    )
    ego_nx = nx.ego_graph(Gnx, seed, radius=radius)

    # compare the two graphs
    cu_edges = ego_cugraph.view_edge_list().to_pandas()
    ego_cugraph_nx = nx.from_pandas_edgelist(cu_edges, source="src", target="dst")
    assert nx.is_isomorphic(ego_nx, ego_cugraph_nx)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("radius", RADIUS)
def test_multi_column_ego_graph(graph_file, seed, radius):
    gc.collect()

    dataset_path = graph_file.get_path()
    df = utils.read_csv_file(dataset_path, read_weights_in_sp=True)
    df.rename(columns={"0": "src_0", "1": "dst_0"}, inplace=True)
    df["src_1"] = df["src_0"] + 1000
    df["dst_1"] = df["dst_0"] + 1000

    G1 = cugraph.Graph()
    G1.from_cudf_edgelist(
        df, source=["src_0", "src_1"], destination=["dst_0", "dst_1"], edge_attr="2"
    )

    seed_df = cudf.DataFrame()
    seed_df["v_0"] = [seed]
    seed_df["v_1"] = [seed + 1000]

    ego_cugraph_res = cugraph.ego_graph(G1, seed_df, radius=radius)

    G2 = cugraph.Graph()
    G2.from_cudf_edgelist(df, source="src_0", destination="dst_0", edge_attr="2")
    ego_cugraph_exp = cugraph.ego_graph(G2, seed, radius=radius)

    # FIXME: Replace with multi-column view_edge_list()
    edgelist_df = ego_cugraph_res.edgelist.edgelist_df
    edgelist_df_res = ego_cugraph_res.unrenumber(edgelist_df, "src")
    edgelist_df_res = ego_cugraph_res.unrenumber(edgelist_df_res, "dst")
    for i in range(len(edgelist_df_res)):
        assert ego_cugraph_exp.has_edge(
            edgelist_df_res["0_src"].iloc[i], edgelist_df_res["0_dst"].iloc[i]
        )


def _canonical_edges(df, weighted=False):
    """Return orientation-independent edge tuples for test comparisons."""
    pdf = df.to_pandas()
    result = set()
    for row in pdf.itertuples(index=False):
        src = getattr(row, "src")
        dst = getattr(row, "dst")
        edge = (min(src, dst), max(src, dst))
        if weighted:
            edge = (*edge, getattr(row, "weight"))
        result.add(edge)
    return result


@pytest.fixture
def multi_seed_graph():
    df = cudf.DataFrame(
        {
            "src": [0, 1, 1, 3, 10, 11],
            "dst": [1, 2, 3, 4, 11, 12],
        }
    )
    graph = cugraph.Graph()
    graph.from_cudf_edgelist(df, source="src", destination="dst")
    return graph


@pytest.fixture
def weighted_multi_seed_graph():
    df = cudf.DataFrame(
        {
            "src": [0, 1, 1, 10, 11],
            "dst": [1, 2, 3, 11, 12],
            "weight": [0.5, 1.5, 2.5, 3.5, 4.5],
        }
    )
    graph = cugraph.Graph()
    graph.from_cudf_edgelist(
        df, source="src", destination="dst", edge_attr="weight"
    )
    return graph


@pytest.mark.sg
def test_multiple_seeds_require_offsets(multi_seed_graph):
    with pytest.raises(ValueError, match="return_offsets=True"):
        cugraph.ego_graph(multi_seed_graph, [0, 10], radius=1)


@pytest.mark.sg
def test_multiple_seed_offsets_partition_output(multi_seed_graph):
    edges, offsets = cugraph.ego_graph(
        multi_seed_graph, [0, 10], radius=1, return_offsets=True
    )

    assert len(offsets) == 3
    assert offsets.iloc[0] == 0
    assert offsets.iloc[-1] == len(edges)
    assert offsets.is_monotonic_increasing

    first = edges.iloc[offsets.iloc[0] : offsets.iloc[1]]
    second = edges.iloc[offsets.iloc[1] : offsets.iloc[2]]

    expected_first = cugraph.ego_graph(
        multi_seed_graph, 0, radius=1
    ).view_edge_list()
    expected_second = cugraph.ego_graph(
        multi_seed_graph, 10, radius=1
    ).view_edge_list()

    assert _canonical_edges(first) == _canonical_edges(expected_first)
    assert _canonical_edges(second) == _canonical_edges(expected_second)


@pytest.mark.sg
def test_multiple_seed_offsets_preserve_weights(weighted_multi_seed_graph):
    edges, offsets = cugraph.ego_graph(
        weighted_multi_seed_graph, [0, 10], radius=1, return_offsets=True
    )

    assert "weight" in edges.columns
    assert len(offsets) == 3
    assert offsets.iloc[-1] == len(edges)

    first = edges.iloc[offsets.iloc[0] : offsets.iloc[1]]
    second = edges.iloc[offsets.iloc[1] : offsets.iloc[2]]

    expected_first = cugraph.ego_graph(
        weighted_multi_seed_graph, 0, radius=1
    ).view_edge_list()
    expected_second = cugraph.ego_graph(
        weighted_multi_seed_graph, 10, radius=1
    ).view_edge_list()

    assert _canonical_edges(first, weighted=True) == _canonical_edges(
        expected_first, weighted=True
    )
    assert _canonical_edges(second, weighted=True) == _canonical_edges(
        expected_second, weighted=True
    )


@pytest.mark.sg
def test_multiple_seed_offsets_use_external_vertex_ids():
    df = cudf.DataFrame(
        {
            "src": [1001, 1002, 1002, 5001, 5002],
            "dst": [1002, 1003, 1004, 5002, 5003],
        }
    )
    graph = cugraph.Graph()
    graph.from_cudf_edgelist(df, source="src", destination="dst", renumber=True)

    edges, offsets = cugraph.ego_graph(
        graph, [1001, 5001], radius=1, return_offsets=True
    )

    external_vertices = {1001, 1002, 1003, 1004, 5001, 5002, 5003}
    assert set(edges["src"].to_pandas()).issubset(external_vertices)
    assert set(edges["dst"].to_pandas()).issubset(external_vertices)

    first = edges.iloc[offsets.iloc[0] : offsets.iloc[1]]
    second = edges.iloc[offsets.iloc[1] : offsets.iloc[2]]
    expected_first = cugraph.ego_graph(graph, 1001, radius=1).view_edge_list()
    expected_second = cugraph.ego_graph(graph, 5001, radius=1).view_edge_list()

    assert _canonical_edges(first) == _canonical_edges(expected_first)
    assert _canonical_edges(second) == _canonical_edges(expected_second)


@pytest.mark.sg
def test_single_seed_default_still_returns_graph(multi_seed_graph):
    result = cugraph.ego_graph(multi_seed_graph, 0, radius=1)
    assert isinstance(result, cugraph.Graph)


@pytest.mark.sg
def test_empty_seed_list_rejected(multi_seed_graph):
    with pytest.raises(ValueError, match="at least one seed"):
        cugraph.ego_graph(multi_seed_graph, [], radius=1)
