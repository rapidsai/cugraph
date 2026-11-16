# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import time

import numpy as np
import pytest

import cugraph
import cudf
from cudf.testing.testing import assert_frame_equal, assert_series_equal
from cugraph.testing import UNDIRECTED_DATASETS
from cugraph.datasets import karate, dolphins, polbooks, karate_asymmetric


# =============================================================================
# Test data
# =============================================================================

# Leiden is deterministic: for a given graph (vertex numbering), resolution,
# n_iterations and random_state, the partition and the modularity are bitwise
# reproducible on every GPU, so these results are exact.
# fmt: off
_test_data = {
    "data_1": {
        "graph": {
            "src_or_offset_array": [0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5],
            "dst_or_index_array": [1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4],
            "weight": [0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2, 0.1, 2.1, 1.1, 5.1,
                       3.1, 4.1, 7.2, 3.2],
        },
        "resolution": 1.0,
        "input_type": "COO",
        "expected_output": {
            "partition": [0, 0, 0, 1, 1, 1],
            "modularity_score": 0.21596893567080255,
        },
    },
    "data_2": {
        "graph": {
            "src_or_offset_array": [0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66,
                                    67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93,
                                    98, 101, 104, 106, 110, 113, 117, 121, 127, 139,
                                    156],

            "dst_or_index_array": [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21,
                                   31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8,
                                   9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0,
                                   6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32,
                                   33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33,
                                   32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1,
                                   32, 33, 25, 27, 29, 32, 33, 25, 27, 31, 23, 24, 31,
                                   29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1,
                                   8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18,
                                   20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 18, 19,
                                   20, 22, 23, 26, 27, 28, 29, 30, 31, 32],
            "weight": [1.0] * 156,
        },
        "resolution": 1.0,
        "input_type": "CSR",
        "expected_output": {
            "partition": [1, 1, 1, 1, 3, 3, 3, 1, 0, 0, 3, 1, 1, 1, 0, 0, 3, 1, 0, 1,
                          0, 1, 0, 2, 2, 2, 0, 2, 2, 0, 0, 2, 0, 0],
            "modularity_score": 0.41978961209730437,
        },
    },
}

# cugraph.leiden(G, resolution=1.0, random_state=0) with the default
# n_iterations=2 on the datasets with their file vertex ids (renumber=False;
# vertex ids enter the hashes). The algorithm makes every decision in exact
# integer arithmetic and sums the modularity in a fixed order, so these values
# are exact on every GPU: rerunning the call reproduces them, and a mismatch
# means that the algorithm changed. The modularity is given as float.hex().
_reference_results = {
    "karate": {
        "num_clusters": 4,
        "modularity_hex": "0x1.addd53fca2404p-2",
        "partition": [1, 1, 1, 1, 3, 3, 3, 1, 0, 0, 3, 1, 1, 1, 0, 0, 3, 1, 0, 1,
                      0, 1, 0, 2, 2, 2, 0, 2, 2, 0, 0, 2, 0, 0],
    },
    "dolphins": {
        "num_clusters": 5,
        "modularity_hex": "0x1.0e9a19a8e5039p-1",
        "partition": [3, 0, 3, 4, 2, 0, 0, 0, 4, 0, 3, 2, 1, 0, 1, 2, 1, 0, 2, 0,
                      3, 2, 0, 2, 2, 0, 0, 0, 3, 2, 3, 0, 0, 1, 1, 2, 4, 1, 1, 4,
                      1, 0, 3, 1, 3, 2, 1, 3, 0, 1, 1, 2, 1, 1, 0, 2, 0, 0, 1, 4,
                      0, 1],
    },
    "polbooks": {
        "num_clusters": 5,
        "modularity_hex": "0x1.0df1f46f4d973p-1",
        "partition": [3, 3, 3, 0, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 2, 2, 2, 0, 0, 0, 0, 4, 2, 1,
                      1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 2, 2],
    },
}
# fmt: on

REFERENCE_DATASETS = [karate, dolphins, polbooks]


# =============================================================================
# Pytest fixtures
# =============================================================================
@pytest.fixture(
    scope="module",
    params=[pytest.param(value, id=key) for (key, value) in _test_data.items()],
)
def input_and_expected_output(request):
    d = request.param.copy()

    input_graph_data = d.pop("graph")
    input_type = d.pop("input_type")
    src_or_offset_array = cudf.Series(
        input_graph_data["src_or_offset_array"], dtype="int32"
    )
    dst_or_index_array = cudf.Series(
        input_graph_data["dst_or_index_array"], dtype="int32"
    )
    weight = cudf.Series(input_graph_data["weight"], dtype="float32")

    resolution = d.pop("resolution")
    output = d

    G = cugraph.Graph()

    if input_type == "COO":
        # Create graph from an edgelist
        df = cudf.DataFrame()
        df["src"] = src_or_offset_array
        df["dst"] = dst_or_index_array
        df["weight"] = cudf.Series(weight, dtype="float32")
        G.from_cudf_edgelist(
            df,
            source="src",
            destination="dst",
            edge_attr="weight",
            store_transposed=False,
        )

    elif input_type == "CSR":
        # Create graph from csr
        offsets = src_or_offset_array
        indices = dst_or_index_array
        G.from_cudf_adjlist(offsets, indices, weight, renumber=False)

    parts, mod = cugraph.leiden(G, resolution=resolution, random_state=0)

    parts = parts.sort_values("vertex").reset_index(drop=True)

    output["input_type"] = input_type
    output["result_output"] = {"partition": parts["partition"], "modularity_score": mod}

    return output


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# =============================================================================
# Helper functions
# =============================================================================
def cugraph_leiden(G, **kwargs):
    # cugraph Leiden Call
    t1 = time.time()
    parts, mod = cugraph.leiden(G, **kwargs)
    t2 = time.time() - t1
    print("Cugraph Leiden Time : " + str(t2))

    return parts, mod


def cugraph_louvain(G):
    # cugraph Louvain Call
    t1 = time.time()
    parts, mod = cugraph.louvain(G)
    t2 = time.time() - t1
    print("Cugraph Louvain Time : " + str(t2))

    return parts, mod


def get_renumber_false_graph(dataset, dtypes=None):
    """
    Returns the graph of dataset with its file vertex ids (renumber=False),
    optionally casting the src, dst and wgt columns to dtypes.
    """
    edgelist = dataset.get_edgelist(download=True)
    if dtypes is not None:
        edgelist = edgelist.astype(dtypes)
    G = cugraph.Graph()
    G.from_cudf_edgelist(edgelist, "src", "dst", "wgt", renumber=False)
    return G


def sorted_partition(parts):
    return parts.sort_values("vertex").reset_index(drop=True)


def get_edges(G):
    """
    Returns the edges of the undirected graph G (each edge once, external
    vertex ids) as numpy arrays (src, dst, weight); weight 1 if unweighted.
    """
    edgelist = G.view_edge_list().to_pandas()
    src = edgelist.iloc[:, 0].to_numpy()
    dst = edgelist.iloc[:, 1].to_numpy()
    if edgelist.shape[1] > 2:
        wgt = edgelist.iloc[:, 2].to_numpy(dtype=np.float64)
    else:
        wgt = np.ones(len(src), dtype=np.float64)
    return src, dst, wgt


def host_modularity(G, parts, resolution=1.0):
    """
    Modularity Q(resolution) of the partition, computed on the host in fp64
    from the edge list of G, self-loops excluded:
    Q = sum_c [ L_c / (2m) - resolution * (K_c / (2m))**2 ]
    """
    src, dst, wgt = get_edges(G)
    keep = src != dst
    src, dst, wgt = src[keep], dst[keep], wgt[keep]
    partition = parts.to_pandas().set_index("vertex")["partition"]
    p_src = partition.loc[src].to_numpy()
    p_dst = partition.loc[dst].to_numpy()
    num_clusters = int(partition.max()) + 1

    two_m = 2.0 * wgt.sum()
    internal = 2.0 * wgt[p_src == p_dst].sum()
    cluster_degree = np.zeros(num_clusters)
    np.add.at(cluster_degree, p_src, wgt)
    np.add.at(cluster_degree, p_dst, wgt)
    return internal / two_m - resolution * np.sum((cluster_degree / two_m) ** 2)


def assert_valid_partition(G, parts):
    """
    Checks the partition contract of cugraph.leiden: every vertex has a
    partition, the partition ids are 0, ..., k-1 ordered by decreasing size,
    and every partition is connected.
    """
    assert len(parts) == G.number_of_vertices()
    assert parts["vertex"].is_unique

    # partition ids are numbered consecutively, by decreasing size
    sizes = parts["partition"].value_counts().sort_index()
    num_clusters = len(sizes)
    assert_series_equal(
        sizes.index.to_series().reset_index(drop=True),
        cudf.Series(np.arange(num_clusters)),
        check_dtype=False,
        check_names=False,
    )
    assert np.all(np.diff(sizes.to_numpy()) <= 0)

    # every partition is connected: the edges inside the partitions form
    # exactly one connected component per partition
    src, dst, _ = get_edges(G)
    partition = parts.to_pandas().set_index("vertex")["partition"]
    intra = partition.loc[src].to_numpy() == partition.loc[dst].to_numpy()
    intra_edges = cudf.DataFrame({"src": src[intra], "dst": dst[intra]})
    num_components, num_covered = 0, 0
    if len(intra_edges) > 0:
        G_intra = cugraph.Graph()
        G_intra.from_cudf_edgelist(intra_edges, "src", "dst")
        components = cugraph.weakly_connected_components(G_intra)
        num_components = components["labels"].nunique()
        num_covered = len(components)
    # vertices without an edge inside their partition are singletons
    num_components += len(parts) - num_covered
    assert num_components == num_clusters


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.sg
@pytest.mark.parametrize("graph_file", UNDIRECTED_DATASETS)
def test_leiden(graph_file):
    edgevals = True

    G = graph_file.get_graph(ignore_weights=not edgevals)
    leiden_parts, leiden_mod = cugraph_leiden(G, random_state=0)
    louvain_parts, louvain_mod = cugraph_louvain(G)

    assert_valid_partition(G, leiden_parts)

    # The reported modularity is the exact modularity of the partition
    assert leiden_mod == pytest.approx(host_modularity(G, leiden_parts), abs=1e-10)

    # Leiden modularity is at least on par with Louvain's
    assert leiden_mod >= 0.99 * louvain_mod


@pytest.mark.sg
def test_leiden_directed_graph():
    edgevals = True
    G = karate_asymmetric.get_graph(
        create_using=cugraph.Graph(directed=True), ignore_weights=not edgevals
    )

    with pytest.raises(ValueError):
        parts, mod = cugraph_leiden(G)


@pytest.mark.sg
def test_leiden_golden_results(input_and_expected_output):
    expected_partition = input_and_expected_output["expected_output"]["partition"]
    expected_mod = input_and_expected_output["expected_output"]["modularity_score"]

    result_partition = input_and_expected_output["result_output"]["partition"]
    result_mod = input_and_expected_output["result_output"]["modularity_score"]

    # the modularity is exact and deterministic
    assert result_mod == pytest.approx(expected_mod, rel=1e-12)

    if input_and_expected_output["input_type"] == "CSR":
        # the partition ids are ordered by decreasing size, and the sizes
        # differ, so the ids are exact
        assert result_partition.to_numpy().tolist() == expected_partition

    else:
        # partitions of equal size: compare up to a bijection of the ids
        expected_to_result_map = {}
        result_to_expected_map = {}
        for e, r in zip(expected_partition, list(result_partition.to_pandas())):
            assert expected_to_result_map.setdefault(e, r) == r
            assert result_to_expected_map.setdefault(r, e) == e


@pytest.mark.sg
@pytest.mark.parametrize("dataset", REFERENCE_DATASETS)
@pytest.mark.parametrize(
    "dtypes", [None, {"src": "int64", "dst": "int64", "wgt": "float64"}]
)
def test_leiden_reference_results(dataset, dtypes):
    G = get_renumber_false_graph(dataset, dtypes=dtypes)
    parts, mod = cugraph.leiden(G, random_state=0)
    parts = sorted_partition(parts)

    expected = _reference_results[dataset.metadata["name"]]
    # bitwise identical, also for int64 vertex ids and float64 weights
    assert float(mod).hex() == expected["modularity_hex"]
    assert parts["partition"].nunique() == expected["num_clusters"]
    assert parts["vertex"].to_numpy().tolist() == list(
        range(len(expected["partition"]))
    )
    assert parts["partition"].to_numpy().tolist() == expected["partition"]


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", UNDIRECTED_DATASETS)
def test_leiden_deterministic(graph_file):
    G = graph_file.get_graph()
    parts, mod = cugraph.leiden(G, random_state=42)

    # same graph, same seed
    parts_2, mod_2 = cugraph.leiden(G, random_state=42)
    assert mod_2 == mod
    assert_frame_equal(sorted_partition(parts_2), sorted_partition(parts))

    # the same graph built again, same seed
    G_2 = graph_file.get_graph()
    parts_3, mod_3 = cugraph.leiden(G_2, random_state=42)
    assert mod_3 == mod
    assert_frame_equal(sorted_partition(parts_3), sorted_partition(parts))

    # only the low 32 bits of the seed are used
    parts_4, mod_4 = cugraph.leiden(G, random_state=42 + 2**32)
    assert mod_4 == mod
    assert_frame_equal(sorted_partition(parts_4), sorted_partition(parts))

    # other seeds give valid partitions too
    for random_state in (1, 2, None):
        other_parts, other_mod = cugraph.leiden(G, random_state=random_state)
        assert_valid_partition(G, other_parts)
        assert other_mod == pytest.approx(host_modularity(G, other_parts), abs=1e-10)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", UNDIRECTED_DATASETS)
@pytest.mark.parametrize("n_iterations", [1, 2, -1])
def test_leiden_n_iterations_and_resolution(graph_file, n_iterations):
    G = graph_file.get_graph()

    num_clusters = []
    for resolution in (0.5, 1.0, 2.0):
        parts, mod = cugraph.leiden(
            G, resolution=resolution, random_state=0, n_iterations=n_iterations
        )
        assert_valid_partition(G, parts)
        assert mod == pytest.approx(
            host_modularity(G, parts, resolution=resolution), abs=1e-10
        )
        num_clusters.append(parts["partition"].nunique())

    # higher resolutions lead to more, smaller communities
    assert num_clusters == sorted(num_clusters)
    assert num_clusters[0] < num_clusters[-1]


@pytest.mark.sg
def test_leiden_deprecated_parameters():
    G = karate.get_graph()
    parts, mod = cugraph.leiden(G, random_state=0)

    # max_iter and theta are accepted (also positionally) but have no effect
    with pytest.warns(FutureWarning, match="max_iter"):
        parts_2, mod_2 = cugraph.leiden(G, 100, 1.0, 0)
    with pytest.warns(FutureWarning, match="theta"):
        parts_3, mod_3 = cugraph.leiden(G, resolution=1.0, random_state=0, theta=1.0)
    with pytest.warns(FutureWarning) as record:
        parts_4, mod_4 = cugraph.leiden(G, 10, 1.0, 0, 0.5)
    assert len(record) == 2

    for p, m in ((parts_2, mod_2), (parts_3, mod_3), (parts_4, mod_4)):
        assert m == mod
        assert_frame_equal(sorted_partition(p), sorted_partition(parts))


@pytest.mark.sg
def test_leiden_invalid_parameters():
    G = karate.get_graph()

    with pytest.raises(ValueError):
        cugraph.leiden(G, n_iterations=0)
    with pytest.raises(ValueError):
        cugraph.leiden(G, n_iterations=-2)
    with pytest.raises(TypeError):
        cugraph.leiden(G, n_iterations=1.5)
    with pytest.raises(ValueError):
        cugraph.leiden(G, resolution=-1.0)
    with pytest.raises(ValueError):
        cugraph.leiden(G, resolution=float("nan"))
    with pytest.raises(ValueError):
        cugraph.leiden(G, beta=-1.0)
    with pytest.raises(NotImplementedError):
        cugraph.leiden(G, beta=0.01)
    # n_iterations and beta are keyword-only
    with pytest.raises(TypeError):
        cugraph.leiden(G, None, 1.0, 0, None, 2)


@pytest.mark.sg
def test_leiden_edge_semantics():
    G = get_renumber_false_graph(karate)
    parts, mod = cugraph.leiden(G, random_state=0)
    parts = sorted_partition(parts)
    edgelist = karate.get_edgelist(download=True)

    def check_same(G_other):
        other_parts, other_mod = cugraph.leiden(G_other, random_state=0)
        assert float(other_mod).hex() == float(mod).hex()
        assert_frame_equal(sorted_partition(other_parts), parts)

    # karate has unit weights: an unweighted graph is the same graph
    G_unweighted = cugraph.Graph()
    G_unweighted.from_cudf_edgelist(edgelist, "src", "dst", renumber=False)
    check_same(G_unweighted)

    # self-loops are ignored
    self_loops = cudf.DataFrame(
        {
            "src": cudf.Series(range(0, 34, 3), dtype="int32"),
            "dst": cudf.Series(range(0, 34, 3), dtype="int32"),
            "wgt": cudf.Series([5.0] * 12, dtype="float32"),
        }
    )
    G_self_loops = cugraph.Graph()
    G_self_loops.from_cudf_edgelist(
        cudf.concat([edgelist, self_loops]), "src", "dst", "wgt", renumber=False
    )
    check_same(G_self_loops)

    # the parallel edges of a MultiGraph are summed (here 1 = 1/2 + 1/4 + 1/4)
    parallel_edges = cudf.concat(
        [
            edgelist.assign(wgt=edgelist["wgt"] * fraction)
            for fraction in (0.5, 0.25, 0.25)
        ]
    )
    G_multi = cugraph.MultiGraph()
    G_multi.from_cudf_edgelist(parallel_edges, "src", "dst", "wgt", renumber=False)
    assert G_multi.number_of_edges() == 3 * G.number_of_edges()
    check_same(G_multi)


@pytest.mark.sg
def test_leiden_invalid_weights():
    edgelist = karate.get_edgelist(download=True)
    edgelist = edgelist.assign(wgt=-edgelist["wgt"])
    G = cugraph.Graph()
    G.from_cudf_edgelist(edgelist, "src", "dst", "wgt", renumber=False)

    with pytest.raises(RuntimeError, match="non-negative"):
        cugraph.leiden(G)
