# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest
import os
import shutil

import cupy
import cudf

from cugraph.datasets import karate
from cugraph.gnn import UniformNeighborSampler, DistSampleWriter
from cugraph.gnn.data_loading.bulk_sampler_io import create_df_from_disjoint_arrays

from pylibcugraph import SGGraph, ResourceHandle, GraphProperties

from cugraph.utilities.utils import (
    create_directory_with_overwrite,
    import_optional,
    MissingModule,
)


torch = import_optional("torch")
if not isinstance(torch, MissingModule):
    if torch.cuda.is_available():
        from rmm.allocators.torch import rmm_torch_allocator

        torch.cuda.change_current_allocator(rmm_torch_allocator)
    else:
        pytest.skip("CUDA-enabled PyTorch is unavailable", allow_module_level=True)


@pytest.fixture
def karate_graph() -> SGGraph:
    el = karate.get_edgelist().reset_index().rename(columns={"index": "eid"})
    G = SGGraph(
        ResourceHandle(),
        GraphProperties(is_multigraph=True, is_symmetric=False),
        el.src.astype("int64"),
        el.dst.astype("int64"),
        edge_id_array=el.eid,
    )

    return G


@pytest.mark.sg
@pytest.mark.parametrize("equal_input_size", [True, False])
@pytest.mark.parametrize("fanout", [[2, 2], [4, 4], [4, 2, 1]])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_dist_sampler_simple(
    scratch_dir, karate_graph, batch_size, fanout, equal_input_size
):
    G = karate_graph

    samples_path = os.path.join(scratch_dir, "test_bulk_sampler_simple")
    create_directory_with_overwrite(samples_path)

    writer = DistSampleWriter(samples_path)

    sampler = UniformNeighborSampler(G, writer, fanout=fanout)

    seeds = cupy.array([0, 5, 10, 15], dtype="int64")

    sampler.sample_from_nodes(
        seeds, batch_size=batch_size, assume_equal_input_size=equal_input_size
    )

    recovered_samples = cudf.read_parquet(samples_path)
    original_el = karate.get_edgelist()

    for b in range(len(seeds) // batch_size):
        el_start = recovered_samples.label_hop_offsets.iloc[b * len(fanout)]
        el_end = recovered_samples.label_hop_offsets.iloc[(b + 1) * len(fanout)]

        src = recovered_samples.majors.iloc[el_start:el_end]
        dst = recovered_samples.minors.iloc[el_start:el_end]
        edge_id = recovered_samples.edge_id.iloc[el_start:el_end]

        map_start = recovered_samples.renumber_map_offsets[b]
        map_end = recovered_samples.renumber_map_offsets[b + 1]
        renumber_map = recovered_samples["map"].iloc[map_start:map_end]

        src = renumber_map.iloc[src.values]
        dst = renumber_map.iloc[dst.values]

        for i in range(len(edge_id)):
            assert original_el.src.iloc[edge_id.iloc[i]] == src.iloc[i]
            assert original_el.dst.iloc[edge_id.iloc[i]] == dst.iloc[i]

    shutil.rmtree(samples_path)


@pytest.mark.sg
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.parametrize("seeds_per_call", [4, 5, 10])
@pytest.mark.parametrize("compression", ["CSR", "COO"])
def test_dist_sampler_buffered_in_memory(
    scratch_dir: str, karate_graph: SGGraph, seeds_per_call: int, compression: str
):
    G = karate_graph

    samples_path = os.path.join(scratch_dir, "test_bulk_sampler_buffered_in_memory")
    create_directory_with_overwrite(samples_path)

    seeds = cupy.arange(10, dtype="int64")

    unbuffered_sampler = UniformNeighborSampler(
        G,
        writer=DistSampleWriter(samples_path),
        local_seeds_per_call=seeds_per_call,
        compression=compression,
    )

    buffered_sampler = UniformNeighborSampler(
        G,
        writer=None,
        local_seeds_per_call=seeds_per_call,
        compression=compression,
    )

    unbuffered_results = unbuffered_sampler.sample_from_nodes(
        seeds,
        batch_size=4,
    )

    unbuffered_results = [
        (create_df_from_disjoint_arrays(r[0]), r[1], r[2]) for r in unbuffered_results
    ]

    buffered_results = buffered_sampler.sample_from_nodes(seeds, batch_size=4)
    buffered_results = [
        (create_df_from_disjoint_arrays(r[0]), r[1], r[2]) for r in buffered_results
    ]

    assert len(buffered_results) == len(unbuffered_results)

    for k in range(len(buffered_results)):
        br, bs, be = buffered_results[k]
        ur, us, ue = unbuffered_results[k]

        assert (be - bs) == (ue - us)

        for col in ur.columns:
            assert (br[col].dropna() == ur[col].dropna()).all()

    shutil.rmtree(samples_path)


@pytest.mark.sg
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_dist_sampler_hetero_from_nodes():
    props = GraphProperties(
        is_symmetric=False,
        is_multigraph=True,
    )

    handle = ResourceHandle()

    srcs = cupy.array([4, 5, 6, 7, 8, 9, 8, 9, 8, 7, 6, 5, 4, 5])
    dsts = cupy.array([0, 1, 2, 3, 3, 0, 4, 5, 6, 8, 7, 8, 9, 9])
    eids = cupy.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7])
    etps = cupy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype="int32")

    graph = SGGraph(
        handle,
        props,
        srcs,
        dsts,
        vertices_array=cupy.arange(10),
        edge_id_array=eids,
        edge_type_array=etps,
        weight_array=cupy.ones((14,), dtype="float32"),
    )

    sampler = UniformNeighborSampler(
        graph,
        fanout=[-1, -1, -1, -1],
        writer=None,
        compression="COO",
        heterogeneous=True,
        vertex_type_offsets=cupy.array([0, 4, 10]),
        num_edge_types=2,
        deduplicate_sources=True,
    )

    out = sampler.sample_from_nodes(
        nodes=cupy.array([4, 5]),
        input_id=cupy.array([5, 10]),
        metadata={"some_key": "some_value"},
    )

    out = [z for z in out]
    assert len(out) == 1
    out, _, _ = out[0]

    lho = out["label_type_hop_offsets"]
    assert out["some_key"] == "some_value"

    # Edge type 0
    emap = out["edge_renumber_map"][
        out["edge_renumber_map_offsets"][0] : out["edge_renumber_map_offsets"][1]
    ]

    smap = out["map"][out["renumber_map_offsets"][1] : out["renumber_map_offsets"][2]]

    dmap = out["map"][out["renumber_map_offsets"][0] : out["renumber_map_offsets"][1]]

    # Edge type 0, hop 0
    hop_start = lho[0]
    hop_end = lho[1]

    assert hop_end - hop_start == 2

    e = out["edge_id"][hop_start:hop_end]
    e = emap[e]
    assert sorted(e.tolist()) == [0, 1]

    s = cupy.asarray(smap[out["majors"][hop_start:hop_end]])
    d = cupy.asarray(dmap[out["minors"][hop_start:hop_end]])

    assert sorted(s.tolist()) == [4, 5]
    assert sorted(d.tolist()) == [0, 1]

    # Edge type 0, hop 1
    hop_start = int(lho[1])
    hop_end = int(lho[2])

    assert hop_end - hop_start == 2

    e = out["edge_id"][hop_start:hop_end]
    e = emap[e]
    assert sorted(e.tolist()) == [4, 5]

    s = cupy.asarray(smap[out["majors"][hop_start:hop_end]])
    d = cupy.asarray(dmap[out["minors"][hop_start:hop_end]])

    assert sorted(s.tolist()) == [8, 9]
    assert sorted(d.tolist()) == [0, 3]

    #############################

    # Edge type 1
    emap = out["edge_renumber_map"][
        out["edge_renumber_map_offsets"][1] : out["edge_renumber_map_offsets"][2]
    ]

    smap = out["map"][out["renumber_map_offsets"][1] : out["renumber_map_offsets"][2]]

    dmap = smap

    # Edge type 1, hop 0
    hop_start = lho[2]
    hop_end = lho[3]

    assert hop_end - hop_start == 3

    e = out["edge_id"][hop_start:hop_end]
    e = emap[e]
    assert sorted(e.tolist()) == [5, 6, 7]

    s = cupy.asarray(smap[out["majors"][hop_start:hop_end]])
    d = cupy.asarray(dmap[out["minors"][hop_start:hop_end]])

    assert sorted(s.tolist()) == [4, 5, 5]
    assert sorted(d.tolist()) == [8, 9, 9]

    # Edge type 1, hop 1
    hop_start = lho[3]
    hop_end = lho[4]

    assert hop_end - hop_start == 3

    e = out["edge_id"][hop_start:hop_end]
    e = emap[e]
    assert sorted(e.tolist()) == [0, 1, 2]

    s = cupy.asarray(smap[out["majors"][hop_start:hop_end]])
    d = cupy.asarray(dmap[out["minors"][hop_start:hop_end]])

    assert sorted(s.tolist()) == [8, 8, 9]
    assert sorted(d.tolist()) == [4, 5, 6]
