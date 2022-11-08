# Copyright (c) 2022, NVIDIA CORPORATION.
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

import numpy as np
import pytest


def create_gs(client, device_id=None):
    from cugraph.gnn.dgl_extensions.cugraph_service_store import CuGraphRemoteStore

    gs = CuGraphRemoteStore(client.graph(), client, device_id, backend_lib="cupy")
    gs.add_node_data_from_parquet(
        file_path="nt.a.parquet", node_col_name="node_id", ntype="nt.a", node_offset=0
    )
    gs.add_node_data_from_parquet(
        file_path="nt.b.parquet",
        node_col_name="node_id",
        ntype="nt.b",
        node_offset=gs.num_nodes(),
    )
    gs.add_node_data_from_parquet(
        file_path="nt.c.parquet",
        node_col_name="node_id",
        ntype="nt.c",
        node_offset=gs.num_nodes(),
    )

    can_etype = "('nt.a', 'connects', 'nt.b')"
    gs.add_edge_data_from_parquet(
        file_path=f"{can_etype}.parquet",
        node_col_names=["src", "dst"],
        src_offset=0,
        dst_offset=3,
        canonical_etype=can_etype,
    )
    can_etype = "('nt.a', 'connects', 'nt.c')"
    gs.add_edge_data_from_parquet(
        file_path=f"{can_etype}.parquet",
        node_col_names=["src", "dst"],
        src_offset=0,
        dst_offset=6,
        canonical_etype=can_etype,
    )
    can_etype = "('nt.c', 'connects', 'nt.c')"
    gs.add_edge_data_from_parquet(
        file_path=f"{can_etype}.parquet",
        node_col_names=["src", "dst"],
        src_offset=6,
        dst_offset=6,
        canonical_etype=can_etype,
    )

    return gs


def assert_valid_device(cp_ar, device_id):
    import cupy as cp

    if device_id is None:
        return True
    else:
        device_n = cp.cuda.Device(device_id)
        if cp_ar.device != device_n:
            print(f"device = {cp_ar.device}, expected_device = {device_n}")


def assert_valid_gs(gs):
    import cudf

    assert gs.etypes[0] == "('nt.a', 'connects', 'nt.b')"
    assert gs.ntypes[0] == "nt.a"
    assert gs.num_nodes_dict["nt.a"] == 3
    assert gs.num_edges_dict["('nt.a', 'connects', 'nt.b')"] == 3
    assert gs.num_nodes("nt.c") == 5

    print("Verified ntypes, etypes, num_nodes")

    # Test Get Node Storage
    result = gs.get_node_storage(key="node_feat", ntype="nt.a", indices_offset=0).fetch(
        [0, 1, 2]
    )
    assert_valid_device(result, gs.device_id)
    result = result.get()
    expected_result = np.asarray([0, 10, 20], dtype=np.int32)
    np.testing.assert_equal(result, expected_result)

    result = gs.get_node_storage(key="node_feat", ntype="nt.b", indices_offset=3).fetch(
        [0, 1, 2]
    )
    assert_valid_device(result, gs.device_id)
    result = result.get()
    expected_result = np.asarray([30, 40, 50], dtype=np.int32)
    np.testing.assert_equal(result, expected_result)

    result = gs.get_node_storage(key="node_feat", ntype="nt.c", indices_offset=5).fetch(
        [1, 2, 3]
    )
    assert_valid_device(result, gs.device_id)
    result = result.get()
    expected_result = np.asarray([60, 70, 80], dtype=np.int32)
    np.testing.assert_equal(result, expected_result)

    # Test Get Edge Storage
    result = gs.get_edge_storage(
        key="edge_feat", etype="('nt.a', 'connects', 'nt.b')", indices_offset=0
    ).fetch([0, 1, 2])
    assert_valid_device(result, gs.device_id)
    result = result.get()
    expected_result = np.asarray([10, 11, 12], dtype=np.int32)
    np.testing.assert_equal(result, expected_result)

    result = gs.get_edge_storage(
        key="edge_feat", etype="('nt.a', 'connects', 'nt.c')", indices_offset=0
    ).fetch([4, 5])
    assert_valid_device(result, gs.device_id)
    result = result.get()
    expected_result = np.asarray([14, 15], dtype=np.int32)
    np.testing.assert_equal(result, expected_result)

    result = gs.get_edge_storage(
        key="edge_feat", etype="('nt.c', 'connects', 'nt.c')", indices_offset=0
    ).fetch([6, 8])
    assert_valid_device(result, gs.device_id)
    result = result.get()
    expected_result = np.asarray([16, 18], dtype=np.int32)
    np.testing.assert_equal(result, expected_result)

    print("Verified edge_feat, node_feat")

    # Verify set_sg_dtype
    # verify extracted_reverse_subgraph
    subgraph, src_range = gs.extracted_reverse_subgraph
    dtype = gs.set_sg_node_dtype(subgraph)
    assert dtype == "int32"

    # Sampling Results
    nodes_cap = {"nt.c": cudf.Series([6]).to_dlpack()}
    result = gs.sample_neighbors(nodes_cap)
    result = {
        k: cudf.DataFrame(
            {
                "src": cudf.from_dlpack(v[0]),
                "dst": cudf.from_dlpack(v[1]),
                "eid": cudf.from_dlpack(v[2]),
            }
        )
        for k, v in result.items()
        if v[0] is not None
    }

    src_vals = result["('nt.c', 'connects', 'nt.c')"]["src"].values.get()
    sorted(src_vals)
    expected_vals = np.asarray([7, 8, 9], dtype=np.int32)
    np.testing.assert_equal(src_vals, expected_vals)


@pytest.mark.skip(reason="Enable when cugraph-service lands in the CI")
def test_remote_wrappers():
    from cugraph_service_client.client import CugraphServiceClient as Client

    # TODO: Check with rick on how to test it
    # Can only be tested after the packages land
    c = Client()
    device_ls = [None, 0, 1]
    for d in device_ls:
        gs = create_gs(c)
        assert_valid_gs(gs)
