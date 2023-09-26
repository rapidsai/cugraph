import pytest
import numpy as np

import cudf
from cugraph.gnn import FeatureStore

import pylibwholegraph.binding.wholememory_binding as wmb
from pylibwholegraph.torch.initialize import init_torch_env_and_create_wm_comm
from pylibwholegraph.utils.multiprocess import multiprocess_run

import torch

def func(world_rank: int, world_size: int):
    wm_comm, _ = init_torch_env_and_create_wm_comm(
        world_rank, world_size, world_rank, world_size,
    )
    wm_comm = wm_comm.wmb_comm

    ar3 = np.random.randint(low=0, high=100, size=100_000).reshape(10_000, -1)
    fs = FeatureStore(backend="wholegraph")
    fs.add_data(ar3, "type2", "feat1")

    indices_to_fetch = np.random.randint(low=0, high=len(ar3), size=1024)
    output_fs = fs.get_data(indices_to_fetch, type_name="type2", feat_name="feat1")
    assert isinstance(output_fs, torch.Tensor)
    assert output_fs.is_cuda
    expected = ar3[indices_to_fetch]
    np.testing.assert_array_equal(output_fs.cpu().numpy(), expected)


    wmb.finalize()

def test_feature_storage_wholegraph_backend():
    gpu_count = wmb.fork_get_gpu_count()
    print('gpu count:', gpu_count)
    assert gpu_count > 0

    # FIXME make this work in an MG environment
    multiprocess_run(1, func)