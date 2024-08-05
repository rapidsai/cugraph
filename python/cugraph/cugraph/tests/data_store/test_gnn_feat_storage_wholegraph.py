# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

import pytest
import numpy as np
import os

from cugraph.gnn import FeatureStore

from cugraph.utilities.utils import import_optional, MissingModule

pylibwholegraph = import_optional("pylibwholegraph")
wmb = import_optional("pylibwholegraph.binding.wholememory_binding")
torch = import_optional("torch")
wgth = import_optional("pylibwholegraph.torch")


def runtest(rank: int, world_size: int):
    torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    pylibwholegraph.torch.initialize.init(
        rank,
        world_size,
        rank,
        world_size,
    )
    wm_comm = wgth.get_global_communicator()

    generator = np.random.default_rng(62)
    arr = (
        generator.integers(low=0, high=100, size=100_000)
        .reshape(10_000, -1)
        .astype("float64")
    )

    fs = FeatureStore(backend="wholegraph")
    fs.add_data(arr, "type2", "feat1")
    wm_comm.barrier()

    indices_to_fetch = np.random.randint(low=0, high=len(arr), size=1024)
    output_fs = fs.get_data(indices_to_fetch, type_name="type2", feat_name="feat1")
    assert isinstance(output_fs, torch.Tensor)
    assert output_fs.is_cuda
    expected = arr[indices_to_fetch]
    np.testing.assert_array_equal(output_fs.cpu().numpy(), expected)

    pylibwholegraph.torch.initialize.finalize()


@pytest.mark.sg
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(
    isinstance(pylibwholegraph, MissingModule), reason="wholegraph not available"
)
def test_feature_storage_wholegraph_backend():
    world_size = torch.cuda.device_count()
    print("gpu count:", world_size)
    assert world_size > 0

    print("ignoring gpu count and running on 1 GPU only")

    torch.multiprocessing.spawn(runtest, args=(1,), nprocs=1)


@pytest.mark.mg
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(
    isinstance(pylibwholegraph, MissingModule), reason="wholegraph not available"
)
def test_feature_storage_wholegraph_backend_mg():
    world_size = torch.cuda.device_count()
    print("gpu count:", world_size)
    assert world_size > 0

    torch.multiprocessing.spawn(runtest, args=(world_size,), nprocs=world_size)
