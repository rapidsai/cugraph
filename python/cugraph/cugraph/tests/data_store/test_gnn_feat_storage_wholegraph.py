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

import pytest
import numpy as np

from cugraph.gnn import FeatureStore

from cugraph.utilities.utils import import_optional, MissingModule

pylibwholegraph = import_optional("pylibwholegraph")
wmb = import_optional("pylibwholegraph.binding.wholememory_binding")
torch = import_optional("torch")


def runtest(world_rank: int, world_size: int):
    from pylibwholegraph.torch.initialize import init_torch_env_and_create_wm_comm

    wm_comm, _ = init_torch_env_and_create_wm_comm(
        world_rank,
        world_size,
        world_rank,
        world_size,
    )
    wm_comm = wm_comm.wmb_comm

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

    wmb.finalize()


@pytest.mark.sg
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(
    isinstance(pylibwholegraph, MissingModule), reason="wholegraph not available"
)
def test_feature_storage_wholegraph_backend():
    from pylibwholegraph.utils.multiprocess import multiprocess_run

    gpu_count = wmb.fork_get_gpu_count()
    print("gpu count:", gpu_count)
    assert gpu_count > 0

    multiprocess_run(1, runtest)


@pytest.mark.mg
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(
    isinstance(pylibwholegraph, MissingModule), reason="wholegraph not available"
)
def test_feature_storage_wholegraph_backend_mg():
    from pylibwholegraph.utils.multiprocess import multiprocess_run

    gpu_count = wmb.fork_get_gpu_count()
    print("gpu count:", gpu_count)
    assert gpu_count > 0

    multiprocess_run(gpu_count, runtest)
