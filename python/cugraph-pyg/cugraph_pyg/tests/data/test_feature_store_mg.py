# Copyright (c) 2024, NVIDIA CORPORATION.
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

import os

import pytest

from cugraph.utilities.utils import import_optional, MissingModule

from cugraph_pyg.data import TensorDictFeatureStore, WholeFeatureStore

torch = import_optional("torch")
pylibwholegraph = import_optional("pylibwholegraph")


def run_test_wholegraph_feature_store_basic_api(rank, world_size, dtype):
    if dtype == "float32":
        torch_dtype = torch.float32
    elif dtype == "int64":
        torch_dtype = torch.int64

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

    features = torch.arange(0, world_size * 2000)
    features = features.reshape((features.numel() // 100, 100)).to(torch_dtype)

    tensordict_store = TensorDictFeatureStore()
    tensordict_store["node", "fea"] = features

    whole_store = WholeFeatureStore()
    whole_store["node", "fea"] = torch.tensor_split(features, world_size)[rank]

    ix = torch.arange(features.shape[0])
    assert (
        whole_store["node", "fea"][ix].cpu() == tensordict_store["node", "fea"][ix]
    ).all()

    label = torch.arange(0, features.shape[0]).reshape((features.shape[0], 1))
    tensordict_store["node", "label"] = label
    whole_store["node", "label"] = torch.tensor_split(label, world_size)[rank]

    assert (
        whole_store["node", "fea"][ix].cpu() == tensordict_store["node", "fea"][ix]
    ).all()

    pylibwholegraph.torch.initialize.finalize()


@pytest.mark.skipif(
    isinstance(pylibwholegraph, MissingModule), reason="wholegraph not available"
)
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.parametrize("dtype", ["float32", "int64"])
@pytest.mark.mg
def test_wholegraph_feature_store_basic_api(dtype):
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        run_test_wholegraph_feature_store_basic_api,
        args=(
            world_size,
            dtype,
        ),
        nprocs=world_size,
    )
