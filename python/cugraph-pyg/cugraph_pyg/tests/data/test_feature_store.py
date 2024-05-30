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

import pytest

from cugraph.utilities.utils import import_optional, MissingModule

from cugraph_pyg.data import TensorDictFeatureStore

torch = import_optional("torch")


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_tensordict_feature_store_basic_api():
    feature_store = TensorDictFeatureStore()

    node_features_0 = torch.randint(128, (100, 1000))
    node_features_1 = torch.randint(256, (100, 10))

    other_features = torch.randint(1024, (10, 5))

    feature_store["node", "feat0"] = node_features_0
    feature_store["node", "feat1"] = node_features_1
    feature_store["other", "feat"] = other_features

    assert (feature_store["node"]["feat0"][:] == node_features_0).all()
    assert (feature_store["node"]["feat1"][:] == node_features_1).all()
    assert (feature_store["other"]["feat"][:] == other_features).all()

    assert len(feature_store.get_all_tensor_attrs()) == 3

    del feature_store["node", "feat0"]
    assert len(feature_store.get_all_tensor_attrs()) == 2
