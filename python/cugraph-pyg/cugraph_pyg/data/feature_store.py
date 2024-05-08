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

import warnings

from typing import Optional, Tuple, List

from cugraph.utilities.utils import import_optional, MissingModule

torch = import_optional("torch")
torch_geometric = import_optional("torch_geometric")
tensordict = import_optional("tensordict")


class TensorDictFeatureStore(
    object
    if isinstance(torch_geometric, MissingModule)
    else torch_geometric.data.FeatureStore
):
    """
    A basic implementation of the PyG FeatureStore interface that stores
    feature data in a single TensorDict.  This type of feature store is
    not distributed, so each node will have to load the entire graph's
    features into memory.
    """

    def __init__(self):
        super().__init__()

        self.__features = {}

    def _put_tensor(
        self,
        tensor: "torch_geometric.typing.FeatureTensorType",
        attr: "torch_geometric.data.feature_store.TensorAttr",
    ) -> bool:
        if attr.group_name in self.__features:
            td = self.__features[attr.group_name]
            batch_size = td.batch_size[0]

            if attr.is_set("index"):
                if attr.attr_name in td.keys():
                    if attr.index.shape[0] != batch_size:
                        raise ValueError(
                            "Leading size of index tensor "
                            "does not match existing tensors for group name "
                            f"{attr.group_name}; Expected {batch_size}, "
                            f"got {attr.index.shape[0]}"
                        )
                    td[attr.attr_name][attr.index] = tensor
                    return True
                else:
                    warnings.warn(
                        "Ignoring index parameter "
                        f"(attribute does not exist for group {attr.group_name})"
                    )

            if tensor.shape[0] != batch_size:
                raise ValueError(
                    "Leading size of input tensor does not match "
                    f"existing tensors for group name {attr.group_name};"
                    f" Expected {batch_size}, got {tensor.shape[0]}"
                )
        else:
            batch_size = tensor.shape[0]
            self.__features[attr.group_name] = tensordict.TensorDict(
                {}, batch_size=batch_size
            )

        self.__features[attr.group_name][attr.attr_name] = tensor
        return True

    def _get_tensor(
        self, attr: "torch_geometric.data.feature_store.TensorAttr"
    ) -> Optional["torch_geometric.typing.FeatureTensorType"]:
        if attr.group_name not in self.__features:
            return None

        if attr.attr_name not in self.__features[attr.group_name].keys():
            return None

        tensor = self.__features[attr.group_name][attr.attr_name]
        return (
            tensor
            if (attr.index is None or (not attr.is_set("index")))
            else tensor[attr.index]
        )

    def _remove_tensor(
        self, attr: "torch_geometric.data.feature_store.TensorAttr"
    ) -> bool:
        if attr.group_name not in self.__features:
            return False

        if attr.attr_name not in self.__features[attr.group_name]:
            return False

        del self.__features[attr.group_name][attr.attr_name]
        return True

    def _get_tensor_size(
        self, attr: "torch_geometric.data.feature_store.TensorAttr"
    ) -> Tuple:
        return self._get_tensor(attr).size()

    def get_all_tensor_attrs(
        self,
    ) -> List["torch_geometric.data.feature_store.TensorAttr"]:
        attrs = []
        for group_name, td in self.__features.items():
            for attr_name in td.keys():
                attrs.append(
                    torch_geometric.data.feature_store.TensorAttr(
                        group_name,
                        attr_name,
                    )
                )

        return attrs
