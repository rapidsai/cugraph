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
wgth = import_optional("pylibwholegraph.torch")


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
        """
        Constructs an empty TensorDictFeatureStore.
        """
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

        if attr.attr_name not in self.__features[attr.group_name].keys():
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


class WholeFeatureStore(
    object
    if isinstance(torch_geometric, MissingModule)
    else torch_geometric.data.FeatureStore
):
    """
    A basic implementation of the PyG FeatureStore interface that stores
    feature data in WholeGraph WholeMemory.  This type of feature store is
    distributed, and avoids data replication across workers.

    Data should be sliced before being passed into this feature store.
    That means each worker should have its own partition and put_tensor
    should be called for each worker's local partition.  When calling
    get_tensor, multi_get_tensor, etc., the entire tensor can be accessed
    regardless of what worker's partition the desired slice of the tensor
    is on.
    """

    def __init__(self, memory_type="distributed", location="cpu"):
        """
        Constructs an empty WholeFeatureStore.

        Parameters
        ----------
        memory_type: str (optional, default='distributed')
            The memory type of this store.  Options are
            'distributed', 'chunked', and 'continuous'.
            For more information consult the WholeGraph
            documentation.
        location: str(optional, default='cpu')
            The location ('cpu' or 'cuda') where data is stored.
        """
        super().__init__()

        self.__features = {}

        self.__wg_comm = wgth.get_global_communicator()
        self.__wg_type = memory_type
        self.__wg_location = location

    def _put_tensor(
        self,
        tensor: "torch_geometric.typing.FeatureTensorType",
        attr: "torch_geometric.data.feature_store.TensorAttr",
    ) -> bool:
        wg_comm_obj = self.__wg_comm

        if attr.is_set("index"):
            if (attr.group_name, attr.attr_name) in self.__features:
                raise NotImplementedError(
                    "Updating an embedding from an index"
                    " is not supported by WholeGraph."
                )
            else:
                warnings.warn(
                    "Ignoring index parameter "
                    f"(attribute does not exist for group {attr.group_name})"
                )

        if len(tensor.shape) > 2:
            raise ValueError("Only 1-D or 2-D tensors are supported by WholeGraph.")

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        ld = torch.tensor(tensor.shape[0], device="cuda", dtype=torch.int64)
        sizes = torch.empty((world_size,), device="cuda", dtype=torch.int64)
        torch.distributed.all_gather_into_tensor(sizes, ld)

        sizes = sizes.cpu()
        ld = sizes.sum()

        td = -1 if len(tensor.shape) == 1 else tensor.shape[1]
        global_shape = [
            int(ld),
            td if td > 0 else 1,
        ]

        if td < 0:
            tensor = tensor.reshape((tensor.shape[0], 1))

        wg_embedding = wgth.create_wholememory_tensor(
            wg_comm_obj,
            self.__wg_type,
            self.__wg_location,
            global_shape,
            tensor.dtype,
            [global_shape[1], 1],
        )

        offset = sizes[:rank].sum() if rank > 0 else 0

        wg_embedding.scatter(
            tensor.clone(memory_format=torch.contiguous_format).cuda(),
            torch.arange(
                offset, offset + tensor.shape[0], dtype=torch.int64, device="cuda"
            ).contiguous(),
        )

        wg_comm_obj.barrier()

        self.__features[attr.group_name, attr.attr_name] = (wg_embedding, td)
        return True

    def _get_tensor(
        self, attr: "torch_geometric.data.feature_store.TensorAttr"
    ) -> Optional["torch_geometric.typing.FeatureTensorType"]:
        if (attr.group_name, attr.attr_name) not in self.__features:
            return None

        emb, td = self.__features[attr.group_name, attr.attr_name]

        if attr.index is None or (not attr.is_set("index")):
            attr.index = torch.arange(emb.shape[0], dtype=torch.int64)

        attr.index = attr.index.cuda()
        t = emb.gather(
            attr.index,
            force_dtype=emb.dtype,
        )

        if td < 0:
            t = t.reshape((t.shape[0],))

        return t

    def _remove_tensor(
        self, attr: "torch_geometric.data.feature_store.TensorAttr"
    ) -> bool:
        if (attr.group_name, attr.attr_name) not in self.__features:
            return False

        del self.__features[attr.group_name, attr.attr_name]
        return True

    def _get_tensor_size(
        self, attr: "torch_geometric.data.feature_store.TensorAttr"
    ) -> Tuple:
        return self.__features[attr.group_name, attr.attr_name].shape

    def get_all_tensor_attrs(
        self,
    ) -> List["torch_geometric.data.feature_store.TensorAttr"]:
        attrs = []
        for (group_name, attr_name) in self.__features.keys():
            attrs.append(
                torch_geometric.data.feature_store.TensorAttr(
                    group_name=group_name,
                    attr_name=attr_name,
                )
            )

        return attrs
