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

from cugraph.utilities.utils import import_optional, MissingModule

torch = import_optional("torch")
dgl = import_optional("dgl")
wgth = import_optional("pylibwholegraph.torch")


class WholeFeatureStore(
    object if isinstance(dgl, MissingModule) else dgl.storages.base.FeatureStorage
):
    """
    Interface for feature storage.
    """

    def __init__(
        self,
        tensor: "torch.Tensor",
        memory_type: str = "distributed",
        location: str = "cpu",
    ):
        """
        Constructs a new WholeFeatureStore object that wraps a WholeGraph wholememory
        distributed tensor.

        Parameters
        ----------
        t: torch.Tensor
            The local slice of the tensor being distributed.  These should be in order
            by rank (i.e. rank 0 contains elements 0-9, rank 1 contains elements 10-19,
            rank 3 contains elements 20-29, etc.)  The sizes do not need to be equal.
        memory_type: str (optional, default='distributed')
            The memory type of this store.  Options are
            'distributed', 'chunked', and 'continuous'.
            For more information consult the WholeGraph
            documentation.
        location: str(optional, default='cpu')
            The location ('cpu' or 'cuda') where data is stored.
        """
        self.__wg_comm = wgth.get_local_node_communicator()

        if len(tensor.shape) > 2:
            raise ValueError("Only 1-D or 2-D tensors are supported by WholeGraph.")

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        ld = torch.tensor(tensor.shape[0], device="cuda", dtype=torch.int64)
        sizes = torch.empty((world_size,), device="cuda", dtype=torch.int64)
        torch.distributed.all_gather_into_tensor(sizes, ld)

        sizes = sizes.cpu()
        ld = sizes.sum()

        self.__td = -1 if len(tensor.shape) == 1 else tensor.shape[1]
        global_shape = [
            int(ld),
            self.__td if self.__td > 0 else 1,
        ]

        if self.__td < 0:
            tensor = tensor.reshape((tensor.shape[0], 1))

        wg_tensor = wgth.create_wholememory_tensor(
            self.__wg_commm,
            memory_type,
            location,
            global_shape,
            tensor.dtype,
            [global_shape[1], 1],
        )

        offset = sizes[:rank].sum() if rank > 0 else 0

        wg_tensor.scatter(
            tensor.clone(memory_format=torch.contiguous_format).cuda(),
            torch.arange(
                offset, offset + tensor.shape[0], dtype=torch.int64, device="cuda"
            ).contiguous(),
        )

        self.__wg_comm.barrier()

        self.__wg_tensor = wg_tensor

    def requires_ddp(self) -> bool:
        return True

    def fetch(
        self,
        indices: torch.Tensor,
        device: torch.cuda.Device,
        pin_memory=False,
        **kwargs,
    ):
        if pin_memory:
            warnings.warn("pin_memory has no effect for WholeFeatureStorage.")

        t = self.__wg_tensor.gather(
            indices.cuda(),
            force_dtype=self.__wg_tensor.dtype,
        )

        if self.__td < 0:
            t = t.reshape((t.shape[0],))

        return t
