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

from typing import Union, Optional, Dict

from cugraph.utilities.utils import import_optional

import cugraph_dgl
from cugraph_dgl.typing import TensorType
from cugraph_dgl.utils.cugraph_conversion_utils import _cast_to_torch_tensor

dgl = import_optional("dgl")
torch = import_optional("torch")


class DataLoader:
    """
    Duck-typed version of dgl.dataloading.DataLoader
    """

    def __init__(
        self,
        graph: "cugraph_dgl.Graph",
        indices: TensorType,
        graph_sampler: "cugraph_dgl.dataloading.Sampler",
        device: Union[int, str, "torch.device"] = None,
        use_ddp: bool = False,
        ddp_seed: int = 0,
        batch_size=1,
        drop_last: bool = False,
        shuffle: bool = False,
        use_prefetch_thread: Optional[bool] = None,
        use_alternate_streams: Optional[bool] = None,
        pin_prefetcher: Optional[bool] = None,
        use_uva=False,
        gpu_cache: Dict[str, Dict[str, int]] = None,
        output_format: str = "dgl.Block",
        **kwargs,
    ):
        """
        Parameters
        ----------
        graph: cugraph_dgl.Graph
            The graph being sampled.  Can be a single-GPU or multi-GPU graph.
        indices: TensorType
            The seed nodes for sampling.  If use_ddp=True, then all seed
            nodes should be provided.  If use_ddp=False, then only the seed
            nodes assigned to this worker should be provided.
        graph_sampler: cugraph_dgl.dataloading.Sampler
            The sampler responsible for sampling the graph and producing
            output minibatches.
        device: Union[int, str, torch.device]
            Optional.
            The device assigned to this loader ('cpu', 'cuda' or device id).
            Defaults to the current device.
        use_ddp: bool
            Optional (default=False).
            If true, this argument will assume the entire list of input seed
            nodes is being passed to each worker, and will appropriately
            split and shuffle the list.
            It false, then it is assumed that the list of input seed nodes
            is comprised of the union of the lists provided to each worker.
        ddp_seed: int
            Optional (default=0).
            The seed used for dividing and shuffling data if use_ddp=True.
            Has no effect if use_ddp=False.
        use_uva: bool
            Optional (default=False).
            Whether to use pinned memory and unified virtual addressing
            to perform sampling.
            This argument is ignored by cuGraph-DGL.
        use_prefetch_thread: bool
            Optional (default=False).
            Whether to spawn a new thread for feature fetching.
            This argument is ignored by cuGraph-DGL.
        use_alternate_streams: bool
            Optional (default=False).
            Whether to perform feature fetching on a separate stream.
            This argument is ignored by cuGraph-DGL.
        pin_prefetcher: bool
            Optional (default=False).
            Whether to pin the feature tensors.
            This argument is currently ignored by cuGraph-DGL.
        gpu_cache: Dict[str, Dict[str, int]]
            List of features to cache using HugeCTR.
            This argument is not supported by cuGraph-DGL and
            will result in an error.
        output_format: str
            Optional (default="dgl.Block").
            The output format for blocks.
            Can be either "dgl.Block" or "cugraph_dgl.nn.SparseGraph".
        """

        if use_uva:
            warnings.warn("The 'use_uva' argument is ignored by cuGraph-DGL.")
        if use_prefetch_thread:
            warnings.warn(
                "The 'use_prefetch_thread' argument is ignored by cuGraph-DGL."
            )
        if use_alternate_streams:
            warnings.warn(
                "The 'use_alternate_streams' argument is ignored by cuGraph-DGL."
            )
        if pin_prefetcher:
            warnings.warn("The 'pin_prefetcher' argument is ignored by cuGraph-DGL.")
        if gpu_cache:
            raise ValueError(
                "HugeCTR is not supported by cuGraph-DGL. "
                "Consider using WholeGraph for feature storage"
                " in cugraph_dgl.Graph instead."
            )

        indices = _cast_to_torch_tensor(indices)

        self.__dataset = dgl.dataloading.create_tensorized_dataset(
            indices,
            batch_size,
            drop_last,
            use_ddp,
            ddp_seed,
            shuffle,
            kwargs.get("persistent_workers", False),
        )

        self.__output_format = output_format
        self.__sampler = graph_sampler
        self.__batch_size = batch_size
        self.__graph = graph
        self.__device = device

    @property
    def dataset(
        self,
    ) -> Union[
        "dgl.dataloading.dataloader.TensorizedDataset",
        "dgl.dataloading.dataloader.DDPTensorizedDataset",
    ]:
        return self.__dataset

    def __iter__(self):
        # TODO move to the correct device (rapidsai/cugraph-gnn#11)
        return self.__sampler.sample(
            self.__graph,
            self.__dataset,
            batch_size=self.__batch_size,
        )
