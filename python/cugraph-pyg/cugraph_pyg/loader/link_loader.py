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

import cugraph_pyg
from typing import Union, Tuple, Callable, Optional

from cugraph.utilities.utils import import_optional

torch_geometric = import_optional("torch_geometric")
torch = import_optional("torch")


class LinkLoader:
    """
    Duck-typed version of torch_geometric.loader.LinkLoader.
    Loads samples from batches of input nodes using a
    `~cugraph_pyg.sampler.BaseSampler.sample_from_edges`
    function.
    """

    def __init__(
        self,
        data: Union[
            "torch_geometric.data.Data",
            "torch_geometric.data.HeteroData",
            Tuple[
                "torch_geometric.data.FeatureStore", "torch_geometric.data.GraphStore"
            ],
        ],
        link_sampler: "cugraph_pyg.sampler.BaseSampler",
        edge_label_index: "torch_geometric.typing.InputEdges" = None,
        edge_label: "torch_geometric.typing.OptTensor" = None,
        edge_label_time: "torch_geometric.typing.OptTensor" = None,
        neg_sampling: Optional["torch_geometric.sampler.NegativeSampling"] = None,
        neg_sampling_ratio: Optional[Union[int, float]] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: Optional[bool] = None,
        custom_cls: Optional["torch_geometric.data.HeteroData"] = None,
        input_id: "torch_geometric.typing.OptTensor" = None,
        batch_size: int = 1,  # refers to number of edges in batch
        shuffle: bool = False,
        drop_last: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
            data: Data, HeteroData, or Tuple[FeatureStore, GraphStore]
                See torch_geometric.loader.NodeLoader.
            link_sampler: BaseSampler
                See torch_geometric.loader.LinkLoader.
            edge_label_index: InputEdges
                See torch_geometric.loader.LinkLoader.
            edge_label: OptTensor
                See torch_geometric.loader.LinkLoader.
            edge_label_time: OptTensor
                See torch_geometric.loader.LinkLoader.
            neg_sampling: Optional[NegativeSampling]
                Type of negative sampling to perform, if desired.
                See torch_geometric.loader.LinkLoader.
            neg_sampling_ratio: Optional[Union[int, float]]
                Negative sampling ratio.  Affects how many negative
                samples are generated.
                See torch_geometric.loader.LinkLoader.
            transform: Callable (optional, default=None)
                This argument currently has no effect.
            transform_sampler_output: Callable (optional, default=None)
                This argument currently has no effect.
            filter_per_worker: bool (optional, default=False)
                This argument currently has no effect.
            custom_cls: HeteroData
                This argument currently has no effect.  This loader will
                always return a Data or HeteroData object.
            input_id: OptTensor
                See torch_geometric.loader.LinkLoader.

        """
        if not isinstance(data, (list, tuple)) or not isinstance(
            data[1], cugraph_pyg.data.GraphStore
        ):
            # Will eventually automatically convert these objects to cuGraph objects.
            raise NotImplementedError("Currently can't accept non-cugraph graphs")

        if not isinstance(link_sampler, cugraph_pyg.sampler.BaseSampler):
            raise NotImplementedError("Must provide a cuGraph sampler")

        if edge_label_time is not None:
            raise ValueError("Temporal sampling is currently unsupported")

        if filter_per_worker:
            warnings.warn("filter_per_worker is currently ignored")

        if custom_cls is not None:
            warnings.warn("custom_cls is currently ignored")

        if transform is not None:
            warnings.warn("transform is currently ignored.")

        if transform_sampler_output is not None:
            warnings.warn("transform_sampler_output is currently ignored.")

        if neg_sampling_ratio is not None:
            warnings.warn(
                "The 'neg_sampling_ratio' argument is deprecated in PyG"
                " and is not supported in cuGraph-PyG."
            )

        (
            input_type,
            edge_label_index,
        ) = torch_geometric.loader.utils.get_edge_label_index(
            data,
            (None, edge_label_index),
        )

        self.__input_data = torch_geometric.sampler.EdgeSamplerInput(
            input_id=torch.arange(
                edge_label_index[0].numel(), dtype=torch.int64, device="cuda"
            )
            if input_id is None
            else input_id,
            row=edge_label_index[0],
            col=edge_label_index[1],
            label=edge_label,
            time=edge_label_time,
            input_type=input_type,
        )

        self.__data = data

        self.__link_sampler = link_sampler
        self.__neg_sampling = neg_sampling

        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__drop_last = drop_last

    def __iter__(self):
        if self.__shuffle:
            perm = torch.randperm(self.__input_data.row.numel())
        else:
            perm = torch.arange(self.__input_data.row.numel())

        if self.__drop_last:
            d = perm.numel() % self.__batch_size
            perm = perm[:-d]

        input_data = torch_geometric.sampler.EdgeSamplerInput(
            input_id=self.__input_data.input_id[perm],
            row=self.__input_data.row[perm],
            col=self.__input_data.col[perm],
            label=None
            if self.__input_data.label is None
            else self.__input_data.label[perm],
            time=None
            if self.__input_data.time is None
            else self.__input_data.time[perm],
            input_type=self.__input_data.input_type,
        )

        return cugraph_pyg.sampler.SampleIterator(
            self.__data,
            self.__link_sampler.sample_from_edges(
                input_data,
                neg_sampling=self.__neg_sampling,
            ),
        )
