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

torch_geometric = import_optional('torch_geometric')
torch = import_optional('torch')

class NodeLoader:
    """
    Duck-typed version of torch_geometric.loader.NodeLoader
    """

    def __init__(self,
        data: Union['torch_geometric.data.Data', 'torch_geometric.data.HeteroData', Tuple['torch_geometric.data.FeatureStore', 'torch_geometric.data.GraphStore']],
        node_sampler: 'cugraph_pyg.sampler.BaseSampler',
        input_nodes: 'torch_geometric.typing.InputNodes' = None,
        input_time: 'torch_geometric.typing.OptTensor' = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: Optional[bool] = None,
        custom_cls: Optional['torch_geometric.data.HeteroData'] = None,
        input_id: 'torch_geometric.typing.OptTensor' = None,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False, 
        **kwargs,):
            """
            Parameters
            ----------
                data: Data, HeteroData, or Tuple[FeatureStore, GraphStore]
                    See torch_geometric.loader.NodeLoader.
                node_sampler: BaseSampler
                    See torch_geometric.loader.NodeLoader.
                input_nodes: InputNodes
                    See torch_geometric.loader.NodeLoader.                
                input_time: OptTensor
                    See torch_geometric.loader.NodeLoader.
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
                    See torch_geometric.loader.NodeLoader.

            """
            if not isinstance(data, (list, tuple)) or not isinstance(data[1], cugraph_pyg.data.GraphStore):
                # Will eventually automatically convert these objects to cuGraph objects.
                raise NotImplementedError("Currently can't accept non-cugraph graphs")
            
            if not isinstance(node_sampler, cugraph_pyg.sampler.BaseSampler):
                raise NotImplementedError("Must provide a cuGraph sampler")

            if input_time is not None:
                raise ValueError("Temporal sampling is currently unsupported")
            
            if filter_per_worker:
                warnings.warn("filter_per_worker is currently ignored")
            
            if custom_cls is not None:
                warnings.warn("custom_cls is currently ignored")
            
            if transform is not None:
                warnings.warn("transform is currently ignored.")

            if transform_sampler_output is not None:
                warnings.warn("transform_sampler_output is currently ignored.")

            input_type, input_nodes, input_id = torch_geometric.loader.utils.get_input_nodes(
                data,
                input_nodes,
                input_id,
            )

            self.__input_data = torch_geometric.loader.node_loader.NodeSamplerInput(
                input_id=input_id,
                node=input_nodes,
                time=None,
                input_type=input_type,
            )

            self.__data = data

            self.__node_sampler = node_sampler

            self.__batch_size = batch_size
            self.__shuffle = shuffle
            self.__drop_last = drop_last
            
    
    def __iter__(self):
        if self.__shuffle:
            perm = torch.randperm(self.__input_data.node.numel())
        else:
            perm = torch.arange(self.__input_data.node.numel())
        
        if self.__drop_last:
            d = perm.numel() % self.__batch_size
            perm = perm[:-d]
        
        input_data = torch_geometric.loader.node_loader.NodeSamplerInput(
            input_id=None if self.__input_data.input_id is None else self.__input_data.input_id[perm],
            node=self.__input_data.node[perm],
            time=None if self.__input_data.time is None else self.__input_data.time[perm],
            input_type=self.__input_data.input_type,
        )
            
        return cugraph_pyg.sampler.SampleIterator(
            self.__data,
            self.__node_sampler.sample_from_nodes(input_data)
        )