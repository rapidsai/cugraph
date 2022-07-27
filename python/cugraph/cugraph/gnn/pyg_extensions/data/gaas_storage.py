from typing import Any
from typing import List
from typing import Union

import torch
from torch import device as TorchDevice
from torch_geometric.typing import ProxyTensor
from torch_geometric.data.storage import GlobalStorage

from gaas_client.client import GaasClient

import numpy as np

EDGE_KEYS = ["_DST_", "_SRC_"] # reversed order in PyG
VERTEX_KEYS = ["_VERTEX_"]


class TorchTensorGaasGraphDataProxy(ProxyTensor):
    """
    Implements a partial Torch Tensor interface that forwards requests to a
    GaaS server maintaining the actual data in a graph instance.
    The interface supported consists of only the APIs specific DGL workflows
    need - anything else will raise AttributeError.
    """
    _data_categories = ["vertex", "edge"]

    def __init__(self, 
                 gaas_client: GaasClient, 
                 gaas_graph_id: int, 
                 data_category: str, 
                 device:TorchDevice=TorchDevice('cpu'),
                 property_keys: List[str]=None,
                 transposed: bool=False,
                 dtype: torch.dtype=torch.float32):
        if data_category not in self._data_categories:
            raise ValueError("data_category must be one of "
                             f"{self._data_categories}, got {data_category}")

        if property_keys is None:
            if data_category == 'vertex':
                property_keys = VERTEX_KEYS
            else:
                property_keys = EDGE_KEYS

        self.__client = gaas_client
        self.__graph_id = gaas_graph_id
        self.__category = data_category
        self.__device = device
        self.__property_keys = np.array(property_keys)
        self.__transposed = transposed
        self.dtype = dtype

    def __getitem__(self, index: Union[int, tuple]) -> Any:
        """
        Returns a torch.Tensor containing the edge or vertex data (based on the
        instance's data_category) for index, retrieved from graph data on the
        instance's GaaS server.
        """
        
        if isinstance(index, torch.Tensor):
            index = [int(i) for i in index]
        
        property_keys = self.__property_keys

        if isinstance(index, (list, tuple)):
            if len(index) == 2:
                index = [index[0], index[1]]
            else:
                index = [index, -1]
        else:
            index = [index, -1]

        if self.__transposed:
            index = [index[1], index[0]]

        if index[1] != -1:
            property_keys = property_keys[index[1]]
            if isinstance(property_keys, str):
                property_keys = [property_keys]

        if self.__category == "edge":
            data = self.__client.get_graph_edge_dataframe_rows(
                index_or_indices=index[0], graph_id=self.__graph_id,
                property_keys=list(property_keys))

        else:
            data = self.__client.get_graph_vertex_dataframe_rows(
                index_or_indices=index[0], graph_id=self.__graph_id,
                property_keys=list(property_keys))

        if self.__transposed:
            torch_data = torch.from_numpy(data.T).to(self.device)
        else:
            # FIXME handle non-numeric datatypes
            torch_data = torch.from_numpy(data)

        return torch_data.to(self.dtype).to(self.__device)

    @property
    def shape(self) -> torch.Size:
        if self.__category == "edge":
            # Handle Edge properties
            s = [self.__client.get_num_edges(self.__graph_id), len(self.__property_keys)]
        elif self.__category == "vertex":
            # Handle Vertex properties
            s = [self.__client.get_num_vertices(self.__graph_id), len(self.__property_keys)]
        else:
            raise AttributeError(f'invalid category {self.__category}')
        
        if self.__transposed:
            s = [s[1],s[0]]
        
        return torch.Size(s)

    @property
    def device(self) -> TorchDevice:
        return self.__device
    
    @property
    def is_cuda(self) -> bool:
        return self.__device._type == 'cuda'

    def to(self, to_device: TorchDevice):
        return TorchTensorGaasGraphDataProxy(
            self.__client, 
            self.__graph_id, 
            self.__category, 
            to_device, 
            property_keys=self.__property_keys, 
            transposed=self.__transposed
        )
    
    def dim(self) -> int:
        return self.shape[0]
    
    def size(self, idx=None) -> Any:
        if idx is None:
            return self.shape
        else:
            return self.shape[idx]


class GaasStorage(GlobalStorage):
    def __init__(self, gaas_client: GaasClient, gaas_graph_id: int, device: TorchDevice=TorchDevice('cpu'), parent=None):
        super().__init__(_parent=parent)
        setattr(self, 'gaas_client', gaas_client)
        setattr(self, 'gaas_graph_id', gaas_graph_id)
        setattr(self, 'node_index', TorchTensorGaasGraphDataProxy(gaas_client, gaas_graph_id, 'vertex', device, dtype=torch.long))
        setattr(self, 'edge_index', TorchTensorGaasGraphDataProxy(gaas_client, gaas_graph_id, 'edge', device, transposed=True, dtype=torch.long))

        vertex_property_keys = gaas_client.get_graph_vertex_property_keys(graph_id=gaas_graph_id)
        if 'y' in vertex_property_keys:
            vertex_property_keys.remove('y')
        #edge_property_keys = gaas_client.get_graph_edge_property_keys(graph_id=gaas_graph_id)

        setattr(self, 'x', TorchTensorGaasGraphDataProxy(gaas_client, gaas_graph_id, 'vertex', device, dtype=torch.float, property_keys=vertex_property_keys))
    
        # The y attribute is special and needs to be overridden
        if self.is_node_attr('y'):
            setattr(self, 'y', TorchTensorGaasGraphDataProxy(gaas_client, gaas_graph_id, 'vertex', device, dtype=torch.float, property_keys=['y'], transposed=False))
        
        setattr(
            self,
            'graph_info',
            gaas_client.get_graph_info(
                keys=[
                    'num_vertices',
                    'num_edges',
                    'num_vertex_properties',
                    'num_edge_properties'
                ],
                graph_id=gaas_graph_id
            )
        )
    
    @property
    def num_nodes(self) -> int:
        return self.graph_info['num_vertices']
    
    @property
    def num_node_features(self) -> int:
        return self.graph_info['num_vertex_properties']
    
    @property
    def num_edge_features(self) -> int:
        return self.graph_info['num_edge_properties']

    @property
    def num_edges(self) -> int:
        return self.graph_info['num_edges']

    def is_node_attr(self, key: str) -> bool:
        if key == 'x':
            return True
        return self.gaas_client.is_vertex_property(key, self.gaas_graph_id)

    def is_edge_attr(self, key: str) -> bool:
        return self.gaas_client.is_edge_property(key, self.gaas_graph_id)
    
    def __getattr__(self, key: str) -> Any:
        if key in self:
            return self[key]
        elif self.gaas_client.is_vertex_property(key, self.gaas_graph_id):
            return TorchTensorGaasGraphDataProxy(
                self.gaas_client,
                self.gaas_graph_id,
                'vertex',
                self.node_index.device,
                [key]
            )
        elif self.gaas_client.is_edge_property(key, self.gaas_graph_id):
            return TorchTensorGaasGraphDataProxy(
                self.gaas_client,
                self.gaas_graph_id,
                'edge',
                self.edge_index.device,
                [key]
            )
        
        raise AttributeError(key)