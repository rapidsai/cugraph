from typing import Any
from torch_geometric.typing import Tensor

import torch
from torch import device as TorchDevice
from torch_geometric.data import Data, RemoteData
from cugraph.gnn.pyg_extensions.data.gaas_storage import GaasStorage

from gaas_client.client import GaasClient
from gaas_client.defaults import graph_id as DEFAULT_GRAPH_ID

import numpy as np

class GaasData(Data, RemoteData):
    def __init__(self, gaas_client: GaasClient, graph_id: int=DEFAULT_GRAPH_ID, device=TorchDevice('cpu'), ephemeral=False):
        super().__init__()
        
        # have to access __dict__ here to ensure the store is a CuGraphStorage
        storage = GaasStorage(gaas_client, graph_id, device=device, parent=self)
        self.__dict__['_store'] = storage
        self.device = device
        self.ephemeral = ephemeral

    def __del__(self):
        print('destroying a gaasdata object')
        if self.ephemeral:
            self.gaas_client.delete_graph(self.gaas_graph_id)
    
    def to(self, to_device: TorchDevice) -> Data:
        return GaasData(
            self.gaas_client,
            self.gaas_graph_id,
            TorchDevice(to_device)
        )

    def cuda(self):
        return self.to('cuda')
    
    def cpu(self):
        return self.to('cpu')
    
    def stores_as(self, data: 'Data'):
        return self
    
    def neighbor_sample(
            self,
            index: Tensor,
            num_neighbors: Tensor,
            replace: bool,
            directed: bool) -> Any:

        if directed is False:
            raise NotImplementedError('Undirected support not available')
        
        if isinstance(index, torch.Tensor):
            index = index.to('cpu')
        if isinstance(num_neighbors, torch.Tensor):
            num_neighbors = num_neighbors.to('cpu')

        new_graph_id = self.gaas_client.uniform_neighbor_sample(
            np.array(index, dtype='int32'),
            np.array(num_neighbors, dtype='int32'),
            replace,
            self.gaas_graph_id
        )

        return GaasData(self.gaas_client, 
                        new_graph_id, 
                        device=self.device, 
                        ephemeral=True)
    
    def extract_subgraph(
            self, 
            node: Tensor, 
            edges: Tensor, 
            enumerated_edges: Tensor,
            perm: Tensor) -> Any:
        """
        node: Nodes to extract
        edges: Edges to extract (0: src, 1: dst)
        enumerated_edges: Numbered edges to extract
        """
        raise NotImplementedError