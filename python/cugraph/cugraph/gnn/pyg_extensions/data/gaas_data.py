from typing import Any
from torch_geometric.typing import Tensor

import torch
from torch import device as TorchDevice
from torch.utils.data import IterableDataset
from torch_geometric.data import Data, RemoteData
from cugraph.gnn.pyg_extensions.data.gaas_storage import GaasStorage

from gaas_client.client import GaasClient
from gaas_client.defaults import graph_id as DEFAULT_GRAPH_ID

import numpy as np
import pandas as pd
import cupy
import cudf

class GaasData(Data, RemoteData, IterableDataset):
    def __init__(self, gaas_client: GaasClient, graph_id: int=DEFAULT_GRAPH_ID, device=TorchDevice('cpu'),
                 ephemeral=False, batch_size=1, shuffle=False):
        super().__init__()
        
        # have to access __dict__ here to ensure the store is a GaasStorage
        storage = GaasStorage(gaas_client, graph_id, device=device, parent=self)
        self.__dict__['_store'] = storage
        self.device = device
        self.ephemeral = ephemeral
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.__extracted_subgraph = None
        self.__iters = 0

    def __del__(self):
        print('destroying a gaasdata object')
        if self.ephemeral:
            self.gaas_client.delete_graph(self.gaas_graph_id)
        if self.__extracted_subgraph is not None:
            self.gaas_client.delete_graph(self.__extracted_subgraph)

    def __next__(self):
        #FIXME handle shuffle
        if self.shuffle:
            raise NotImplementedError('shuffle currently not supported')
        
        start = self.__iters * self.batch_size
        end = min(self.num_edges, (1 + self.__iters) * self.batch_size)
        if start >= self.num_edges:
            raise StopIteration
        batch_idx = range(start, end)

        self.__iters += 1

        eix = self.edge_index[-1, batch_idx]
        
        #FIXME property handle edge labels
        eli = torch.zeros(eix.shape[1], dtype=torch.long, device=self.device)

        yield eix, eli

    def __iter__(self):
        self.reset_iter()
        return self

    def reset_iter(self):
        self.__iters__ = 0
    
    def to(self, to_device: TorchDevice) -> Data:
        return GaasData(
            gaas_client=self.gaas_client,
            graph_id=self.gaas_graph_id,
            device=TorchDevice(to_device),
            ephemeral=self.ephemeral,
            batch_size=self.batch_size,
            shuffle=self.shuffle
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

        sampling_results = self.gaas_client.uniform_neighbor_sample(
            np.array(index, dtype='int32'),
            np.array(num_neighbors, dtype='int32'),
            replace,
            self.extracted_subgraph
        )

        toseries_fn = pd.Series if self.device == 'cpu' else cudf.Series
        concat_fn = pd.concat if self.device == 'cpu' else cudf.concat
        stack_fn = np.stack if self.device == 'cpu' else cupy.stack
        sort_fn = np.sort if self.device == 'cpu' else cupy.sort
        searchsorted_fn = np.searchsorted if self.device == 'cpu' else cupy.searchsorted
        arrayconcat_fn = np.concatenate if self.device == 'cpu' else cupy.concatenate
        arange_fn = np.arange if self.device == 'cpu' else cupy.arange
        toarray_fn = pd.Series.to_numpy if self.device == 'cpu' else cudf.Series.to_cupy

        destinations = toseries_fn(sampling_results.destinations)
        sources = toseries_fn(sampling_results.sources)

        nodes_of_interest = toarray_fn(concat_fn([destinations, sources], axis=0).unique(), dtype='long')
        if self.device == 'cpu':
            noi_tensor = Tensor(nodes_of_interest)
        else:
            noi_tensor = torch.from_dlpack(nodes_of_interest.toDlpack())
        

        rda = stack_fn([nodes_of_interest, arange_fn(len(nodes_of_interest), dtype='long')], axis=-1)
        rda = sort_fn(rda,axis=0)

        ixe = searchsorted_fn(rda[:,0], arrayconcat_fn([toarray_fn(destinations), toarray_fn(sources)]))
        eix = rda[ixe,1].reshape((2, len(sources)))

        if self.device == 'cpu':
            ei = Tensor(eix)
        else:
            ei = torch.from_dlpack(eix.toDlpack())

        sampled_y = self.y
        if sampled_y is not None:
            sampled_y = sampled_y[noi_tensor]

        sampled_x = self.x[noi_tensor]

        data = Data(
            x=sampled_x, 
            edge_index=ei,
            edge_attr=None,
            y=sampled_y
        )
        
        return data
        
    @property
    def extracted_subgraph(self) -> int:
        if self.__extracted_subgraph is None:
            sG = self.gaas_client.extract_subgraph(
                default_edge_weight=1.0,
                allow_multi_edges=True,
                renumber_graph=False,
                add_edge_data=False,
                graph_id=self.gaas_graph_id
            )
            self.__extracted_subgraph = sG
        
        return self.__extracted_subgraph
            
        
    
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