from cugraph.utilities.utils import import_optional, MissingModule

from typing import Union, Optional

import numpy as np
import cupy
import cudf
import pandas

# Have to use import_optional even though these are required
# dependencies in order to build properly.
torch_geometric = import_optional('torch_geometric')
torch = import_optional('torch')
tensordict = import_optional('tensordict')

GraphStore = object if isinstance(torch_geometric, MissingModule) else torch_geometric.GraphStore
TensorType = Union['torch.Tensor', cupy.ndarray, np.ndarray, cudf.Series, pandas.Series]

class DistGraphStore(GraphStore):
    """
    This object uses lazy graph creation.  Users can repeatedly call
    put_edge_index, and the tensors won't be converted into a cuGraph
    graph until one is needed (i.e. when creating a loader).
    """

    def __init__(self, ):
        self._edge_indices = tensordict.TensorDict({}, batch_size=(2,))
        self._sizes = {}
        self.__graph = None

    def _put_edge_index(self, edge_index:'torch_geometric.typing.EdgeTensorType', edge_attr:'torch_geometric.data.EdgeAttr') ->bool:
        if edge_attr.layout != 'coo':
            raise ValueError("Only COO format supported")

        if isinstance(edge_index, (cupy.ndarray, cudf.Series)):
            edge_index = torch.as_tensor(edge_index, device='cuda')
        elif isinstance(edge_index, (np.ndarray)):
            edge_index = torch.as_tensor(edge_index, device='cpu')
        elif isinstance(edge_index, pandas.Series):
            edge_index = torch.as_tensor(edge_index.values, device='cpu')
        elif isinstance(edge_index, cudf.Series):
            edge_index = torch.as_tensor(edge_index.values, device='cuda')
        
        self._edge_indices[edge_attr.edge_type] = torch.stack(edge_index)
        self._sizes[edge_attr.edge_type] = edge_attr.size

        # invalidate the graph
        self.__graph = None
        return True

    def _get_edge_index(self, edge_attr:'torch_geometric.data.EdgeAttr')->Optional['torch_geometric.typing.EdgeTensorType']:
        ei = torch_geometric.EdgeIndex(
            self._edge_indices[edge_attr.edge_type]
        )
        
        
        if edge_attr.layout == 'csr':
            return ei.sort_by('row').values.get_csr()
        elif edge_attr.layout == 'csc':
            return ei.sort_by('col').values.get_csc()

        return ei

    def _remove_edge_index(self, edge_attr:'torch_geometric.data.EdgeAttr')->bool:
        del self._edge_indices[edge_attr.edge_type]
        
        # invalidate the graph
        self.__graph = None
        return True

    def get_all_edge_attrs(self) -> List['torch_geometric.data.EdgeAttr']:
        attrs = []
        for et in self._edge_indices.keys(leaves_only=True, include_nested=True):
            attrs.append(
                torch_geometric.data.EdgeAttr(
                    edge_type=et,
                    layout='coo',
                    is_sorted=False,
                    size=self._sizes[et]
                )
            )
        
        return attrs