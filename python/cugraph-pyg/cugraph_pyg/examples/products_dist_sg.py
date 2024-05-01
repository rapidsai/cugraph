import torch
import cupy

import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
from rmm.allocators.torch import rmm_torch_allocator

# Must change allocators immediately upon import
# or else other imports will cause memory to be
# allocated and prevent changing the allocator
rmm.reinitialize(devices=[0], pool_allocator=True, managed_memory=True)
cupy.cuda.set_allocator(rmm_cupy_allocator)
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

import torch_geometric
import cugraph_pyg
from cugraph_pyg.loader import NeighborLoader

# Enable cudf spilling to save gpu memory
from cugraph.testing.mg_utils import enable_spilling
enable_spilling()

# Model parameters
HIDDEN_CHANNELS = 256
NUM_LAYERS = 2
LR = 0.001
NUM_EPOCHS=4
BATCH_SIZE=1024
FANOUT = 30

device = torch.device('cuda')

from ogb.nodeproppred import PygNodePropPredDataset
dataset = PygNodePropPredDataset(name='ogbn-products',
                                 root='/datasets/ogb_datasets') # FIXME remove this
split_idx = dataset.get_idx_split()
data = dataset[0]

graph_store = cugraph_pyg.data.GraphStore()
graph_store[('paper','cites','paper'), 'coo'] = data.edge_index

feature_store = cugraph_pyg.data.TensorDictFeatureStore()
feature_store['paper', 'x'] = data.x
feature_store['paper', 'y'] = data.y

print(graph_store._graph)