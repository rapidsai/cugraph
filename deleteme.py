# %% [markdown]
# # PyG+cuGraph Heterogeneous MAG Example
# # Skip notebook test
# 
# ### Requires installation of PyG

# %% [markdown]
# ## Setup

# %%
import sys
import rmm

rmm.reinitialize(pool_allocator=True,initial_pool_size=5e+9, maximum_pool_size=20e+9)

# %% [markdown]
# ## Load MAG into CPU Memory

# %%
import cugraph
import cudf
from ogb.nodeproppred import NodePropPredDataset

dataset = NodePropPredDataset(name = 'ogbn-mag') 

data = dataset[0]

# %% [markdown]
# ## Create PropertyGraph from MAG Data

# %% [markdown]
# ### Partially Load the Vertex Data (just ids)

# %%
import cudf
import dask_cudf
import cugraph
from cugraph.experimental import MGPropertyGraph
from cugraph.experimental import PropertyGraph
pG = PropertyGraph()

vertex_offsets = {}
last_offset = 0

for node_type, num_nodes in data[0]['num_nodes_dict'].items():
    vertex_offsets[node_type] = last_offset
    last_offset += num_nodes
    
    blank_df = cudf.DataFrame({'id':range(vertex_offsets[node_type], vertex_offsets[node_type] + num_nodes)})
    blank_df.id = blank_df.id.astype('int32')
    if isinstance(pG, MGPropertyGraph):
        blank_df = dask_cudf.from_cudf(blank_df, npartitions=2)
    pG.add_vertex_data(blank_df, vertex_col_name='id', type_name=node_type)

vertex_offsets

# %% [markdown]
# ### Add the Remaining Node Features

# %%
for i, (node_type, node_features) in enumerate(data[0]['node_feat_dict'].items()):
    vertex_offset = vertex_offsets[node_type]

    feature_df = cudf.DataFrame(node_features)
    feature_df.columns = [str(c) for c in range(feature_df.shape[1])]
    feature_df['id'] = range(vertex_offset, vertex_offset + node_features.shape[0])
    feature_df.id = feature_df.id.astype('int32')
    if isinstance(pG, MGPropertyGraph):
        feature_df = dask_cudf.from_cudf(feature_df, npartitions=2)

    pG.add_vertex_data(feature_df, vertex_col_name='id', type_name=node_type)

# %% [markdown]
# ### Add the Edges

# %%
for i, (edge_key, eidx) in enumerate(data[0]['edge_index_dict'].items()):
    node_type_src, edge_type, node_type_dst = edge_key
    print(node_type_src, edge_type, node_type_dst)
    vertex_offset_src = vertex_offsets[node_type_src]
    vertex_offset_dst = vertex_offsets[node_type_dst]
    eidx = [n + vertex_offset_src for n in eidx[0]], [n + vertex_offset_dst for n in eidx[1]]

    edge_df = cudf.DataFrame({'src':eidx[0], 'dst':eidx[1]})
    edge_df.src = edge_df.src.astype('int32')
    edge_df.dst = edge_df.dst.astype('int32')
    edge_df['type'] = edge_type
    if isinstance(pG, MGPropertyGraph):
        edge_df = dask_cudf.from_cudf(edge_df, npartitions=2)

    # Adding backwards edges is currently required in both the cuGraph PG and PyG APIs.
    pG.add_edge_data(edge_df, vertex_col_names=['src','dst'], type_name=edge_type)
    pG.add_edge_data(edge_df, vertex_col_names=['dst','src'], type_name=f'{edge_type}_bw')

# %% [markdown]
# ### Add the Target Variable

# %%
y_df = cudf.DataFrame(data[1]['paper'], columns=['y'])
y_df['id'] = range(vertex_offsets['paper'], vertex_offsets['paper'] + len(y_df))
y_df.id = y_df.id.astype('int32')
if isinstance(pG, MGPropertyGraph):
    y_df = dask_cudf.from_cudf(y_df, npartitions=2)

pG.add_vertex_data(y_df, vertex_col_name='id', type_name='paper')

# %% [markdown]
# ### Construct a Graph Store, Feature Store, and Loaders

# %%
from cugraph.gnn.pyg_extensions import to_pyg

feature_store, graph_store = to_pyg(pG)

# %%
from torch_geometric.loader import LinkNeighborLoader
from cugraph.gnn.pyg_extensions import CuGraphLinkNeighborLoader
loader = CuGraphLinkNeighborLoader(
    data=(feature_store, graph_store),
    edge_label_index='writes',
    shuffle=True,
    num_neighbors=[10,25],
    batch_size=50,
)

test_loader = CuGraphLinkNeighborLoader(
    data=(feature_store, graph_store),
    edge_label_index='writes',
    shuffle=True,
    num_neighbors=[10,25],
    batch_size=50,
)


# %% [markdown]
# ### Create the Network

# %%
edge_types = [attr.edge_type for attr in graph_store.get_all_edge_attrs()]
edge_types

# %%
num_classes = pG.get_vertex_data(columns=['y'])['y'].max() + 1
if isinstance(pG, MGPropertyGraph):
    num_classes = num_classes.compute()
num_classes

# %%
import torch
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, Linear, SAGEConv

class HeteroGNN(torch.nn.Module):
    def __init__(self, edge_types, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in edge_types
            })
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        print(x_dict, edge_index_dict)
        return self.lin(x_dict['paper'])


model = HeteroGNN(edge_types, hidden_channels=64, out_channels=num_classes,
                  num_layers=2).cuda()

with torch.no_grad():  # Initialize lazy modules.
    data = next(iter(loader))
    out = model(data.x_dict, data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

num_batches = 5
def train():
    model.train()
    optimizer.zero_grad()
    for b_i, data in enumerate(loader):
        if b_i == num_batches:
            break

        out = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(out, data.y_dict['paper'])
        loss.backward()
        optimizer.step()
    
    return float(loss) / num_batches


@torch.no_grad()
def test():
    model.eval()
    test_iter = iter(test_loader)

    acc = 0.0
    for _ in range(2*num_batches):
        data = next(test_iter)
        pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)

        
        acc += (pred == data['paper'].y).sum() / len(data['paper'])
    return acc / (2*num_batches)


for epoch in range(1, 101):
    loss = train()
    train_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}')


# %% [markdown]
# ### Train the Network

# %%
for epoch in range(1, 101):
    loss = train()
    train_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}')


