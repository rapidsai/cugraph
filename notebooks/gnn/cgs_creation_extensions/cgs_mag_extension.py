# Copyright (c) 2022, NVIDIA CORPORATION.
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

# cuGraph or cuGraph-Service is required; each has its own version of
# import_optional and we need to select the correct one.

from ogb.nodeproppred import NodePropPredDataset

def create_mag(server):
    dataset = NodePropPredDataset(name = 'ogbn-mag') 

    data = dataset[0]

    # Can't import these before loading MAG; known OGB issue
    import cudf
    import dask_cudf

    from cugraph.experimental import MGPropertyGraph
    from cugraph.experimental import PropertyGraph

    pG = PropertyGraph()

    vertex_offsets = {}
    last_offset = 0

    for node_type, num_nodes in data[0]['num_nodes_dict'].items():
        vertex_offsets[node_type] = last_offset
        last_offset += num_nodes
        
        blank_df = cudf.DataFrame({'id':range(vertex_offsets[node_type], vertex_offsets[node_type] + num_nodes)})
        blank_df.id = blank_df.id.astype('int64')
        if isinstance(pG, MGPropertyGraph):
            blank_df = dask_cudf.from_cudf(blank_df, npartitions=2)
        pG.add_vertex_data(blank_df, vertex_col_name='id', type_name=node_type)


    for i, (node_type, node_features) in enumerate(data[0]['node_feat_dict'].items()):
        vertex_offset = vertex_offsets[node_type]

        feature_df = cudf.DataFrame(node_features)
        feature_df.columns = [str(c) for c in range(feature_df.shape[1])]
        feature_df['id'] = range(vertex_offset, vertex_offset + node_features.shape[0])
        feature_df.id = feature_df.id.astype('int64')
        if isinstance(pG, MGPropertyGraph):
            feature_df = dask_cudf.from_cudf(feature_df, npartitions=2)

        pG.add_vertex_data(feature_df, vertex_col_name='id', type_name=node_type)

    # Fill in an empty value for vertices without properties.
    pG.fillna_vertices(0.0)

    for i, (edge_key, eidx) in enumerate(data[0]['edge_index_dict'].items()):
        node_type_src, edge_type, node_type_dst = edge_key
        
        vertex_offset_src = vertex_offsets[node_type_src]
        vertex_offset_dst = vertex_offsets[node_type_dst]
        eidx = [n + vertex_offset_src for n in eidx[0]], [n + vertex_offset_dst for n in eidx[1]]

        edge_df = cudf.DataFrame({'src':eidx[0], 'dst':eidx[1]})
        edge_df.src = edge_df.src.astype('int64')
        edge_df.dst = edge_df.dst.astype('int64')
        edge_df['type'] = edge_type
        if isinstance(pG, MGPropertyGraph):
            edge_df = dask_cudf.from_cudf(edge_df, npartitions=2)

        # Adding backwards edges is currently required in both the cuGraph PG and PyG APIs.
        pG.add_edge_data(edge_df, vertex_col_names=['src','dst'], type_name=edge_type)
        pG.add_edge_data(edge_df, vertex_col_names=['dst','src'], type_name=f'{edge_type}_bw')

    y_df = cudf.DataFrame(data[1]['paper'], columns=['y'])
    y_df['id'] = range(vertex_offsets['paper'], vertex_offsets['paper'] + len(y_df))
    y_df.id = y_df.id.astype('int64')
    if isinstance(pG, MGPropertyGraph):
        y_df = dask_cudf.from_cudf(y_df, npartitions=2)

    pG.add_vertex_data(y_df, vertex_col_name='id', type_name='paper')

    return pG