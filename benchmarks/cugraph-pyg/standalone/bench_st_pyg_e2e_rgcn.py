if __name__ == '__main__':
    import os
    import sys
    import time

    import cugraph
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    from cugraph.dask.comms import comms as Comms
    from cugraph.generators import rmat

    import rmm
    import cupy as cp
    import numpy as np
    import cudf, dask_cudf
    from cugraph.testing.mg_utils import start_dask_client, stop_dask_client

    from cugraph_pyg.sampler import CuGraphSampler
    from cugraph_pyg.data import to_pyg

    import torch
    import pandas as pd

    from ogb.nodeproppred import NodePropPredDataset

    from cugraph.experimental import PropertyGraph, MGPropertyGraph

    from cugraph_pyg.data import to_pyg
    from cugraph_pyg.sampler import CuGraphSampler

    from torch_geometric.loader import NodeLoader


    def make_loader_hetero_mag(mg=False, batch_size=100, num_neighbors=[10,25], replace=False):
        # Load MAG into CPU memory
        dataset = NodePropPredDataset(name="ogbn-mag")

        data = dataset[0]
        if mg:
            pG = MGPropertyGraph()
        else:
            pG = PropertyGraph()

        # Load the vertex ids into a new property graph
        vertex_offsets = {}
        last_offset = 0

        for node_type, num_nodes in data[0]["num_nodes_dict"].items():
            vertex_offsets[node_type] = last_offset
            last_offset += num_nodes

            blank_df = cudf.DataFrame(
                {
                    "id": range(
                        vertex_offsets[node_type], vertex_offsets[node_type] + num_nodes
                    )
                }
            )
            blank_df.id = blank_df.id.astype("int64")
            if mg:
                blank_df = dask_cudf.from_cudf(blank_df, npartitions=2)

            pG.add_vertex_data(blank_df, vertex_col_name="id", type_name=node_type)

        # Add the remaining vertex features
        for i, (node_type, node_features) in enumerate(data[0]["node_feat_dict"].items()):
            vertex_offset = vertex_offsets[node_type]

            feature_df = cudf.DataFrame(node_features)
            feature_df.columns = [str(c) for c in range(feature_df.shape[1])]
            feature_df["id"] = range(vertex_offset, vertex_offset + node_features.shape[0])
            feature_df.id = feature_df.id.astype("int64")
            if mg:
                feature_df = dask_cudf.from_cudf(feature_df, npartitions=2)

            pG.add_vertex_data(feature_df, vertex_col_name="id", type_name=node_type)

        # Fill in an empty value for vertices without properties.
        pG.fillna_vertices(0.0)

        # Add the edges
        for i, (edge_key, eidx) in enumerate(data[0]["edge_index_dict"].items()):
            node_type_src, edge_type, node_type_dst = edge_key
            print(node_type_src, edge_type, node_type_dst)
            vertex_offset_src = vertex_offsets[node_type_src]
            vertex_offset_dst = vertex_offsets[node_type_dst]
            eidx = [n + vertex_offset_src for n in eidx[0]], [
                n + vertex_offset_dst for n in eidx[1]
            ]

            edge_df = cudf.DataFrame({"src": eidx[0], "dst": eidx[1]})
            edge_df.src = edge_df.src.astype("int64")
            edge_df.dst = edge_df.dst.astype("int64")
            edge_df["type"] = edge_type
            if mg:
                edge_df = dask_cudf.from_cudf(edge_df, npartitions=2)

            # Adding backwards edges is currently required in both
            # the cuGraph PG and PyG APIs.
            pG.add_edge_data(edge_df, vertex_col_names=["src", "dst"], type_name=edge_type)
            pG.add_edge_data(
                edge_df, vertex_col_names=["dst", "src"], type_name=f"{edge_type}_bw"
            )

        # Add the target variable
        y_df = cudf.DataFrame(data[1]["paper"], columns=["y"])
        y_df["id"] = range(vertex_offsets["paper"], vertex_offsets["paper"] + len(y_df))
        y_df.id = y_df.id.astype("int64")
        if mg:
            y_df = dask_cudf.from_cudf(y_df, partitions=2)

        pG.add_vertex_data(y_df, vertex_col_name="id", type_name="paper")

        # Construct a graph/feature store and loaders
        feature_store, graph_store = to_pyg(pG, renumber_graph=False, backend='cupy')
        sampler = CuGraphSampler(
            data=(feature_store, graph_store),
            shuffle=True,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            replace=replace,
            directed=True
        )
        loader = NodeLoader(
            data=(feature_store, graph_store),
            shuffle=True,
            batch_size=batch_size,
            node_sampler=sampler,
            input_nodes=("author", graph_store.get_vertex_index("author")),
        )

        return loader

    # BEGIN WORKFLOW

    n_devices = os.getenv('DASK_NUM_WORKERS', 4)
    n_devices = int(n_devices)

    visible_devices = ','.join([str(i) for i in range(1, n_devices+1)])

    cluster = LocalCUDACluster(protocol='ucx', rmm_pool_size='25GB', CUDA_VISIBLE_DEVICES=visible_devices)
    client = Client(cluster)
    Comms.initialize(p2p=True)
    rmm.reinitialize(pool_allocator=True)
    torch.cuda.memory.change_current_allocator(rmm.rmm_torch_allocator)

    kx_mag = 1
    seed = 0x08

    print("creating loader...")
    st = time.perf_counter_ns()

    loader = make_loader_hetero_mag(mg=True)

    print(f"done creating loader, took {((time.perf_counter_ns() - st) / 1e9)}s")

    #batch_sizes = [100, 500, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

    print(next(iter(loader)))

    '''
    run_start = time.perf_counter_ns()
    df = pd.DataFrame()
    for run, batch_size in enumerate(batch_sizes):
        if batch_size > num_verts:
            num_start_verts = int(num_verts * 0.25)
        else:
            num_start_verts = batch_size

        start_list = torch.randint(0, num_verts, (num_start_verts,), dtype=torch.int64)

        num_tries = 20
        t = []
        for i in range(num_tries):
            s = time.perf_counter_ns()
            sampler.sample_from_nodes((None, start_list, None))   
            e = time.perf_counter_ns()
            t.append((e - s) / 1e6)

        t = sorted(t)
        median = t[len(t) // 2]
        lq = t[len(t) // 4]
        uq = t[len(t) * 3 // 4]
        iqr = uq - lq
        lf = lq - 1.5 * iqr
        uf = uq + 1.5 * iqr

        old_len = len(t)
        t = [x for x in t if x > lf and x < uf]

        mean = sum(t) / len(t)
        min_t = min(t)
        max_t = max(t)
        removed = old_len - len(t)

        df = pd.concat([
            df,
            pd.DataFrame([[batch_size, mean, median, min_t, max_t, removed]], columns=['batch size','mean','median','min','max','outliers'])
        ])

        percent_done = int((run + 1) / len(batch_sizes) * 100)
        print(f'{percent_done}% complete, {(time.perf_counter_ns() - run_start) / 1e9} seconds elapsed.')

    '''

    stop_dask_client(client)
    print("\ndask_client fixture: client.close() called")




