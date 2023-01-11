if __name__ == '__main__':
    import os
    import sys
    import time

    import cugraph
    from dask_cuda import LocalCUDACluster
    from cugraph.dask.common.mg_utils import get_visible_devices
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

    # module-wide fixtures

    visible_devices = get_visible_devices()
    n_devices = os.getenv('DASK_NUM_WORKERS', 4)
    n_devices = int(n_devices)

    visible_devices = ','.join([visible_devices[i] for i in range(1, n_devices+1)])

    cluster = LocalCUDACluster(protocol='ucx', rmm_pool_size='25GB', CUDA_VISIBLE_DEVICES=visible_devices)
    client = Client(cluster)
    Comms.initialize(p2p=True)
    rmm.reinitialize(pool_allocator=True)

    scale = 24
    edge_factor = 4

    num_edges = (2**scale) * edge_factor
    seed = 0x08

    print("creating graph...")
    st = time.perf_counter_ns()

    edgelist_df = rmat(
        scale,
        num_edges,
        0.57,  # from Graph500
        0.19,  # from Graph500
        0.19,  # from Graph500
        seed,
        clip_and_flip=False,
        scramble_vertex_ids=False,  # FIXME: need to understand relevance of this
        create_using=None,  # None == return edgelist
        mg=True,
    )
    edgelist_df['src'] = edgelist_df.src.astype('int64')
    edgelist_df['dst'] = edgelist_df.dst.astype('int64')

    vertex_df = dask_cudf.concat(
        [edgelist_df['src'], edgelist_df['dst']]
    ).unique()
    vertex_df.name = 'vtx'

    vertex_df = vertex_df.to_frame().reset_index().rename(columns={'index':'vid'})

    edgelist_df = edgelist_df.merge(
        vertex_df, how='left', left_on='src', right_on='vtx'
    ).drop(
        ['src', 'vtx'], axis=1
    ).rename(
        columns={'vid':'src'}
    ).merge(
        vertex_df, how='left', left_on='dst', right_on='vtx'
    ).drop(
        ['dst', 'vtx'], axis=1
    ).rename(
        columns={'vid':'dst'}
    ).reset_index(drop=True).repartition(npartitions=n_devices*4)
    
    edgelist_df["weight"] = cp.float32(1)
    edgelist_df['eid'] = 1
    edgelist_df['eid'] = edgelist_df['eid'].cumsum() - 1
    edgelist_df = edgelist_df.persist()

    vertex_df = dask_cudf.concat(
        [edgelist_df['src'], edgelist_df['dst']]
    ).unique()
    vertex_df.name = 'vtx'

    pG = cugraph.experimental.MGPropertyGraph()

    pG.add_vertex_data(vertex_df.to_frame(), vertex_col_name='vtx', type_name='vt1')

    pG.add_edge_data(
        edgelist_df,
        vertex_col_names=['src','dst'],
        type_name='et1',
        edge_id_col_name='eid',
        property_columns=['weight']
    )

    print(f"done creating graph, took {((time.perf_counter_ns() - st) / 1e9)}s")

    data = to_pyg(pG, renumber_graph=False, backend='cupy')

    sampler = CuGraphSampler(
        data, 
        method='uniform_neighbor',        
        replace=False,
        directed=True,
        edge_types=['et1'],
        num_neighbors=[10,25]
    )

    rng = np.random.default_rng(seed)
    num_verts = pG.get_num_vertices()

    batch_sizes = [100, 500, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

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

    print(df)
    stop_dask_client(client)
    print("\ndask_client fixture: client.close() called")




