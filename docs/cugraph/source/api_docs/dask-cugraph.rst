~~~~~~~~~~~~~~~~~~~~~~
Multi-GPU with cuGraph
~~~~~~~~~~~~~~~~~~~~~~

cuGraph supports multi-GPU leveraging `Dask <https://dask.org>`_. Dask is a flexible library for parallel computing in Python which makes scaling out your workflow smooth and simple. cuGraph also uses other Dask-based RAPIDS projects such as `dask-cuda <https://github.com/rapidsai/dask-cuda>`_.

Distributed graph analytics
===========================

The current solution is able to scale across multiple GPUs on multiple machines. Distributing the graph and computation lets you analyze datasets far larger than a single GPU’s memory.

With cuGraph and Dask, whether you’re using a single NVIDIA GPU or multiple nodes, your RAPIDS workflow will run smoothly, intelligently distributing the workload across the available resources.

If your graph comfortably fits in memory on a single GPU, you would want to use the single-GPU version of cuGraph. If you want to distribute your workflow across multiple GPUs and have more data than you can fit in memory on a single GPU, you would want to use cuGraph's multi-GPU features.

Example
========

.. code-block:: python

    import dask_cudf
    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster

    import cugraph
    import cugraph.dask as dask_cugraph
    import cugraph.dask.comms.comms as Comms
    from cugraph.generators.rmat import rmat

    input_data_path = "input_data.csv"

    # cluster initialization
    cluster = LocalCUDACluster()
    client = Client(cluster)
    Comms.initialize(p2p=True)

    # helper function to generate random input data
    input_data = rmat(
        scale=5,
        num_edges=400,
        a=0.30,
        b=0.65,
        c=0.05,
        seed=456,
        clip_and_flip=False,
        scramble_vertex_ids=False,
        create_using=None,
    )
    input_data.to_csv(input_data_path, index=False)

    # helper function to set the reader chunk size to automatically get one partition per GPU  
    chunksize = dask_cugraph.get_chunksize(input_data_path)

    # multi-GPU CSV reader
    e_list = dask_cudf.read_csv(
        input_data_path, 
        chunksize=chunksize,
        names=['src', 'dst'],
        dtype=['int32', 'int32'],
    )

    # create graph from input data
    G = cugraph.DiGraph()
    G.from_dask_cudf_edgelist(e_list, source='src', destination='dst')

    # run PageRank
    pr_df = dask_cugraph.pagerank(G, tol=1e-4)

    # cluster clean up
    Comms.destroy()
    client.close()
    cluster.close()


|
