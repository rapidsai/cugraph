=======================
pylibwholegraph API doc
=======================

.. currentmodule:: pylibwholegraph

APIs
----
.. autosummary::
    :toctree: ../../api/wg

    torch.initialize.init_torch_env
    torch.initialize.init_torch_env_and_create_wm_comm
    torch.initialize.finalize
    torch.comm.WholeMemoryCommunicator
    torch.comm.set_world_info
    torch.comm.create_group_communicator
    torch.comm.destroy_communicator
    torch.comm.get_global_communicator
    torch.comm.get_local_node_communicator
    torch.comm.get_local_device_communicator
    torch.tensor.WholeMemoryTensor
    torch.tensor.create_wholememory_tensor
    torch.tensor.create_wholememory_tensor_from_filelist
    torch.tensor.destroy_wholememory_tensor
    torch.embedding.WholeMemoryOptimizer
    torch.embedding.create_wholememory_optimizer
    torch.embedding.destroy_wholememory_optimizer
    torch.embedding.WholeMemoryCachePolicy
    torch.embedding.create_wholememory_cache_policy
    torch.embedding.create_builtin_cache_policy
    torch.embedding.destroy_wholememory_cache_policy
    torch.embedding.WholeMemoryEmbedding
    torch.embedding.create_embedding
    torch.embedding.create_embedding_from_filelist
    torch.embedding.destroy_embedding
    torch.embedding.WholeMemoryEmbeddingModule
    torch.graph_structure.GraphStructure
