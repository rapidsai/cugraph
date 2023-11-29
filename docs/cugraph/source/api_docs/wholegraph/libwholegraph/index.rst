=====================
libwholegraph API doc
=====================

Doxygen WholeGraph C API documentation
--------------------------------------
For doxygen documentation, please refer to `Doxygen Documentation <../../doxygen_docs/libwholegraph/html/index.html>`_

WholeGraph C API documentation
------------------------------

Library Level APIs
++++++++++++++++++

.. doxygenenum:: wholememory_error_code_t
    :project: libwholegraph
.. doxygenfunction:: wholememory_init
    :project: libwholegraph
.. doxygenfunction:: wholememory_finalize
    :project: libwholegraph
.. doxygenfunction:: fork_get_device_count
    :project: libwholegraph

WholeMemory Communicator APIs
+++++++++++++++++++++++++++++

.. doxygentypedef:: wholememory_comm_t
    :project: libwholegraph
.. doxygenstruct:: wholememory_unique_id_t
    :project: libwholegraph
.. doxygenfunction:: wholememory_create_unique_id
    :project: libwholegraph
.. doxygenfunction:: wholememory_create_communicator
    :project: libwholegraph
.. doxygenfunction:: wholememory_destroy_communicator
    :project: libwholegraph
.. doxygenfunction:: wholememory_communicator_get_rank
    :project: libwholegraph
.. doxygenfunction:: wholememory_communicator_get_size
    :project: libwholegraph
.. doxygenfunction:: wholememory_communicator_barrier
    :project: libwholegraph

WholeMemoryHandle APIs
++++++++++++++++++++++

.. doxygenenum:: wholememory_memory_type_t
    :project: libwholegraph
.. doxygenenum:: wholememory_memory_location_t
    :project: libwholegraph
.. doxygentypedef:: wholememory_handle_t
    :project: libwholegraph
.. doxygenstruct:: wholememory_gref_t
    :project: libwholegraph
.. doxygenfunction:: wholememory_malloc
    :project: libwholegraph
.. doxygenfunction:: wholememory_free
    :project: libwholegraph
.. doxygenfunction:: wholememory_get_communicator
    :project: libwholegraph
.. doxygenfunction:: wholememory_get_memory_type
    :project: libwholegraph
.. doxygenfunction:: wholememory_get_memory_location
    :project: libwholegraph
.. doxygenfunction:: wholememory_get_total_size
    :project: libwholegraph
.. doxygenfunction:: wholememory_get_data_granularity
    :project: libwholegraph
.. doxygenfunction:: wholememory_get_local_memory
    :project: libwholegraph
.. doxygenfunction:: wholememory_get_rank_memory
    :project: libwholegraph
.. doxygenfunction:: wholememory_get_global_pointer
    :project: libwholegraph
.. doxygenfunction:: wholememory_get_global_reference
    :project: libwholegraph
.. doxygenfunction:: wholememory_determine_partition_plan
    :project: libwholegraph
.. doxygenfunction:: wholememory_determine_entry_partition_plan
    :project: libwholegraph
.. doxygenfunction:: wholememory_get_partition_plan
    :project: libwholegraph
.. doxygenfunction:: wholememory_load_from_file
    :project: libwholegraph
.. doxygenfunction:: wholememory_store_to_file
    :project: libwholegraph

WholeMemoryTensor APIs
++++++++++++++++++++++

.. doxygenenum:: wholememory_dtype_t
    :project: libwholegraph
.. doxygenstruct:: wholememory_array_description_t
    :project: libwholegraph
.. doxygenstruct:: wholememory_matrix_description_t
    :project: libwholegraph
.. doxygenstruct:: wholememory_tensor_description_t
    :project: libwholegraph
.. doxygentypedef:: wholememory_tensor_t
    :project: libwholegraph
.. doxygenfunction:: wholememory_dtype_get_element_size
    :project: libwholegraph
.. doxygenfunction:: wholememory_dtype_is_floating_number
    :project: libwholegraph
.. doxygenfunction:: wholememory_dtype_is_integer_number
    :project: libwholegraph
.. doxygenfunction:: wholememory_create_array_desc
    :project: libwholegraph
.. doxygenfunction:: wholememory_create_matrix_desc
    :project: libwholegraph
.. doxygenfunction:: wholememory_initialize_tensor_desc
    :project: libwholegraph
.. doxygenfunction:: wholememory_copy_array_desc_to_matrix
    :project: libwholegraph
.. doxygenfunction:: wholememory_copy_array_desc_to_tensor
    :project: libwholegraph
.. doxygenfunction:: wholememory_copy_matrix_desc_to_tensor
    :project: libwholegraph
.. doxygenfunction:: wholememory_convert_tensor_desc_to_array
    :project: libwholegraph
.. doxygenfunction:: wholememory_convert_tensor_desc_to_matrix
    :project: libwholegraph
.. doxygenfunction:: wholememory_get_memory_element_count_from_array
    :project: libwholegraph
.. doxygenfunction:: wholememory_get_memory_size_from_array
    :project: libwholegraph
.. doxygenfunction:: wholememory_get_memory_element_count_from_matrix
    :project: libwholegraph
.. doxygenfunction:: wholememory_get_memory_size_from_matrix
    :project: libwholegraph
.. doxygenfunction:: wholememory_get_memory_element_count_from_tensor
    :project: libwholegraph
.. doxygenfunction:: wholememory_get_memory_size_from_tensor
    :project: libwholegraph
.. doxygenfunction:: wholememory_unsqueeze_tensor
    :project: libwholegraph
.. doxygenfunction:: wholememory_create_tensor
    :project: libwholegraph
.. doxygenfunction:: wholememory_destroy_tensor
    :project: libwholegraph
.. doxygenfunction:: wholememory_make_tensor_from_pointer
    :project: libwholegraph
.. doxygenfunction:: wholememory_make_tensor_from_handle
    :project: libwholegraph
.. doxygenfunction:: wholememory_tensor_has_handle
    :project: libwholegraph
.. doxygenfunction:: wholememory_tensor_get_memory_handle
    :project: libwholegraph
.. doxygenfunction:: wholememory_tensor_get_tensor_description
    :project: libwholegraph
.. doxygenfunction:: wholememory_tensor_get_global_reference
    :project: libwholegraph
.. doxygenfunction:: wholememory_tensor_map_local_tensor
    :project: libwholegraph
.. doxygenfunction:: wholememory_tensor_get_data_pointer
    :project: libwholegraph
.. doxygenfunction:: wholememory_tensor_get_entry_per_partition
    :project: libwholegraph
.. doxygenfunction:: wholememory_tensor_get_subtensor
    :project: libwholegraph
.. doxygenfunction:: wholememory_tensor_get_root
    :project: libwholegraph

Ops on WholeMemory Tensors
++++++++++++++++++++++++++

.. doxygenfunction:: wholememory_gather
    :project: libwholegraph
.. doxygenfunction:: wholememory_scatter
    :project: libwholegraph

WholeTensorEmbedding APIs
+++++++++++++++++++++++++

.. doxygentypedef:: wholememory_embedding_cache_policy_t
    :project: libwholegraph
.. doxygentypedef:: wholememory_embedding_optimizer_t
    :project: libwholegraph
.. doxygentypedef:: wholememory_embedding_t
    :project: libwholegraph
.. doxygenenum:: wholememory_access_type_t
    :project: libwholegraph
.. doxygenenum:: wholememory_optimizer_type_t
    :project: libwholegraph
.. doxygenfunction:: wholememory_create_embedding_optimizer
    :project: libwholegraph
.. doxygenfunction:: wholememory_optimizer_set_parameter
    :project: libwholegraph
.. doxygenfunction:: wholememory_destroy_embedding_optimizer
    :project: libwholegraph
.. doxygenfunction:: wholememory_create_embedding_cache_policy
    :project: libwholegraph
.. doxygenfunction:: wholememory_destroy_embedding_cache_policy
    :project: libwholegraph
.. doxygenfunction:: wholememory_create_embedding
    :project: libwholegraph
.. doxygenfunction:: wholememory_destroy_embedding
    :project: libwholegraph
.. doxygenfunction:: wholememory_embedding_get_embedding_tensor
    :project: libwholegraph
.. doxygenfunction:: wholememory_embedding_gather
    :project: libwholegraph
.. doxygenfunction:: wholememory_embedding_gather_gradient_apply
    :project: libwholegraph
.. doxygenfunction:: wholememory_embedding_get_optimizer_state_names
    :project: libwholegraph
.. doxygenfunction:: wholememory_embedding_get_optimizer_state
    :project: libwholegraph
.. doxygenfunction:: wholememory_embedding_writeback_cache
    :project: libwholegraph
.. doxygenfunction:: wholememory_embedding_drop_all_cache
    :project: libwholegraph

Ops on graphs stored in WholeMemory
+++++++++++++++++++++++++++++++++++

.. doxygenfunction:: wholegraph_csr_unweighted_sample_without_replacement
    :project: libwholegraph
.. doxygenfunction:: wholegraph_csr_weighted_sample_without_replacement
    :project: libwholegraph

Miscellaneous Ops for graph
+++++++++++++++++++++++++++

.. doxygenfunction:: graph_append_unique
    :project: libwholegraph
.. doxygenfunction:: csr_add_self_loop
    :project: libwholegraph
