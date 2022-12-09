#
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
#
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
#

import numpy as np
from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr, make_unique
from libcpp.utility cimport move, pair
from libcpp.vector cimport vector

import cudf
from rmm._lib.device_buffer cimport device_buffer

from pylibraft.common.handle cimport handle_t
from cugraph.structure.graph_utilities cimport (shuffled_vertices_t,
                                                major_minor_weights_t,
                                                renum_tuple_t,
                                                call_shuffle,
                                                call_renumber,
                                                )
from cugraph.structure.graph_primtypes cimport move_device_buffer_to_series


cdef renumber_helper(shuffled_vertices_t* ptr_maj_min_w, vertex_t, weights):
    # extract shuffled result:
    #
    cdef pair[unique_ptr[device_buffer], size_t] pair_s_major   = deref(ptr_maj_min_w).get_major_wrap()
    cdef pair[unique_ptr[device_buffer], size_t] pair_s_minor   = deref(ptr_maj_min_w).get_minor_wrap()
    cdef pair[unique_ptr[device_buffer], size_t] pair_s_weights = deref(ptr_maj_min_w).get_weights_wrap()

    shuffled_major_series = move_device_buffer_to_series(
        move(pair_s_major.first), vertex_t, "shuffled_major")

    shuffled_minor_series = move_device_buffer_to_series(
        move(pair_s_minor.first), vertex_t, "shuffled_minor")

    shuffled_df = cudf.DataFrame()
    # Some workers might have no data therefore ensure the empty column have the appropriate
    # vertex_t or weight_t. Failing to do that will create am empty column of type object
    # which is not supported by '__cuda_array_interface__'
    if shuffled_major_series is None:
        shuffled_df['major_vertices'] = cudf.Series(dtype=vertex_t)
    else:
        shuffled_df['major_vertices']= shuffled_major_series
    if shuffled_minor_series is None:
        shuffled_df['minor_vertices'] = cudf.Series(dtype=vertex_t)
    else:
        shuffled_df['minor_vertices']= shuffled_minor_series

    if weights is not None:
        weight_t = weights.dtype
        shuffled_weights_series = move_device_buffer_to_series(
            move(pair_s_weights.first), weight_t, "shuffled_weights")
        if shuffled_weights_series is None:
            shuffled_df['value']= cudf.Series(dtype=weight_t)
        else:
            shuffled_df['value']= shuffled_weights_series

    return shuffled_df


def renumber(input_df,           # maybe use cpdef ?
             renumbered_src_col_name,
             renumbered_dst_col_name,
             num_global_edges,
             rank,
             handle,
             is_multi_gpu,
             transposed):
    """
    Call MNMG renumber
    """
    cdef size_t handle_size_t = <size_t>handle.getHandle()
    # TODO: get handle_t out of handle...
    handle_ptr = <handle_t*>handle_size_t

    # FIXME: call_shuffle currently works on major/minor while call_renumber is updated to work on
    # source/destination. We'd better update call_shuffle to work on source/destination as well to
    # avoid switching between major/minor & source/destination. Deferring this work at this moment
    # expecting this legacy code path will be replaced with the new pylibcugrpah & C API based path.

    if not transposed:
        major_vertices = input_df[renumbered_src_col_name]
        minor_vertices = input_df[renumbered_dst_col_name]
    else:
        major_vertices = input_df[renumbered_dst_col_name]
        minor_vertices = input_df[renumbered_src_col_name]

    cdef uintptr_t c_edge_weights = <uintptr_t>NULL # set below...

    vertex_t = major_vertices.dtype
    if num_global_edges > (2**31 - 1):
        edge_t = np.dtype("int64")
    else:
        edge_t = vertex_t
    if "value" in input_df.columns:
        weights = input_df['value']
        weight_t = weights.dtype
        c_edge_weights = weights.__cuda_array_interface__['data'][0]
    else:
        weights = None
        weight_t = np.dtype("float32")

    if (vertex_t != np.dtype("int32") and vertex_t != np.dtype("int64")):
        raise Exception("Incorrect vertex_t type.")
    if (edge_t != np.dtype("int32") and edge_t != np.dtype("int64")):
        raise Exception("Incorrect edge_t type.")
    if (weight_t != np.dtype("float32") and weight_t != np.dtype("float64")):
        raise Exception("Incorrect weight_t type.")
    if (vertex_t != np.dtype("int32") and edge_t != np.dtype("int64")):
        raise Exception("Incompatible vertex_t and edge_t types.")

    # FIXME: needs to be edge_t type not int
    cdef int num_local_edges = len(major_vertices)

    cdef uintptr_t c_major_vertices = major_vertices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_minor_vertices = minor_vertices.__cuda_array_interface__['data'][0]

    cdef uintptr_t shuffled_src = <uintptr_t>NULL
    cdef uintptr_t shuffled_dst = <uintptr_t>NULL

    # FIXME: Fix fails when do_check = True
    cdef bool do_check = False # ? for now...
    cdef bool mg_flag = is_multi_gpu # run Single-GPU or MNMG

    cdef pair[unique_ptr[device_buffer], size_t] pair_original

    # tparams: vertex_t, edge_t, weight_t:
    #
    cdef unique_ptr[major_minor_weights_t[int, int, float]] ptr_shuffled_32_32_32
    cdef unique_ptr[major_minor_weights_t[int, int, double]] ptr_shuffled_32_32_64
    cdef unique_ptr[major_minor_weights_t[int, long, float]] ptr_shuffled_32_64_32
    cdef unique_ptr[major_minor_weights_t[int, long, double]] ptr_shuffled_32_64_64
    cdef unique_ptr[major_minor_weights_t[long, long, float]] ptr_shuffled_64_64_32
    cdef unique_ptr[major_minor_weights_t[long, long, double]] ptr_shuffled_64_64_64

    # tparams: vertex_t, edge_t:
    #
    cdef unique_ptr[renum_tuple_t[int, int]] ptr_renum_tuple_32_32
    cdef unique_ptr[renum_tuple_t[int, long]] ptr_renum_tuple_32_64
    cdef unique_ptr[renum_tuple_t[long, long]] ptr_renum_tuple_64_64

    # tparam: vertex_t:
    #
    cdef unique_ptr[vector[int]] edge_counts_32
    cdef unique_ptr[vector[long]] edge_counts_64

    # tparam: vertex_t:
    #
    cdef unique_ptr[vector[int]] uniq_partition_vector_32
    cdef unique_ptr[vector[long]] uniq_partition_vector_64

    # tparam: vertex_t:
    #
    cdef unique_ptr[vector[int]] uniq_segment_vector_32
    cdef unique_ptr[vector[long]] uniq_segment_vector_64

    cdef size_t rank_indx = <size_t>rank

    if (vertex_t == np.dtype("int32")):
        if ( edge_t == np.dtype("int32")):
            if( weight_t == np.dtype("float32")):
                if(is_multi_gpu):
                    ptr_shuffled_32_32_32.reset(call_shuffle[int, int, float](deref(handle_ptr),
                                                                           <int*>c_major_vertices,
                                                                           <int*>c_minor_vertices,
                                                                           <float*>c_edge_weights,
                                                                           num_local_edges,
                                                                           weights is not None).release())
                    shuffled_df = renumber_helper(ptr_shuffled_32_32_32.get(), vertex_t, weights)
                    major_vertices = shuffled_df['major_vertices']
                    minor_vertices = shuffled_df['minor_vertices']
                    num_local_edges = len(shuffled_df)
                    if not transposed:
                        major = renumbered_src_col_name; minor = renumbered_dst_col_name
                    else:
                        major = renumbered_dst_col_name; minor = renumbered_src_col_name
                    shuffled_df = shuffled_df.rename(columns={'major_vertices':major, 'minor_vertices':minor}, copy=False)
                    edge_counts_32 = move(ptr_shuffled_32_32_32.get().get_edge_counts_wrap())
                else:
                    shuffled_df = input_df
                    edge_counts_32 = make_unique[vector[int]](1, num_local_edges)

                if not transposed:
                    shuffled_src = major_vertices.__cuda_array_interface__['data'][0]
                    shuffled_dst = minor_vertices.__cuda_array_interface__['data'][0]
                else:
                    shuffled_src = minor_vertices.__cuda_array_interface__['data'][0]
                    shuffled_dst = major_vertices.__cuda_array_interface__['data'][0]

                ptr_renum_tuple_32_32.reset(call_renumber[int, int](deref(handle_ptr),
                                                                    <int*>shuffled_src,
                                                                    <int*>shuffled_dst,
                                                                    deref(edge_counts_32.get()),
                                                                    transposed,
                                                                    do_check,
                                                                    mg_flag).release())

                pair_original = ptr_renum_tuple_32_32.get().get_dv_wrap() # original vertices: see helper

                original_series = move_device_buffer_to_series(
                    move(pair_original.first), vertex_t, "original")

                # extract unique_ptr[partition_offsets]:
                #
                uniq_partition_vector_32 = move(ptr_renum_tuple_32_32.get().get_partition_offsets_wrap())

                # create series out of a partition range from rank to rank+1:
                #
                if is_multi_gpu:
                    new_series = cudf.Series(np.arange(uniq_partition_vector_32.get()[0].at(rank_indx),
                                                       uniq_partition_vector_32.get()[0].at(rank_indx+1)),
                                             dtype=vertex_t)
                else:
                    new_series = cudf.Series(np.arange(0, ptr_renum_tuple_32_32.get().get_num_vertices()),
                                             dtype=vertex_t)
                # create new cudf df
                #
                # and add the previous series to it:
                #
                renumbered_map = cudf.DataFrame()
                renumbered_map['original_ids'] = original_series
                renumbered_map['new_ids'] = new_series

                uniq_segment_vector_32 = move(ptr_renum_tuple_32_32.get().get_segment_offsets_wrap())
                segment_offsets = [None] * <Py_ssize_t>(deref(uniq_segment_vector_32).size())
                for i in range(len(segment_offsets)):
                  segment_offsets[i] = deref(uniq_segment_vector_32)[i]

                return renumbered_map, segment_offsets, shuffled_df

            elif( weight_t == np.dtype("float64")):
                if(is_multi_gpu):
                    ptr_shuffled_32_32_64.reset(call_shuffle[int, int, double](deref(handle_ptr),
                                                                            <int*>c_major_vertices,
                                                                            <int*>c_minor_vertices,
                                                                            <double*>c_edge_weights,
                                                                            num_local_edges,
                                                                            weights is not None).release())

                    shuffled_df = renumber_helper(ptr_shuffled_32_32_64.get(), vertex_t, weights)
                    major_vertices = shuffled_df['major_vertices']
                    minor_vertices = shuffled_df['minor_vertices']
                    num_local_edges = len(shuffled_df)
                    if not transposed:
                        major = renumbered_src_col_name; minor = renumbered_dst_col_name
                    else:
                        major = renumbered_dst_col_name; minor = renumbered_src_col_name
                    shuffled_df = shuffled_df.rename(columns={'major_vertices':major, 'minor_vertices':minor}, copy=False)
                    edge_counts_32 = move(ptr_shuffled_32_32_64.get().get_edge_counts_wrap())
                else:
                    shuffled_df = input_df
                    edge_counts_32 = make_unique[vector[int]](1, num_local_edges)

                if not transposed:
                    shuffled_src = major_vertices.__cuda_array_interface__['data'][0]
                    shuffled_dst = minor_vertices.__cuda_array_interface__['data'][0]
                else:
                    shuffled_src = minor_vertices.__cuda_array_interface__['data'][0]
                    shuffled_dst = major_vertices.__cuda_array_interface__['data'][0]

                ptr_renum_tuple_32_32.reset(call_renumber[int, int](deref(handle_ptr),
                                                                    <int*>shuffled_src,
                                                                    <int*>shuffled_dst,
                                                                    deref(edge_counts_32.get()),
                                                                    transposed,
                                                                    do_check,
                                                                    mg_flag).release())

                pair_original = ptr_renum_tuple_32_32.get().get_dv_wrap() # original vertices: see helper

                original_series = move_device_buffer_to_series(
                    move(pair_original.first), vertex_t, "original")

                # extract unique_ptr[partition_offsets]:
                #
                uniq_partition_vector_32 = move(ptr_renum_tuple_32_32.get().get_partition_offsets_wrap())

                # create series out of a partition range from rank to rank+1:
                #
                if is_multi_gpu:
                    new_series = cudf.Series(np.arange(uniq_partition_vector_32.get()[0].at(rank_indx),
                                                       uniq_partition_vector_32.get()[0].at(rank_indx+1)),
                                             dtype=vertex_t)
                else:
                    new_series = cudf.Series(np.arange(0, ptr_renum_tuple_32_32.get().get_num_vertices()),
                                             dtype=vertex_t)

                # create new cudf df
                #
                # and add the previous series to it:
                #
                renumbered_map = cudf.DataFrame()
                renumbered_map['original_ids'] = original_series
                renumbered_map['new_ids'] = new_series

                uniq_segment_vector_32 = move(ptr_renum_tuple_32_32.get().get_segment_offsets_wrap())
                segment_offsets = [None] * <Py_ssize_t>(deref(uniq_segment_vector_32).size())
                for i in range(len(segment_offsets)):
                  segment_offsets[i] = deref(uniq_segment_vector_32)[i]

                return renumbered_map, segment_offsets, shuffled_df

        elif ( edge_t == np.dtype("int64")):
            if( weight_t == np.dtype("float32")):
                if(is_multi_gpu):
                    ptr_shuffled_32_64_32.reset(call_shuffle[int, long, float](deref(handle_ptr),
                                                                            <int*>c_major_vertices,
                                                                            <int*>c_minor_vertices,
                                                                            <float*>c_edge_weights,
                                                                            num_local_edges,
                                                                            weights is not None).release())

                    shuffled_df = renumber_helper(ptr_shuffled_32_64_32.get(), vertex_t, weights)
                    major_vertices = shuffled_df['major_vertices']
                    minor_vertices = shuffled_df['minor_vertices']
                    num_local_edges = len(shuffled_df)
                    if not transposed:
                        major = renumbered_src_col_name; minor = renumbered_dst_col_name
                    else:
                        major = renumbered_dst_col_name; minor = renumbered_src_col_name
                    shuffled_df = shuffled_df.rename(columns={'major_vertices':major, 'minor_vertices':minor}, copy=False)
                    edge_counts_64 = move(ptr_shuffled_32_64_32.get().get_edge_counts_wrap())
                else:
                    shuffled_df = input_df
                    edge_counts_64 = make_unique[vector[long]](1, num_local_edges)

                if not transposed:
                    shuffled_src = major_vertices.__cuda_array_interface__['data'][0]
                    shuffled_dst = minor_vertices.__cuda_array_interface__['data'][0]
                else:
                    shuffled_src = minor_vertices.__cuda_array_interface__['data'][0]
                    shuffled_dst = major_vertices.__cuda_array_interface__['data'][0]

                ptr_renum_tuple_32_64.reset(call_renumber[int, long](deref(handle_ptr),
                                                                     <int*>shuffled_src,
                                                                     <int*>shuffled_dst,
                                                                     deref(edge_counts_64.get()),
                                                                     transposed,
                                                                     do_check,
                                                                     mg_flag).release())

                pair_original = ptr_renum_tuple_32_64.get().get_dv_wrap() # original vertices: see helper

                original_series = move_device_buffer_to_series(
                    move(pair_original.first), vertex_t, "original")

                # extract unique_ptr[partition_offsets]:
                #
                uniq_partition_vector_32 = move(ptr_renum_tuple_32_64.get().get_partition_offsets_wrap())

                # create series out of a partition range from rank to rank+1:
                #
                if is_multi_gpu:
                    new_series = cudf.Series(np.arange(uniq_partition_vector_32.get()[0].at(rank_indx),
                                                       uniq_partition_vector_32.get()[0].at(rank_indx+1)),
                                             dtype=vertex_t)
                else:
                    new_series = cudf.Series(np.arange(0, ptr_renum_tuple_32_64.get().get_num_vertices()),
                                             dtype=vertex_t)

                # create new cudf df
                #
                # and add the previous series to it:
                #
                renumbered_map = cudf.DataFrame()
                renumbered_map['original_ids'] = original_series
                renumbered_map['new_ids'] = new_series

                uniq_segment_vector_32 = move(ptr_renum_tuple_32_64.get().get_segment_offsets_wrap())
                segment_offsets = [None] * <Py_ssize_t>(deref(uniq_segment_vector_32).size())
                for i in range(len(segment_offsets)):
                  segment_offsets[i] = deref(uniq_segment_vector_32)[i]

                return renumbered_map, segment_offsets, shuffled_df
            elif( weight_t == np.dtype("float64")):
                if(is_multi_gpu):
                    ptr_shuffled_32_64_64.reset(call_shuffle[int, long, double](deref(handle_ptr),
                                                                             <int*>c_major_vertices,
                                                                             <int*>c_minor_vertices,
                                                                             <double*>c_edge_weights,
                                                                             num_local_edges,
                                                                             weights is not None).release())

                    shuffled_df = renumber_helper(ptr_shuffled_32_64_64.get(), vertex_t, weights)
                    major_vertices = shuffled_df['major_vertices']
                    minor_vertices = shuffled_df['minor_vertices']
                    num_local_edges = len(shuffled_df)
                    if not transposed:
                        major = renumbered_src_col_name; minor = renumbered_dst_col_name
                    else:
                        major = renumbered_dst_col_name; minor = renumbered_src_col_name
                    shuffled_df = shuffled_df.rename(columns={'major_vertices':major, 'minor_vertices':minor}, copy=False)
                    edge_counts_64 = move(ptr_shuffled_32_64_64.get().get_edge_counts_wrap())
                else:
                    shuffled_df = input_df
                    edge_counts_64 = make_unique[vector[long]](1, num_local_edges)

                if not transposed:
                    shuffled_src = major_vertices.__cuda_array_interface__['data'][0]
                    shuffled_dst = minor_vertices.__cuda_array_interface__['data'][0]
                else:
                    shuffled_src = minor_vertices.__cuda_array_interface__['data'][0]
                    shuffled_dst = major_vertices.__cuda_array_interface__['data'][0]

                ptr_renum_tuple_32_64.reset(call_renumber[int, long](deref(handle_ptr),
                                                                     <int*>shuffled_src,
                                                                     <int*>shuffled_dst,
                                                                     deref(edge_counts_64.get()),
                                                                     transposed,
                                                                     do_check,
                                                                     mg_flag).release())

                pair_original = ptr_renum_tuple_32_64.get().get_dv_wrap() # original vertices: see helper

                original_series = move_device_buffer_to_series(
                    move(pair_original.first), vertex_t, "original")

                # extract unique_ptr[partition_offsets]:
                #
                uniq_partition_vector_32 = move(ptr_renum_tuple_32_64.get().get_partition_offsets_wrap())

                # create series out of a partition range from rank to rank+1:
                #
                if is_multi_gpu:
                    new_series = cudf.Series(np.arange(uniq_partition_vector_32.get()[0].at(rank_indx),
                                                       uniq_partition_vector_32.get()[0].at(rank_indx+1)),
                                             dtype=vertex_t)
                else:
                    new_series = cudf.Series(np.arange(0, ptr_renum_tuple_32_64.get().get_num_vertices()),
                                             dtype=vertex_t)
                # create new cudf df
                #
                # and add the previous series to it:
                #
                renumbered_map = cudf.DataFrame()
                renumbered_map['original_ids'] = original_series
                renumbered_map['new_ids'] = new_series

                uniq_segment_vector_32 = move(ptr_renum_tuple_32_64.get().get_segment_offsets_wrap())
                segment_offsets = [None] * <Py_ssize_t>(deref(uniq_segment_vector_32).size())
                for i in range(len(segment_offsets)):
                  segment_offsets[i] = deref(uniq_segment_vector_32)[i]

                return renumbered_map, segment_offsets, shuffled_df

    elif (vertex_t == np.dtype("int64")):
        if ( edge_t == np.dtype("int64")):
            if( weight_t == np.dtype("float32")):
                if(is_multi_gpu):
                    ptr_shuffled_64_64_32.reset(call_shuffle[long, long, float](deref(handle_ptr),
                                                                            <long*>c_major_vertices,
                                                                            <long*>c_minor_vertices,
                                                                            <float*>c_edge_weights,
                                                                            num_local_edges,
                                                                            weights is not None).release())

                    shuffled_df = renumber_helper(ptr_shuffled_64_64_32.get(), vertex_t, weights)
                    major_vertices = shuffled_df['major_vertices']
                    minor_vertices = shuffled_df['minor_vertices']
                    num_local_edges = len(shuffled_df)
                    if not transposed:
                        major = renumbered_src_col_name; minor = renumbered_dst_col_name
                    else:
                        major = renumbered_dst_col_name; minor = renumbered_src_col_name
                    shuffled_df = shuffled_df.rename(columns={'major_vertices':major, 'minor_vertices':minor}, copy=False)
                    edge_counts_64 = move(ptr_shuffled_64_64_32.get().get_edge_counts_wrap())
                else:
                    shuffled_df = input_df
                    edge_counts_64 = make_unique[vector[long]](1, num_local_edges)

                if not transposed:
                    shuffled_src = major_vertices.__cuda_array_interface__['data'][0]
                    shuffled_dst = minor_vertices.__cuda_array_interface__['data'][0]
                else:
                    shuffled_src = minor_vertices.__cuda_array_interface__['data'][0]
                    shuffled_dst = major_vertices.__cuda_array_interface__['data'][0]

                ptr_renum_tuple_64_64.reset(call_renumber[long, long](deref(handle_ptr),
                                                                      <long*>shuffled_src,
                                                                      <long*>shuffled_dst,
                                                                      deref(edge_counts_64.get()),
                                                                      transposed,
                                                                      do_check,
                                                                      mg_flag).release())

                pair_original = ptr_renum_tuple_64_64.get().get_dv_wrap() # original vertices: see helper

                original_series = move_device_buffer_to_series(
                    move(pair_original.first), vertex_t, "original")

                # extract unique_ptr[partition_offsets]:
                #
                uniq_partition_vector_64 = move(ptr_renum_tuple_64_64.get().get_partition_offsets_wrap())

                # create series out of a partition range from rank to rank+1:
                #
                if is_multi_gpu:
                    new_series = cudf.Series(np.arange(uniq_partition_vector_64.get()[0].at(rank_indx),
                                                       uniq_partition_vector_64.get()[0].at(rank_indx+1)),
                                             dtype=vertex_t)
                else:
                    new_series = cudf.Series(np.arange(0, ptr_renum_tuple_64_64.get().get_num_vertices()),
                                             dtype=vertex_t)

                # create new cudf df
                #
                # and add the previous series to it:
                #
                renumbered_map = cudf.DataFrame()
                renumbered_map['original_ids'] = original_series
                renumbered_map['new_ids'] = new_series

                uniq_segment_vector_64 = move(ptr_renum_tuple_64_64.get().get_segment_offsets_wrap())
                segment_offsets = [None] * <Py_ssize_t>(deref(uniq_segment_vector_64).size())
                for i in range(len(segment_offsets)):
                  segment_offsets[i] = deref(uniq_segment_vector_64)[i]

                return renumbered_map, segment_offsets, shuffled_df

            elif( weight_t == np.dtype("float64")):
                if(is_multi_gpu):
                    ptr_shuffled_64_64_64.reset(call_shuffle[long, long, double](deref(handle_ptr),
                                                                              <long*>c_major_vertices,
                                                                              <long*>c_minor_vertices,
                                                                              <double*>c_edge_weights,
                                                                              num_local_edges,
                                                                              weights is not None).release())

                    shuffled_df = renumber_helper(ptr_shuffled_64_64_64.get(), vertex_t, weights)
                    major_vertices = shuffled_df['major_vertices']
                    minor_vertices = shuffled_df['minor_vertices']
                    num_local_edges = len(shuffled_df)
                    if not transposed:
                        major = renumbered_src_col_name; minor = renumbered_dst_col_name
                    else:
                        major = renumbered_dst_col_name; minor = renumbered_src_col_name
                    shuffled_df = shuffled_df.rename(columns={'major_vertices':major, 'minor_vertices':minor}, copy=False)
                    edge_counts_64 = move(ptr_shuffled_64_64_64.get().get_edge_counts_wrap())
                else:
                    shuffled_df = input_df
                    edge_counts_64 = make_unique[vector[long]](1, num_local_edges)

                if not transposed:
                    shuffled_src = major_vertices.__cuda_array_interface__['data'][0]
                    shuffled_dst = minor_vertices.__cuda_array_interface__['data'][0]
                else:
                    shuffled_src = minor_vertices.__cuda_array_interface__['data'][0]
                    shuffled_dst = major_vertices.__cuda_array_interface__['data'][0]

                ptr_renum_tuple_64_64.reset(call_renumber[long, long](deref(handle_ptr),
                                                                      <long*>shuffled_src,
                                                                      <long*>shuffled_dst,
                                                                      deref(edge_counts_64.get()),
                                                                      transposed,
                                                                      do_check,
                                                                      mg_flag).release())

                pair_original = ptr_renum_tuple_64_64.get().get_dv_wrap() # original vertices: see helper

                original_series = move_device_buffer_to_series(
                    move(pair_original.first), vertex_t, "original")

                # extract unique_ptr[partition_offsets]:
                #
                uniq_partition_vector_64 = move(ptr_renum_tuple_64_64.get().get_partition_offsets_wrap())

                # create series out of a partition range from rank to rank+1:
                #
                if is_multi_gpu:
                    new_series = cudf.Series(np.arange(uniq_partition_vector_64.get()[0].at(rank_indx),
                                                       uniq_partition_vector_64.get()[0].at(rank_indx+1)),
                                             dtype=vertex_t)
                else:
                    new_series = cudf.Series(np.arange(0, ptr_renum_tuple_64_64.get().get_num_vertices()),
                                             dtype=vertex_t)

                # create new cudf df
                #
                # and add the previous series to it:
                #
                renumbered_map = cudf.DataFrame()
                renumbered_map['original_ids'] = original_series
                renumbered_map['new_ids'] = new_series

                uniq_segment_vector_64 = move(ptr_renum_tuple_64_64.get().get_segment_offsets_wrap())
                segment_offsets = [None] * <Py_ssize_t>(deref(uniq_segment_vector_64).size())
                for i in range(len(segment_offsets)):
                  segment_offsets[i] = deref(uniq_segment_vector_64)[i]

                return renumbered_map, segment_offsets, shuffled_df
