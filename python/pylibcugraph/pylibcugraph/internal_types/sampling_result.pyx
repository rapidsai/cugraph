# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3


from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
)
from pylibcugraph._cugraph_c.algorithms cimport (
    cugraph_sample_result_t,
    cugraph_sample_result_get_major_offsets,
    cugraph_sample_result_get_majors,
    cugraph_sample_result_get_minors,
    cugraph_sample_result_get_label_hop_offsets,
    cugraph_sample_result_get_label_type_hop_offsets,
    cugraph_sample_result_get_sources, # deprecated
    cugraph_sample_result_get_destinations, # deprecated
    cugraph_sample_result_get_edge_weight,
    cugraph_sample_result_get_edge_id,
    cugraph_sample_result_get_edge_type,
    cugraph_sample_result_get_hop, # deprecated
    cugraph_sample_result_get_start_labels,
    cugraph_sample_result_get_offsets, # deprecated
    cugraph_sample_result_get_renumber_map,
    cugraph_sample_result_get_renumber_map_offsets,
    cugraph_sample_result_get_edge_renumber_map,
    cugraph_sample_result_get_edge_renumber_map_offsets,
    cugraph_sample_result_get_edge_start_time,
    cugraph_sample_result_get_edge_end_time,
    cugraph_sample_result_free,
)
from pylibcugraph.utils cimport (
    create_cupy_array_view_for_device_ptr,
)


cdef class SamplingResult:
    """
    Cython interface to a cugraph_sample_result_t pointer. Instances of this
    call will take ownership of the pointer and free it under standard python
    GC rules (ie. when all references to it are no longer present).

    This class provides methods to return non-owning cupy ndarrays for the
    corresponding array members. Returning these cupy arrays increments the ref
    count on the SamplingResult instances from which the cupy arrays are
    referencing.
    """
    def __cinit__(self):
        """
        Initialize this object and allocate its underlying native resources.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Algorithm result returned by the backend binding.
        """
        # This SamplingResult instance owns sample_result_ptr now. It will be
        # freed when this instance is deleted (see __dealloc__())
        self.c_sample_result_ptr = NULL

    def __dealloc__(self):
        """
        Release native resources owned by this object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Algorithm result returned by the backend binding.
        """
        if self.c_sample_result_ptr is not NULL:
            cugraph_sample_result_free(self.c_sample_result_ptr)

    cdef set_ptr(self, cugraph_sample_result_t* sample_result_ptr):
        self.c_sample_result_ptr = sample_result_ptr

    def get_major_offsets(self):
        """
        Return major offsets from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: create_cupy_array_view_for_device_ptr(device_array_view_ptr,.
        """
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")

        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_major_offsets(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_majors(self):
        """
        Return majors from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: create_cupy_array_view_for_device_ptr(device_array_view_ptr,.
        """
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_majors(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_minors(self):
        """
        Return minors from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: create_cupy_array_view_for_device_ptr(device_array_view_ptr,.
        """
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_minors(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_sources(self):
        """
        Return sources from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: create_cupy_array_view_for_device_ptr(device_array_view_ptr,.
        """
        # Deprecated
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_sources(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_destinations(self):
        """
        Return destinations from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: create_cupy_array_view_for_device_ptr(device_array_view_ptr,.
        """
        # Deprecated
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_destinations(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_edge_weights(self):
        """
        Return edge weights from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: create_cupy_array_view_for_device_ptr(device_array_view_ptr,.
        """
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_edge_weight(self.c_sample_result_ptr)
        )

        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_indices(self):
        """
        Return indices from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: self.get_edge_weights().
        """
        # Deprecated
        return self.get_edge_weights()

    def get_edge_ids(self):
        """
        Return edge ids from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: create_cupy_array_view_for_device_ptr(device_array_view_ptr,.
        """
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_edge_id(self.c_sample_result_ptr)
        )

        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_edge_types(self):
        """
        Return edge types from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: create_cupy_array_view_for_device_ptr(device_array_view_ptr,.
        """
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_edge_type(self.c_sample_result_ptr)
        )

        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_edge_start_time(self):
        """
        Return edge start time from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: create_cupy_array_view_for_device_ptr(device_array_view_ptr,.
        """
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_edge_start_time(self.c_sample_result_ptr)
        )

        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_edge_end_time(self):
        """
        Return edge end time from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: create_cupy_array_view_for_device_ptr(device_array_view_ptr,.
        """
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_edge_end_time(self.c_sample_result_ptr)
        )

        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_batch_ids(self):
        """
        Return batch ids from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: create_cupy_array_view_for_device_ptr(device_array_view_ptr,.
        """
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_start_labels(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_label_hop_offsets(self):
        """
        Return label hop offsets from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: create_cupy_array_view_for_device_ptr(device_array_view_ptr,.
        """
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_label_hop_offsets(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_label_type_hop_offsets(self):
        """
        Return label type hop offsets from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: create_cupy_array_view_for_device_ptr(device_array_view_ptr,.
        """
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_label_type_hop_offsets(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    # Deprecated
    def get_offsets(self):
        """
        Return offsets from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: create_cupy_array_view_for_device_ptr(device_array_view_ptr,.
        """
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_offsets(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    # Deprecated
    def get_hop_ids(self):
        """
        Return hop ids from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: create_cupy_array_view_for_device_ptr(device_array_view_ptr,.
        """
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_hop(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_renumber_map(self):
        """
        Return renumber map from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: create_cupy_array_view_for_device_ptr(device_array_view_ptr,.
        """
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_renumber_map(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_renumber_map_offsets(self):
        """
        Return renumber map offsets from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: create_cupy_array_view_for_device_ptr(device_array_view_ptr,.
        """
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_renumber_map_offsets(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)


    def get_edge_renumber_map(self):
        """
        Return edge renumber map from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: create_cupy_array_view_for_device_ptr(device_array_view_ptr,.
        """
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_edge_renumber_map(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)

    def get_edge_renumber_map_offsets(self):
        """
        Return edge renumber map offsets from this result object.
        
        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        
        Returns
        -------
        object
                Output value: create_cupy_array_view_for_device_ptr(device_array_view_ptr,.
        """
        if self.c_sample_result_ptr is NULL:
            raise ValueError("pointer not set, must call set_ptr() with a "
                             "non-NULL value first.")
        cdef cugraph_type_erased_device_array_view_t* device_array_view_ptr = (
            cugraph_sample_result_get_edge_renumber_map_offsets(self.c_sample_result_ptr)
        )
        if device_array_view_ptr is NULL:
            return None

        return create_cupy_array_view_for_device_ptr(device_array_view_ptr,
                                                     self)
