# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

cdef class GraphProperties:
    """
    Class wrapper around C cugraph_graph_properties_t struct
    """
    def __cinit__(self, is_symmetric=False, is_multigraph=False):
        """
        Initialize this object and allocate its underlying native resources.

        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        is_symmetric : object
            Input argument `is_symmetric` passed to the backend algorithm.
        is_multigraph : object
            Input argument `is_multigraph` passed to the backend algorithm.

        Returns
        -------
        object
                Algorithm result returned by the backend binding.
        """
        self.c_graph_properties.is_symmetric = is_symmetric
        self.c_graph_properties.is_multigraph = is_multigraph

    # Pickle support methods: get args for __new__ (__cinit__), get/set state
    def __getnewargs_ex__(self):
        """
        Internal helper for  getnewargs ex  .

        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.

        Returns
        -------
        tuple
                Tuple containing: (), {"is_symmetric":is_symmetric, "is_multigraph":is_multigraph}.
        """
        is_symmetric = self.c_graph_properties.is_symmetric
        is_multigraph = self.c_graph_properties.is_multigraph
        return ((),{"is_symmetric":is_symmetric, "is_multigraph":is_multigraph})

    def __getstate__(self):
        """
        Internal helper for  getstate  .

        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.

        Returns
        -------
        object
                Output value: ().
        """
        return ()

    def __setstate__(self, state):
        """
        Internal helper for  setstate  .

        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        state : object
            Input argument `state` passed to the backend algorithm.

        Returns
        -------
        object
                Algorithm result returned by the backend binding.
        """
        pass

    @property
    def is_symmetric(self):
        """
        Execute is symmetric using the pylibcugraph backend.

        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.

        Returns
        -------
        object
                Output value: bool(self.c_graph_properties.is_symmetric).
        """
        return bool(self.c_graph_properties.is_symmetric)

    @is_symmetric.setter
    def is_symmetric(self, value):
        """
        Execute is symmetric using the pylibcugraph backend.

        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        value : object
            Input argument `value` passed to the backend algorithm.

        Returns
        -------
        object
                Algorithm result returned by the backend binding.
        """
        self.c_graph_properties.is_symmetric = value

    @property
    def is_multigraph(self):
        """
        Execute is multigraph using the pylibcugraph backend.

        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.

        Returns
        -------
        object
                Output value: bool(self.c_graph_properties.is_multigraph).
        """
        return bool(self.c_graph_properties.is_multigraph)

    @is_multigraph.setter
    def is_multigraph(self, value):
        """
        Execute is multigraph using the pylibcugraph backend.

        Parameters
        ----------
        self : object
            Input argument `self` passed to the backend algorithm.
        value : object
            Input argument `value` passed to the backend algorithm.

        Returns
        -------
        object
                Algorithm result returned by the backend binding.
        """
        self.c_graph_properties.is_multigraph = value
