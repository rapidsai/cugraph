# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# from cugraph.utilities.grmat import grmat_gen
# from cugraph.utilities.pointer_utils import device_of_gpu_pointer

from cugraph.utilities.utils import (
    import_optional,
    ensure_cugraph_obj,
    is_matrix_type,
    is_cp_matrix_type,
    is_sp_matrix_type,
    renumber_vertex_pair,
    cupy_package,
    ensure_valid_dtype,
)
from cugraph.utilities.path_retrieval import get_traversed_cost
