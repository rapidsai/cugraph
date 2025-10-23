# SPDX-FileCopyrightText: Copyright (c) 2019-2021, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cugraph.utilities import grmat_wrapper


def grmat_gen(argv):
    vertices, edges, source_col, dest_col = grmat_wrapper.grmat_gen(argv)

    return vertices, edges, source_col, dest_col
