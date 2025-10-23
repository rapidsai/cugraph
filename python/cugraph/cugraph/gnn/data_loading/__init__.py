# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cugraph.gnn.data_loading.dist_sampler import (
    NeighborSampler,
    DistSampler,
)
from cugraph.gnn.data_loading.dist_io import (
    DistSampleWriter,
    DistSampleReader,
    BufferedSampleReader,
)


def UniformNeighborSampler(*args, **kwargs):
    return NeighborSampler(
        *args,
        **kwargs,
        biased=False,
    )


def BiasedNeighborSampler(*args, **kwargs):
    return NeighborSampler(
        *args,
        **kwargs,
        biased=True,
    )
