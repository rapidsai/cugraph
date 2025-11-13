# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings

from cugraph.gnn.data_loading.dist_sampler import (
    DEPRECATED__NeighborSampler,
    DEPRECATED__DistSampler,
)
from cugraph.gnn.data_loading.dist_io import (
    DistSampleWriter,
    DistSampleReader,
    BufferedSampleReader,
)


def DistSampler(*args, **kwargs):
    warnings.warn(
        FutureWarning,
        "DistSampler is deprecated and will be removed in a future release.  Please migrate to the distributed sampling API in cuGraph-PyG.",
    )
    return DEPRECATED__DistSampler(*args, **kwargs)


def NeighborSampler(*args, **kwargs):
    warnings.warn(
        FutureWarning,
        "NeighborSampler is deprecated and will be removed in a future release.  Please migrate to the distributed sampling API in cuGraph-PyG.",
    )
    return DEPRECATED__NeighborSampler(*args, **kwargs)


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
