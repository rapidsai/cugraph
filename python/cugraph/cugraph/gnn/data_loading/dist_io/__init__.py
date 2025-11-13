# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings

from .reader import DEPRECATED__BufferedSampleReader, DEPRECATED__DistSampleReader
from .writer import DEPRECATED__DistSampleWriter


def BufferedSampleReader(*args, **kwargs):
    warnings.warn(
        FutureWarning,
        "BufferedSampleReader is deprecated and will be removed in a future release.  Please migrate to the distributed sampling API in cuGraph-PyG.",
    )
    return DEPRECATED__BufferedSampleReader(*args, **kwargs)


def DistSampleReader(*args, **kwargs):
    warnings.warn(
        FutureWarning,
        "DistSampleReader is deprecated and will be removed in a future release.  Please migrate to the distributed sampling API in cuGraph-PyG.",
    )
    return DEPRECATED__DistSampleReader(*args, **kwargs)


def DistSampleWriter(*args, **kwargs):
    warnings.warn(
        FutureWarning,
        "DistSampleWriter is deprecated and will be removed in a future release.  Please migrate to the distributed sampling API in cuGraph-PyG.",
    )
    return DEPRECATED__DistSampleWriter(*args, **kwargs)
