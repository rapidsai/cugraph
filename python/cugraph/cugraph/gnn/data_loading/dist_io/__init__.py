# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings

from .reader import DEPRECATED__BufferedSampleReader, DEPRECATED__DistSampleReader
from .writer import DEPRECATED__DistSampleWriter


def BufferedSampleReader(*args, **kwargs):
    warnings.warn(
        "BufferedSampleReader is deprecated and will be removed in a future release.  Please migrate to the distributed sampling API in cuGraph-PyG.",
        FutureWarning,
    )
    return DEPRECATED__BufferedSampleReader(*args, **kwargs)


def DistSampleReader(*args, **kwargs):
    warnings.warn(
        "DistSampleReader is deprecated and will be removed in a future release.  Please migrate to the distributed sampling API in cuGraph-PyG.",
        FutureWarning,
    )
    return DEPRECATED__DistSampleReader(*args, **kwargs)


def DistSampleWriter(*args, **kwargs):
    warnings.warn(
        "DistSampleWriter is deprecated and will be removed in a future release.  Please migrate to the distributed sampling API in cuGraph-PyG.",
        FutureWarning,
    )
    return DEPRECATED__DistSampleWriter(*args, **kwargs)
