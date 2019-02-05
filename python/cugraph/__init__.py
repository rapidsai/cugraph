# Copyright (c) 2018, NVIDIA CORPORATION.
# Versioneer
import cugraph

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
