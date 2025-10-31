# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Exception classes for cugraph.
"""


class FailedToConvergeError(Exception):
    """
    Raised when an algorithm fails to converge within a predetermined set of
    constraints which vary based on the algorithm, and may or may not be
    user-configurable.
    """

    pass
