# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import importlib.resources

# Read VERSION file from the module that is symlinked to VERSION file
# in the root of the repo at build time or copied to the moudle at
# installation. VERSION is a separate file that allows CI build-time scripts
# to update version info (including commit hashes) without modifying
# source files.
__version__ = (
    importlib.resources.files(__package__).joinpath("VERSION").read_text().strip()
)
try:
    __git_commit__ = (
        importlib.resources.files(__package__)
        .joinpath("GIT_COMMIT")
        .read_text()
        .strip()
    )
except FileNotFoundError:
    __git_commit__ = ""

__all__ = ["__git_commit__", "__version__"]
