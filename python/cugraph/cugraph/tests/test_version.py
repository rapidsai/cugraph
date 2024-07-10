# Copyright (c) 2024, NVIDIA CORPORATION.

import cugraph


def test_version_constants_are_populated():
    # __git_commit__ will only be non-empty in a built distribution
    assert isinstance(cugraph.__git_commit__, str)

    # __version__ should always be non-empty
    assert isinstance(cugraph.__version__, str)
    assert len(cugraph.__version__) > 0
