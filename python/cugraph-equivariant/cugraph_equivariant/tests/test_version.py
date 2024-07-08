# Copyright (c) 2024, NVIDIA CORPORATION.

import cugraph_equivariant


def test_version_constants_are_populated():
    # __git_commit__ will only be non-empty in a built distribution
    assert isinstance(cugraph_equivariant.__git_commit__, str)

    # __version__ should always be non-empty
    assert isinstance(cugraph_equivariant.__version__, str)
    assert len(cugraph_equivariant.__version__) > 0
