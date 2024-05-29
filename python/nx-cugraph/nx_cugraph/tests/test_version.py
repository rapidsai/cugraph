# Copyright (c) 2024, NVIDIA CORPORATION.

import nx_cugraph
import _nx_cugraph
import pytest


@pytest.mark.parametrize("mod", [nx_cugraph, _nx_cugraph])
def test_version_constants_are_populated(mod):
    # __git_commit__ will only be non-empty in a built distribution
    assert isinstance(mod.__git_commit__, str)

    # __version__ should always be non-empty
    assert isinstance(mod.__version__, str)
    assert len(mod.__version__) > 0
