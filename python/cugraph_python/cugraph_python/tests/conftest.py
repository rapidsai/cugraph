# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test modules live in per-category subdirectories without __init__.py files. Pytest puts this
# directory on sys.path when it loads this file, which is what lets them
# ``from testing_utils import ...`` however the suite is invoked (source tree or installed wheel).
