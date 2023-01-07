# Copyright (c) 2018-2023, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil

from setuptools import find_packages, Command
from skbuild import setup

from setuputils import get_environment_option

import versioneer


INSTALL_REQUIRES = []

CUDA_HOME = get_environment_option("CUDA_HOME")

if not CUDA_HOME:
    path_to_cuda_gdb = shutil.which("cuda-gdb")
    if path_to_cuda_gdb is None:
        raise OSError(
            "Could not locate CUDA. "
            "Please set the environment variable "
            "CUDA_HOME to the path to the CUDA installation "
            "and try again."
        )
    CUDA_HOME = os.path.dirname(os.path.dirname(path_to_cuda_gdb))

if not os.path.isdir(CUDA_HOME):
    raise OSError("Invalid CUDA_HOME: " "directory does not exist: {CUDA_HOME}")


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    user_options = [
        ("all", None, None),
    ]

    def initialize_options(self):
        self.all = None

    def finalize_options(self):
        pass

    def run(self):
        setupFileDir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(setupFileDir)
        os.system("rm -rf build")
        os.system("rm -rf dist")
        os.system("rm -rf dask-worker-space")
        os.system('find . -name "__pycache__" -type d -exec rm -rf {} +')
        os.system("rm -rf *.egg-info")
        os.system("rm -rf _skbuild")


cmdclass = versioneer.get_cmdclass()
cmdclass["clean"] = CleanCommand
# FIXME possibly remove this since there is no Cython code in cugraph_pyg
PACKAGE_DATA = {key: ["*.pxd"] for key in find_packages(include=["cugraph_pyg*"])}


setup(
    name="cugraph_pyg",
    description="cugraph_pyg - PyG support for cuGraph massive-scale,"
    " ultra-fast GPU graph analytics.",
    version=versioneer.get_version(),
    classifiers=[
        # "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        # "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    # Include the separately-compiled shared library
    author="NVIDIA Corporation",
    setup_requires=[],
    packages=find_packages(include=["cugraph_pyg", "cugraph_pyg.*"]),
    package_data=PACKAGE_DATA,
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    license="Apache",
    cmdclass=cmdclass,
    zip_safe=False,
)
