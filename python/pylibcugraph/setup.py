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

from setuptools import find_packages, Command
from skbuild import setup

import versioneer


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
        os.system('find . -name "*.cpp" -type f -delete')
        os.system('find . -name "*.cpython*.so" -type f -delete')
        os.system("rm -rf _skbuild")


cmdclass = versioneer.get_cmdclass()
cmdclass["clean"] = CleanCommand


def exclude_libcxx_symlink(cmake_manifest):
    return list(
        filter(
            lambda name: not ("include/rapids/libcxx/include" in name), cmake_manifest
        )
    )


cuda_suffix = os.getenv("RAPIDS_PY_WHEEL_CUDA_SUFFIX", default="")


# Ensure that wheel version patching works for nightlies.
if "RAPIDS_PY_WHEEL_VERSIONEER_OVERRIDE" in os.environ:
    orig_get_versions = versioneer.get_versions

    version_override = os.environ["RAPIDS_PY_WHEEL_VERSIONEER_OVERRIDE"]

    def get_versions():
        data = orig_get_versions()
        data["version"] = version_override
        return data

    versioneer.get_versions = get_versions


setup(
    name=f"pylibcugraph{cuda_suffix}",
    description="pylibcuGraph - RAPIDS GPU Graph Analytics",
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
    packages=find_packages(include=["pylibcugraph", "pylibcugraph.*"]),
    package_data={key: ["*.pxd"] for key in find_packages(include=["pylibcugraph*"])},
    include_package_data=True,
    install_requires=[
        f"pylibraft{cuda_suffix}==23.2.*",
        f"rmm{cuda_suffix}==23.2.*",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-xdist",
            "pytest-benchmark",
            "scipy",
            "pandas",
            "numpy",
            "networkx>=2.5.1",
            "scikit-learn>=0.23.1",
            "dask",
            "distributed",
            "dask-cuda",
            f"cudf{cuda_suffix}",
        ]
    },
    cmake_process_manifest_hook=exclude_libcxx_symlink,
    license="Apache 2.0",
    cmdclass=cmdclass,
    zip_safe=False,
)
