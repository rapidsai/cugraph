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


cuda_suffix = os.getenv("RAPIDS_PY_WHEEL_CUDA_SUFFIX", default="")

INSTALL_REQUIRES = [
    "numba",
    "dask-cuda",
    f"rmm{cuda_suffix}==23.2.*",
    f"cudf{cuda_suffix}==23.2.*",
    f"raft-dask{cuda_suffix}==23.2.*",
    f"dask-cudf{cuda_suffix}==23.2.*",
    f"pylibcugraph{cuda_suffix}==23.2.*",
    "cupy-cuda11x",
]

extras_require = {
    "test": [
        "pytest",
        "pytest-xdist",
        "pytest-benchmark",
        "scipy",
        "numpy",
        "pandas",
        "networkx>=2.5.1",
        "scikit-learn>=0.23.1",
        "python-louvain",
        # cudf will use fsspec but is protocol independent. cugraph tests
        # specifically require http for the test files it asks cudf to read.
        "fsspec[http]>=0.6.0",
    ]
}


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

PACKAGE_DATA = {key: ["*.pxd"] for key in find_packages(include=["cugraph*"])}

PACKAGE_DATA["cugraph.experimental.datasets"].extend(
    [
        "cugraph/experimental/datasets/metadata/*.yaml",
        "cugraph/experimental/datasets/*.yaml",
    ]
)


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
    name=f"cugraph{cuda_suffix}",
    description="cuGraph - RAPIDS GPU Graph Analytics",
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
    packages=find_packages(include=["cugraph", "cugraph.*"]),
    package_data=PACKAGE_DATA,
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    license="Apache 2.0",
    cmdclass=cmdclass,
    zip_safe=False,
    extras_require=extras_require,
)
