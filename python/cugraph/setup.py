# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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


INSTALL_REQUIRES = [
    "numba",
    f"rmm{os.getenv('PYTHON_PACKAGE_CUDA_SUFFIX', default='')}"
    f"pylibcugraph{os.getenv('PYTHON_PACKAGE_CUDA_SUFFIX', default='')}"
]

CUDA_HOME = get_environment_option('CUDA_HOME')

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
    raise OSError(
        "Invalid CUDA_HOME: " "directory does not exist: {CUDA_HOME}"
    )


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = [('all', None, None), ]

    def initialize_options(self):
        self.all = None

    def finalize_options(self):
        pass

    def run(self):
        setupFileDir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(setupFileDir)
        os.system('rm -rf build')
        os.system('rm -rf dist')
        os.system('rm -rf dask-worker-space')
        os.system('find . -name "__pycache__" -type d -exec rm -rf {} +')
        os.system('rm -rf *.egg-info')
        os.system('find . -name "*.cpp" -type f -delete')
        os.system('find . -name "*.cpython*.so" -type f -delete')
        os.system('rm -rf _skbuild')


cmdclass = versioneer.get_cmdclass()
cmdclass["clean"] = CleanCommand
PACKAGE_DATA = {
    key: ["*.pxd"] for key in find_packages(include=["cugraph*"])
}

PACKAGE_DATA['cugraph.experimental.datasets'].extend(
    ['cugraph/experimental/datasets/metadata/*.yaml',
     'cugraph/experimental/datasets/*.yaml'])


setup(name='cugraph'+os.getenv("PYTHON_PACKAGE_CUDA_SUFFIX", default=""),
      description="cuGraph - RAPIDS GPU Graph Analytics",
      version=versioneer.get_version(),
      author="NVIDIA Corporation",
      license="Apache",
      classifiers=[
          # "Development Status :: 4 - Beta",
          "Intended Audience :: Developers",
          # "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9"
      ],
      cmdclass=cmdclass,
      include_package_data=True,
      package_data=PACKAGE_DATA,
      packages=find_packages(include=['cugraph', 'cugraph.*']),
      setup_requires=[
        f"rmm{os.getenv('PYTHON_PACKAGE_CUDA_SUFFIX', default='')}",
        f"raft-dask{os.getenv('PYTHON_PACKAGE_CUDA_SUFFIX', default='')}",
        f"pylibcugraph{os.getenv('PYTHON_PACKAGE_CUDA_SUFFIX', default='')}",
      ],
      install_requires=[
        "numba",
        f"dask-cuda>=22.8",
        f"cudf{os.getenv('PYTHON_PACKAGE_CUDA_SUFFIX', default='')}",
        f"raft-dask{os.getenv('PYTHON_PACKAGE_CUDA_SUFFIX', default='')}",
        f"dask-cudf{os.getenv('PYTHON_PACKAGE_CUDA_SUFFIX', default='')}",
        f"pylibcugraph{os.getenv('PYTHON_PACKAGE_CUDA_SUFFIX', default='')}"
      ],
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
          ]
      },
      zip_safe=False)
