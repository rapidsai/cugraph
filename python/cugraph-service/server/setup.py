# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

from setuptools import setup, find_packages
import versioneer

cmdclass = versioneer.get_cmdclass()

install_requires = [
    "cudf",
    "cugraph",
    "cugraph-service-client",
    "cupy-cuda11x",
    "dask-cuda",
    "dask-cudf",
    "distributed ==2023.1.1",
    "numpy",
    "rmm",
    "thriftpy2",
    "ucx-py",
]

setup(
    name="cugraph-service-server",
    description="cuGraph Service server",
    version=versioneer.get_version(),
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
    ],
    author="NVIDIA Corporation",
    url="https://github.com/rapidsai/cugraph",
    packages=find_packages(
        include=["cugraph_service_server", "cugraph_service_server.*"]
    ),
    entry_points={
        "console_scripts": [
            "cugraph-service-server=cugraph_service_server.__main__:main"
        ],
    },
    install_requires=install_requires,
    license="Apache",
    cmdclass=cmdclass,
    zip_safe=True,
)
