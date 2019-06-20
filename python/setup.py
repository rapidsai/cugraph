# Copyright (c) 2018, NVIDIA CORPORATION.

from distutils.sysconfig import get_python_lib
import os
import sys

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import versioneer


INSTALL_REQUIRES = ['numba', 'cython']


try:
    NUMPY_INCLUDE = numpy.get_include()
except AttributeError:
    NUMPY_INCLUDE = numpy.get_numpy_include()

conda_include_dir = os.path.normpath(sys.prefix) + '/include'
CYTHON_FILES = ['cugraph/*.pyx']

if (os.environ.get('CONDA_PREFIX', None)):
    conda_prefix = os.environ.get('CONDA_PREFIX')
    conda_include_dir = conda_prefix + '/include'
    conda_lib_dir = conda_prefix + '/lib'

EXTENSIONS = [
    Extension("cugraph",
              sources=CYTHON_FILES,
              include_dirs=[NUMPY_INCLUDE,
                            conda_include_dir,
                            '../cpp/src',
                            '../cpp/include',
                            '../cpp/build/gunrock',
                            '../cpp/build/gunrock/externals/moderngpu/include',
                            '../cpp/build/gunrock/externals/cub'],
              library_dirs=[get_python_lib()],
              runtime_library_dirs=[conda_lib_dir],
              libraries=['cugraph', 'cudf'],
              language='c++',
              extra_compile_args=['-std=c++14'])
]

setup(name='cugraph',
      description="cuGraph - GPU Graph Analytics",
      version=versioneer.get_version(),
      classifiers=[
          # "Development Status :: 4 - Beta",
          "Intended Audience :: Developers",
          # "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7"
      ],
      # Include the separately-compiled shared library
      author="NVIDIA Corporation",
      setup_requires=['cython'],
      ext_modules=cythonize(EXTENSIONS),
      install_requires=INSTALL_REQUIRES,
      license="Apache",
      cmdclass=versioneer.get_cmdclass(),
      zip_safe=False)
