from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import os
import sys

from distutils.sysconfig import get_python_lib

install_requires = [
    'numpy',
    'cython'
]

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# temporary fix. cudf 0.5 will have a cudf.get_include()
cudf_include = os.path.normpath(sys.prefix) + '/include'

cython_files = ['python/cugraph.pyx']

extensions = [
    Extension("cugraph",
              sources=cython_files,
              include_dirs=[numpy_include,
                            cudf_include,
                            'src',
                            'include',
                            '../gunrock',
                            '../gunrock/externals/moderngpu/include',
                            '../gunrock/externals/cub'],
              library_dirs=[get_python_lib()],
              libraries=['cugraph', 'cudf'],
              language='c++',
              extra_compile_args=['-std=c++11'])
]

setup(name='cugraph',
      description='cuGraph - RAPIDS Graph Analytic Algorithms',
      author='NVIDIA Corporation',
      # todo: Add support for versioneer
      version='0.1',
      ext_modules=cythonize(extensions),
      install_requires=install_requires,
      license="Apache",
      zip_safe=False)
