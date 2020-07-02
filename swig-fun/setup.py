#-*- coding:utf-8 -*-
import sys
import os
from setuptools import setup, find_packages, Extension

from distutils import util
from distutils import sysconfig
from distutils.sysconfig import get_config_var

import os


# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


platform = util.get_platform()

# special settings for Linux
if platform.startswith("linux"):
    _UNWANTED_OPTS = frozenset(['-Wstrict-prototypes'])
    os.environ['OPT'] = ' '.join(
        _ for _ in get_config_var('OPT').strip().split() if _ not in _UNWANTED_OPTS)

    print( os.environ['OPT'])
    os.environ['CC'] = 'gcc'
    os.environ['CXX'] = 'g++'


lib_path = "cpp"

    
module_lib_ml_magic = Extension(
    name = '_ml_magic', 
    sources = [os.path.join(lib_path, 'ml-magic.cpp'), os.path.join(lib_path, 'ml-magic.i')], 
    extra_compile_args=['-DUSE_ASSERT_EXCEPTIONS','-DSWIG_PYTHON_SILENT_MEMLEAK','-std=gnu++11', "-fopenmp"],
    include_dirs=[numpy_include, '.', 'cpp/eigen'],
    extra_link_args=['-lgomp'],
    language='c++',
    swig_opts=["-c++"]
)


setup (
      name = 'ml_magic',
      version = '0.1',
      author = "Mads Fogtmann",
      description = "Magically machine learning in cpp",
      packages = find_packages(where = '.'),
      package_dir = {'':'.'},
      ext_modules=[module_lib_ml_magic]
)
