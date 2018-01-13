#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @time:2017/11/27 19:09
# @author:Gray

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension("sdtw", ["cy_sdtw.pyx"], include_dirs=[numpy.get_include()]),
    Extension("slndtw", ["cy_slndtw.pyx"], include_dirs=[numpy.get_include()]),
]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)
