from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[
    Extension("krig",
              sources=["krig.pyx"],
              extra_compile_args=['/O2']),
    Extension("lhs",
              sources=["lhs.pyx"],
              extra_compile_args=['/O2'])
    ],
    include_dirs = [numpy.get_include()]
    )
