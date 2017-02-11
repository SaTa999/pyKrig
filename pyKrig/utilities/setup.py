from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[
    Extension("utilities",
              sources=["utilities.pyx"],
              extra_compile_args=['/O2'])
    ],
    include_dirs = [numpy.get_include()]
    )
