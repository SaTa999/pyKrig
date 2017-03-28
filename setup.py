#! /usr/bin/env python
import os
import sysconfig
from Cython.Distutils import build_ext
import numpy


DESCRIPTION = 'pyKrig: Kriging & ANOVA implemted in python'
DISTNAME = 'pyKrig'
MAINTAINER = 'STakanashi'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/SaTa999/pyKrig'
VERSION = '0.1.0'

try:
    from setuptools import setup, Extension
    from setuptools.command.install_lib import install_lib
except ImportError:
    from distutils.core import setup, Extension
    from distutils.command.install_lib import install_lib


def check_dependencies():
    install_requires = []
    try:
        import numpy
    except ImportError:
        install_requires.append('numpy')
    try:
        import scipy
    except ImportError:
        install_requires.append('scipy')
    try:
        import matplotlib
    except ImportError:
        install_requires.append('matplotlib')
    try:
        import pandas
    except ImportError:
        install_requires.append('pandas')
    return install_requires


try:
    from Cython.Distutils import build_ext
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False


if USE_CYTHON:
    _ext_modules = [
    Extension("pyKrig.utilities.krig", sources=["pyKrig/utilities/krig.pyx"], extra_compile_args=['/O2']),
    Extension("pyKrig.utilities.lhs", sources=["pyKrig/utilities/lhs.pyx"], extra_compile_args=['/O2'])
    ]
    _cmdclass = {'build_ext': build_ext}
else:
    _ext_modules = [
    Extension("pyKrig.utilities.krig", sources=["pyKrig/utilities/krig.c"], extra_compile_args=['/O2']),
    Extension("pyKrig.utilities.lhs", sources=["pyKrig/utilities/lhs.c"], extra_compile_args=['/O2'])
    ]
    _cmdclass = {}


class BuildExtWithoutPlatformSuffix(build_ext):
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        return get_ext_filename_without_platform_suffix(filename)


def get_ext_filename_without_platform_suffix(filename):
    name, ext = os.path.splitext(filename)
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

    if ext_suffix == ext:
        return filename

    ext_suffix = ext_suffix.replace(ext, '')
    idx = name.find(ext_suffix)

    if idx == -1:
        return filename
    else:
        return name[:idx] + ext


_cmdclass["build_ext"] = BuildExtWithoutPlatformSuffix


if __name__ == "__main__":
    install_requires = check_dependencies()

    setup(name=DISTNAME,
          author=MAINTAINER,
          maintainer=MAINTAINER,
          description=DESCRIPTION,
          license=LICENSE,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          install_requires=install_requires,
          packages=['pyKrig', 'pyKrig.utilities'],
          classifiers=[
              'Intended Audience :: Science/Research',
              'Programming Language :: Python :: 3.6',
              'License :: OSI Approved :: MIT License',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows'],
          cmdclass = _cmdclass,
          ext_modules = _ext_modules,
          include_dirs=[numpy.get_include()]
          )
