#! /usr/bin/env python
DESCRIPTION = 'pyKrig: Kriging & ANOVA implemted in python'
DISTNAME = 'pyKrig'
MAINTAINER = 'STakanashi'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/SaTa999/pyKrig'
VERSION = '0.1.0'

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup


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
          )
