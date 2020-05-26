# -*- coding: utf-8 -*-
#
# Copyright 2019 Pietro Barbiero, Giovanni Squillero and Alberto Tonda
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('evofs', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'evofs'
DESCRIPTION = 'Multi-objective evolutionary feature selection.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'P. Barbiero'
MAINTAINER_EMAIL = 'barbiero@tutanota.com'
URL = 'https://github.com/pietrobarbiero/moea-feature-selection'
LICENSE = 'Apache 2.0'
DOWNLOAD_URL = 'https://github.com/pietrobarbiero/moea-feature-selection.git'
VERSION = __version__
INSTALL_REQUIRES = ['numpy', 'pandas', 'inspyred', 'scikit-learn']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/x-rst',
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
