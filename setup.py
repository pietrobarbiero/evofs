# -*- coding: utf-8 -*-
#
# Copyright 2019 Pietro Barbiero and Giovanni Squillero
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

from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='evofs',
    version='0.0.0',
    description='Multi-objective evolutionary feature selection.',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url='https://github.com/pietrobarbiero/moea-feature-selection/',
    author='Pietro Barbiero, Giovanni Squillero and Alberto Tonda',
    author_email='barbiero@tutanota.com',
    license="Apache 2.0",
    packages=find_packages(exclude=('tests', 'docs')),
    include_package_data=True,
)
