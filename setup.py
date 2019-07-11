#!/usr/bin/env python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

import sys

from setuptools import find_packages
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

try:
    from pip._internal.req import parse_requirements
except:
    from pip.req import parse_requirements


if __name__ == '__main__':
    requirements_path = 'requirements.txt'
    if sys.version_info[0] < 3:
        requirements_path = 'requirements_py2.txt'
    install_reqs = parse_requirements(requirements_path, session=False)
    reqs = [str(ir.req) for ir in install_reqs]

    setup(name='nlg-eval',
          version='2.1',
          description="Wrapper for multiple NLG evaluation methods and metrics.",
          author='Shikhar Sharma, Hannes Schulz, Justin Harris',
          author_email='shikhar.sharma@microsoft.com, hannes.schulz@microsoft.com, justin.harris@microsoft.com',
          url='https://github.com/Maluuba/nlg-eval',
          packages=find_packages(),
          include_package_data=True,
          scripts=['bin/nlg-eval'],
          install_requires=reqs,
    )
