#!/usr/bin/env python

import sys
from setuptools import setup, find_packages

try:
    from pip._internal.req import parse_requirements
except:
    from pip.req import parse_requirements
requirements_path = 'requirements.txt'
if sys.version_info[0] < 3:
    requirements_path = 'requirements_py2.txt'
install_reqs = parse_requirements(requirements_path, session=False)
reqs = [str(ir.req) for ir in install_reqs]

setup(name='nlg-eval',
      version='2.0',
      description="Wrapper for multiple NLG evaluation methods and metrics.",
      author='Shikhar Sharma, Hannes Schulz',
      author_email='shikhar.sharma@microsoft.com, hannes.schulz@microsoft.com, justin.harris@microsoft.com',
      url='https://github.com/Maluuba/nlg-eval',
      packages=find_packages(),
      include_package_data=True,
      scripts=['bin/nlg-eval'],
      install_requires=reqs)
