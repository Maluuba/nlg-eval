#!/usr/bin/env python2.7

from setuptools import setup
from pip.req import parse_requirements
install_reqs = parse_requirements('requirements.txt', session=False)
# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]

setup(name='nlg-eval',
      version='1.0',
      description='Wrapper for multiple NLG evaluation methods and metrics',
      author='Shikhar Sharma, Hannes Schulz',
      author_email='shikhar.sharma@microsoft.com, hannes.schulz@microsoft.com',
      url='http://github.com/Maluuba/nlg-eval',
      packages=['nlgeval'],
      scripts=['bin/nlg-eval'],
      install_requires=reqs)

