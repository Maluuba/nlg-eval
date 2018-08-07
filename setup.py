#!/usr/bin/env python

import sys
from setuptools import find_packages
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

try:
    from pip._internal.req import parse_requirements
except:
    from pip.req import parse_requirements


def _download_files():
    pass

#https://stackoverflow.com/a/36902139/1226799
class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        _download_files()


class PostInstallCommand(install):
    def run(self):
        install.run(self)
        _download_files()


requirements_path = 'requirements.txt'
if sys.version_info[0] < 3:
    requirements_path = 'requirements_py2.txt'
install_reqs = parse_requirements(requirements_path, session=False)
reqs = [str(ir.req) for ir in install_reqs]

setup(name='nlg-eval',
      version='2.0',
      description="Wrapper for multiple NLG evaluation methods and metrics.",
      author='Shikhar Sharma, Hannes Schulz, Justin Harris',
      author_email='shikhar.sharma@microsoft.com, hannes.schulz@microsoft.com, justin.harris@microsoft.com',
      url='https://github.com/Maluuba/nlg-eval',
      packages=find_packages(),
      include_package_data=True,
      scripts=['bin/nlg-eval'],
      install_requires=reqs,
      cmdclass={
          'develop': PostDevelopCommand,
          'install': PostInstallCommand,
      })
