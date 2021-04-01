#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path
import sys

pkg_name = 'element_miniscope'
here = path.abspath(path.dirname(__file__))

long_description = """"
DataJoint Element for miniscope calcium imaging data.
"""

with open(path.join(here, 'requirements.txt')) as f:
    requirements = f.read().splitlines()

with open(path.join(here, pkg_name, 'version.py')) as f:
    exec(f.read())

setup(
    name='element-miniscope',
    version=__version__,
    description="Miniscope DataJoint Element",
    long_description=long_description,
    author='DataJoint NEURO',
    author_email='info@vathes.com',
    license='MIT',
    url='https://github.com/datajoint/element-miniscope',
    keywords='neuroscience calcium-imaging science datajoint miniscope',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=requirements,
)
