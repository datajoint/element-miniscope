#!/usr/bin/env python
import subprocess
from os import path
from setuptools import find_packages, setup

pkg_name = next(p for p in find_packages() if "." not in p)
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), "r") as f:
    long_description = f.read()

with open(path.join(here, "requirements.txt")) as f:
    requirements = f.read().splitlines()

with open(path.join(here, pkg_name, "version.py")) as f:
    exec(f.read())

# Prerequisite of caiman installation to run its setup.py
subprocess.call(["pip", "install", "numpy", "Cython"])

extras_require = {
    "caiman": "caiman @ git+https://github.com/datajoint-company/CaImAn.git"
}


setup(
    name=pkg_name.replace("_", "-"),
    version=__version__,  # noqa F821
    description="Miniscope DataJoint Element",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DataJoint",
    author_email="info@datajoint.com",
    license="MIT",
    url=f'https://github.com/datajoint/{pkg_name.replace("_", "-")}',
    keywords="neuroscience calcium-imaging science datajoint miniscope",
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    scripts=[],
    install_requires=requirements,
    extras_require=extras_require,
)
