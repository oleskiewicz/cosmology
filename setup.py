#!/usr/bin/env python3
from os import path

from setuptools import setup

with open(path.join(path.abspath(path.dirname(__file__)), "README")) as f:
    long_description = f.read()

setup(
    name="cosmology",
    version="0.1",
    author="Piotr Oleskiewicz",
    author_email="piotr.oleskiewicz@durham.ac.uk",
    description=("A set of tools for cosmological calculations."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GPLv3",
    keywords="utility cosmology astrophysics dark_matter",
    url="https://github.com/oleskiewicz/cosmology",
    install_requires=["numpy", "scipy"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "Intended Audience :: Science/Research"
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
