#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dense-depth-fkugler", 
    version="1.0.0",
    author="Felix Kugler",
    author_email="e1526144@student.tuwien.ac.at",
    description="Dense depth estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/relgukxilef/dense-depth-estimation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.4, <3.8',
)
