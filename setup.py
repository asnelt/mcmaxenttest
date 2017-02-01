# Copyright (C) 2017 Arno Onken
#
# This file is part of the mcmaxenttest package.
#
# The mcmaxenttest package is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# The mcmaxenttest package is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
'''
Setup for mcmaxenttest package.
'''
from setuptools import setup

setup(
    name="mcmaxenttest",
    version="1.0",
    description=("Statistical test to detect higher-order correlations between"
                 " count variables."),
    long_description=("This package implements a statistical test that can"
                      " assess higher-order correlations of neural population"
                      " spike counts in terms of an information theoretic"
                      " analysis. The test yields reliable results even when"
                      " the number of experimental samples is small."),
    keywords="monte carlo poisson test maximum entropy correlation",
    url="https://github.com/asnelt/mcmaxenttest/",
    author="Arno Onken",
    author_email="asnelt@asnelt.org",
    license="GPLv3+",
    packages=["mcmaxenttest"],
    scripts=["demo_mcmaxenttest"],
    install_requires=["scipy"],
    test_suite="nose2.collector.collector",
    tests_require=["nose2"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        ("License :: OSI Approved :: GNU General Public License v3 or later"
        " (GPLv3+)"),
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering"]
)
