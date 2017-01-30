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
    keywords="monte carlo poisson test maximum entropy correlation",
    url="https://github.com/asnelt/mcmaxenttest/",
    author="Arno Onken",
    author_email="asnelt@asnelt.org",
    license="GPL-3.0",
    packages=["mcmaxenttest"],
    install_requires=["scipy"],
    test_suite="nose2.collector.collector"
)
