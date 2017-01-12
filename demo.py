# -*- coding: utf-8 -*-
# Copyright (C) 2012  Arno Onken
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Demonstration of a Monte Carlo maximum entropy test showing how to apply
the test to spike count data.
"""

from mcmaxenttest import *

# First variable: put the random integer values of your first element here
x = arange(5)
# Second variable: put the random integer values of your second element here
y = arange(5)
# Apply test
(h,p) = mc_2nd_order_poisson_test(x, y)
# Print results
print("Rejected: " + str(h))
print("p-value:  " + str(p))

