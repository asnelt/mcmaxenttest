# -*- coding: utf-8 -*-
# Copyright (C) 2012, 2017  Arno Onken
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

from scipy.stats import poisson
from mcmaxenttest import *


# Number of test repetitions
N_TRIALS = 20
# Number of samples to draw in each trial
N_SAMPLES = 100
# Alpha level of the test
ALPHA = 0.05
# Poisson firing rate
RATE = 3 # Corresponds to 30 Hz for 100 ms bins

# Apply test to independent Poisson samples
print("Applying test to independent Poisson samples...")
# Rejection flags
h_ind = zeros((N_TRIALS, 1), dtype=bool)
# p-values
p_ind = zeros((N_TRIALS, 1))
for i in range(N_TRIALS):
    print(" Trial " + str(i) + " of " + str(N_TRIALS))
    # Draw independent Poisson samples
    x = poisson.rvs([RATE] * N_SAMPLES)
    y = poisson.rvs([RATE] * N_SAMPLES)
    # Apply test
    (h_ind[i], p_ind[i]) = mc_2nd_order_poisson_test(x, y, alpha=ALPHA)

# Print results
print("Rejections for independent Poisson samples: " + str(h_ind.mean() * 100) + "%")
print("Average p-value:                            " + str(p_ind.mean()))

