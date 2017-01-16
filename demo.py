# Copyright (C) 2012, 2017 Arno Onken
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
the test to count data.
"""

from scipy.stats import poisson, norm, multivariate_normal, uniform
import numpy as np
from mcmaxenttest import mc_2nd_order_poisson_test


# Number of test repetitions
N_TRIALS = 20
# Number of samples to draw in each trial
N_SAMPLES = 100
# Significance level of the test
ALPHA = 0.05
# Poisson rate
RATE = 5

# Apply test to independent Poisson samples
print("Applying test to independent Poisson samples...")
# Rejection results
h_ind = np.zeros((N_TRIALS, 1), dtype=bool)
# p-values
p_ind = np.zeros((N_TRIALS, 1))
for i in range(N_TRIALS):
    print(" Trial " + str(i+1) + " of " + str(N_TRIALS))
    # Draw independent Poisson samples
    x = poisson.rvs([RATE] * N_SAMPLES)
    y = poisson.rvs([RATE] * N_SAMPLES)
    # Apply test
    (h_ind[i], p_ind[i]) = mc_2nd_order_poisson_test(x, y, alpha=ALPHA)

# Apply test to samples from a higher-order distribution
print("Applying test to higher-order samples...")
# Correlation parameters of mixture components
RHO_1 = 0.9
RHO_2 = -0.9
# Rejection results
h_ho = np.zeros((N_TRIALS, 1), dtype=bool)
# p-values
p_ho = np.zeros((N_TRIALS, 1))
for i in range(N_TRIALS):
    print(" Trial " + str(i+1) + " of " + str(N_TRIALS))
    # Draw samples from a higher-order mixture distribution
    u = norm.cdf(multivariate_normal.rvs([0, 0], \
            [[1, RHO_1], [RHO_1, 1]], N_SAMPLES))
    v = norm.cdf(multivariate_normal.rvs([0, 0], \
            [[1, RHO_2], [RHO_2, 1]], N_SAMPLES))
    z = uniform.rvs(size=N_SAMPLES)
    u[z > 0.5, :] = v[z > 0.5, :]
    # Transform uniform marginals to Poisson marginals
    x = poisson.ppf(u[:, 0], RATE)
    y = poisson.ppf(u[:, 1], RATE)
    # Apply test
    (h_ho[i], p_ho[i]) = mc_2nd_order_poisson_test(x, y, alpha=ALPHA)

# Print results
print("Independent Poisson samples:")
print(" Rejections:      " + str(h_ind.mean() * 100) + "%")
print(" Average p-value: " + str(p_ind.mean()))
print("Higher-order samples:")
print(" Rejections:      " + str(h_ho.mean() * 100) + "%")
print(" Average p-value: " + str(p_ho.mean()))

