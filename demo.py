# Copyright (C) 2012, 2017 Arno Onken
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Demonstration of a second order maximum entropy test showing how to apply the
test to count data.
"""

from scipy.stats import poisson, norm, multivariate_normal, uniform
import numpy as np
from mcmaxenttest import order2_poisson_test

def main():
    '''
    Main function for demonstration of maximum entropy test.
    '''
    test_independent_poisson()
    test_higher_order_poisson()

def test_independent_poisson():
    '''
    Apply test to independent Poisson samples.
    '''
    print("Applying test to independent Poisson samples...")
    # Number of test repetitions
    n_trials = 20
    # Number of samples to draw in each trial
    n_samples = 100
    # Significance level of the test
    alpha = 0.05
    # Poisson rate
    rate = 5
    # Rejection results
    rejected_ind = np.zeros((n_trials, 1), dtype=bool)
    # p-values
    p_values_ind = np.zeros((n_trials, 1))
    for i in range(n_trials):
        print(" Trial " + str(i+1) + " of " + str(n_trials))
        # Draw independent Poisson samples
        counts_0 = poisson.rvs([rate] * n_samples)
        counts_1 = poisson.rvs([rate] * n_samples)
        # Apply test
        (rejected_ind[i], p_values_ind[i]) = order2_poisson_test(counts_0, \
                counts_1, alpha=alpha)
    print_results("Independent Poisson samples:", rejected_ind, p_values_ind)

def test_higher_order_poisson():
    '''
    Apply test to samples from a higher-order distribution.
    '''
    print("Applying test to higher-order samples...")
    # Number of test repetitions
    n_trials = 20
    # Number of samples to draw in each trial
    n_samples = 100
    # Significance level of the test
    alpha = 0.05
    # Poisson rate
    rate = 5
    # Correlation parameters of mixture components
    rho = 0.9
    # Rejection results
    rejected_ho = np.zeros((n_trials, 1), dtype=bool)
    # p-values
    p_values_ho = np.zeros((n_trials, 1))
    for i in range(n_trials):
        print(" Trial " + str(i+1) + " of " + str(n_trials))
        # Draw samples from a higher-order mixture distribution
        mix_0 = norm.cdf(multivariate_normal.rvs([0, 0], \
                [[1, rho], [rho, 1]], n_samples))
        mix_1 = norm.cdf(multivariate_normal.rvs([0, 0], \
                [[1, -rho], [-rho, 1]], n_samples))
        cond = uniform.rvs(size=n_samples)
        mix_0[cond > 0.5, :] = mix_1[cond > 0.5, :]
        # Transform uniform marginals to Poisson marginals
        counts_0 = poisson.ppf(mix_0[:, 0], rate)
        counts_1 = poisson.ppf(mix_0[:, 1], rate)
        # Apply test
        (rejected_ho[i], p_values_ho[i]) = order2_poisson_test(counts_0, \
                counts_1, alpha=alpha)
    # Print results
    print_results("Higher-order samples:", rejected_ho, p_values_ho)

def print_results(head_string, rejected, p_values):
    '''
    Prints the test results.

    Args:
        head_string: The heading to print.
        rejected: Array of boolean test rejections.
        p_values: Array of p-values.
    '''
    print(head_string)
    print(" Rejections:      " + str(rejected.mean() * 100) + "%")
    print(" Average p-value: " + str(p_values.mean()))

if __name__ == "__main__":
    main()
