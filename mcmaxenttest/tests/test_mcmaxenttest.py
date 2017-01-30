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
This module implements tests for the mcmaxenttest module.
'''
import unittest
from scipy.stats import poisson, norm, multivariate_normal, uniform
import numpy as np
from mcmaxenttest import mcmaxenttest

class Order2PoissonTestCase(unittest.TestCase):
    '''
    This class represents test cases for the order2_poisson_test function.
    '''
    def setUp(self):
        '''
        Saves the current random state for later recovery and sets the random
        seed to get reproducible results.
        '''
        # Save random state for later recovery
        self.random_state = np.random.get_state()
        # Set fixed random seed
        np.random.seed(0)

    def tearDown(self):
        '''
        Recovers the original random state.
        '''
        # Recover original random state
        np.random.set_state(self.random_state)

    def test_independent_samples(self):
        '''
        Checks the test results on independent samples.
        '''
        # Draw independent Poisson samples
        counts_0 = poisson.rvs([5] * 100)
        counts_1 = poisson.rvs([5] * 100)
        # Apply test to samples
        (rejection, p_value) = mcmaxenttest.order2_poisson_test(counts_0, \
                                                                counts_1)
        self.assertFalse(rejection)
        self.assertAlmostEqual(p_value, 0.8331668)

    def test_higher_order_samples(self):
        '''
        Checks the test results on higher-order samples.
        '''
        # Draw samples from a higher-order mixture distribution
        mix_0 = norm.cdf(multivariate_normal.rvs([0, 0], \
                [[1, 0.9], [0.9, 1]], 100))
        mix_1 = norm.cdf(multivariate_normal.rvs([0, 0], \
                [[1, -0.9], [-0.9, 1]], 100))
        cond = uniform.rvs(size=100)
        mix_0[cond > 0.5, :] = mix_1[cond > 0.5, :]
        # Transform uniform marginals to Poisson marginals
        counts_0 = poisson.ppf(mix_0[:, 0], 5)
        counts_1 = poisson.ppf(mix_0[:, 1], 5)
        # Apply test to samples
        (rejection, p_value) = mcmaxenttest.order2_poisson_test(counts_0, \
                                                                counts_1)
        self.assertTrue(rejection)
        self.assertAlmostEqual(p_value, 0.0029970)
