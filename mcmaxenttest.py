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
'''
This module implements a second order maximum entropy test for count data.
'''

from scipy.optimize import root, minimize
from scipy.stats import poisson
import numpy as np

def order2_poisson_test(counts_0, counts_1, alpha=0.05, nmc=1000, maxiter=1000):
    '''
    Test for linear correlation maximum entropy between counts_0 and counts_1,
    assuming that counts_0 and counts_1 are non-negative integer arrays.

    Args:
        counts_0: First array of non-negative integer random values.
        counts_1: Second array of non-negative integer random values of the same
                 size as counts_0.
        alpha: Significance level. Defaults to 0.05.
        nmc: Number of Monte Carlo samples. Defaults to 1000.
        maxiter: Maximum number of iterations for the survival function
                 optimization. Defaults to 1000.

    Returns:
        rejected: 1 indicates rejection of the linear correlation maximum
                  entropy hypothesis at the specified significance level;
                  0 otherwise.
        p_value: p-value at the specified significance level.
    '''
    # Generate contingency table
    cont = np.zeros((int(counts_0.max())+1, int(counts_1.max())+1), \
            dtype=np.float64)
    for i in range(counts_0.size):
        cont[int(counts_0[i]), int(counts_1[i])] += 1
    emp_p = cont / counts_0.size
    # Maximum entropy model for fixed marginals and correlation
    (emp_lambda_0, emp_lambda_1, emp_correlation) = constraints(emp_p)
    initial_point = [emp_lambda_0, emp_lambda_1, emp_correlation]
    if maxiter < 1:
        # Do not search for supremum of survival function
        max_survival = survival(initial_point, cont, nmc)
    else:
        # Search for supremum of survival function
        fun = lambda theta: -survival(theta.T, cont, nmc)
        # Constraints for probability mass functions
        bnds = [(np.finfo(float).eps, counts_0.max()+1.0), \
                (np.finfo(float).eps, counts_1.max()+1.0), (-1.0, 1.0)]
        result = minimize(fun, initial_point, method='TNC', bounds=bnds, \
                options={'maxiter': maxiter})
        # Run stochastic value function again at optimal point
        max_survival = -fun(result.x)
    # Monte Carlo finite sample correction
    p_value = (nmc * max_survival + 1.0) / (nmc + 1.0)
    rejected = p_value < alpha
    return (rejected, p_value)

def survival(theta, cont, nmc):
    '''
    Survival function of the test statistic.
    '''
    # Extract arguments
    total_sum = cont.sum()
    lambda_0 = theta[0]
    lambda_1 = theta[1]
    correlation = theta[2]
    marginal_0 = poisson.pmf(np.arange(np.ceil(np.maximum(cont.shape[0], \
            poisson.isf(1e-4, lambda_0)+1))), lambda_0)
    marginal_1 = poisson.pmf(np.arange(np.ceil(np.maximum(cont.shape[1], \
            poisson.isf(1e-4, lambda_1)+1))), lambda_1)
    # Find corresponding maximum entropy distribution
    epmf = reference_pmf(marginal_0, marginal_1, correlation).flatten()
    epmf = epmf[epmf > 0.0]
    pmf = cont.flatten() / total_sum
    pmf = pmf[pmf > 0.0]
    # Divergence measure: Entropy difference
    test_stat = abs((pmf * np.log2(pmf)).sum() - (epmf * np.log2(epmf)).sum())
    # Find critical region via Monte Carlo sampling
    sample_stat = np.zeros((nmc,), dtype=np.float64)
    for i in range(nmc):
        # Draw a sample from the multinomial distribution
        sample = np.random.multinomial(total_sum, epmf).flatten() / total_sum
        sample = sample[sample > 0.0]
        sample_stat[i] = abs((sample * np.log2(sample)).sum() \
                - (epmf * np.log2(epmf)).sum())
    # Correct for ties
    n_ties = (test_stat == sample_stat).sum()
    uni = np.random.rand(n_ties+1)
    # Estimate survival function
    value = ((test_stat < sample_stat).sum() \
            + (uni[-1] >= uni[1:n_ties]).sum()) / float(nmc)
    return value

def reference_pmf(marginal_0, marginal_1, correlation):
    '''
    Maximum entropy distribution with marginals and correlation
    coefficient as constraints.
    '''
    n_0 = marginal_0.size
    n_1 = marginal_1.size
    support_0 = np.arange(n_0)
    support_1 = np.arange(n_1)
    # Means of the marginals
    lambda_0 = np.dot(marginal_0, support_0)
    lambda_1 = np.dot(marginal_1, support_1)
    # Standard deviations of the marginals
    sigma_0 = np.sqrt(np.dot(marginal_0, (support_0-lambda_0)**2))
    sigma_1 = np.sqrt(np.dot(marginal_1, (support_1-lambda_1)**2))
    product_moment = correlation*sigma_0*sigma_1+lambda_0*lambda_1
    # Start with independent distribution
    initial_point = np.concatenate((marginal_0, marginal_1, [0.0]))
    # Find root
    result = root(maxent_errors, initial_point, args=(marginal_0, marginal_1, \
            product_moment))
    # Compute maximum entropy distribution
    p_me = np.outer(result.x[0:n_0], result.x[n_0:(n_0+n_1)]) \
            * np.exp(result.x[-1] * np.outer(support_0, support_1))
    # Cut off negative values
    p_me[p_me < 0] = 0
    # Renormalize distribution
    return p_me / p_me.sum()

def maxent_errors(point, marginal_0, marginal_1, product_moment):
    '''
    Error function for finding the discrete bivariate maximum entropy
    distribution with marginals and correlation coefficient constraints.
    '''
    n_0 = marginal_0.size
    n_1 = marginal_1.size
    support_0 = np.arange(n_0)
    support_1 = np.arange(n_1)
    # Separate input
    f_0 = point[0:n_0]
    f_1 = point[n_0:(n_0+n_1)]
    rho = point[-1]
    # Output array
    values = np.zeros(point.size, dtype=np.float64)
    values[0:n_0] = f_0 * np.dot(f_1, np.exp(rho \
            * np.outer(support_1, support_0))) - marginal_0
    values[n_0:(n_0+n_1)] = f_1 * np.dot(f_0, np.exp(rho \
            * np.outer(support_0, support_1))) - marginal_1
    point_moment = (np.outer(support_0, support_1) * np.outer(f_0, f_1) \
            * np.exp(rho * np.outer(support_0, support_1))).sum()
    values[-1] = point_moment - product_moment
    return values

def constraints(p):
    '''
    Constraints for the maximum entropy distribution.
    '''
    dim = p.shape
    # Marginal distributions
    marginal_0 = p.sum(axis=1)
    marginal_1 = p.sum(axis=0)
    support_0 = np.arange(dim[0])
    support_1 = np.arange(dim[1])
    # Expectations
    lambda_0 = np.dot(marginal_0, support_0)
    lambda_1 = np.dot(marginal_1, support_1)
    # Standard deviations
    sigma_0 = np.sqrt(np.dot(marginal_0, (support_0 - lambda_0)**2))
    sigma_1 = np.sqrt(np.dot(marginal_1, (support_1 - lambda_1)**2))
    # Correlation coefficient
    if sigma_0 == 0 or sigma_1 == 0:
        correlation = 0.0
    else:
        correlation = (np.multiply(p, np.outer(support_0, support_1)).sum() \
                - lambda_0*lambda_1) / (sigma_0*sigma_1)
    return (lambda_0, lambda_1, correlation)
