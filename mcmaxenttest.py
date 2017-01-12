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

from numpy import zeros, arange, maximum, log2, float64, random, multiply
from numpy import exp, dot, outer, sqrt, finfo, ceil
from numpy import concatenate
from scipy.stats import poisson
from scipy.optimize import anneal, fsolve

def mc_2nd_order_poisson_test(x, y, alpha=0.05, n_mc=100, n_iter=1):
    ''' Test for linear correlation maximum entropy between x and y, assuming
        that x and y are non-negative integer vectors.

        Arguments:
         x      - Vector of integer random values
         y      - Vector of integer random values of the same size as x
         alpha  - Significance level (default alpha = 0.05)
         n_mc   - Number of Monte Carlo samples (default n_mc = 1000)
         n_iter - Maximum number of iterations of the survival function search
                  (default n_iter = 0)

        Returns:
         h      - 1 indicates rejection of the linear correlation maximum entropy
                  hypothesis at the specified significance level; 0 otherwise
         p      - p-value at the specified significance level
    '''
    # Generate contingency table
    cont = zeros((x.max()+1, y.max()+1), dtype='float64')
    for i in range(x.size):
        cont[x[i], y[i]] += 1
    # Maximum entropy model for fixed marginals and correlation
    emp_p = cont / x.size
    (emp_lambda_0, emp_lambda_1, emp_correlation) = constraints(emp_p)
    x0 = [emp_lambda_0, emp_lambda_1, emp_correlation]
    if (n_iter < 1):
        # Do not search supremum of survival function
        g = survival(x0, cont, n_mc)
    else:
        # Search supremum of survival function
        valfun = lambda theta: -survival(theta.T, cont, n_mc)
        # Constraints for probability mass functions
        lb = [finfo(float).eps, finfo(float).eps, -1.0]
        # Use maximum count as upper bound for rate
        ub = [x.max()+1.0, y.max()+1.0, 1.0]
        (theta, retval) = anneal(valfun, x0, lower=lb, upper=ub, maxiter=n_iter)
        g = -valfun(theta)
    # Monte Carlo finite sample correction
    p = (n_mc * g + 1.0) / (n_mc + 1.0)
    h = p < alpha
    return (h, p)

def survival(theta, cont, n_mc):
    ''' Survival function of the test statistic.
    '''
    # Extract arguments
    n = cont.sum()
    lambda_0 = theta[0]
    lambda_1 = theta[1]
    correlation = theta[2]
    marginal_0 = poisson.pmf(arange(ceil(maximum(cont.shape[0], poisson.isf(1e-4, lambda_0)+1))), lambda_0)
    marginal_1 = poisson.pmf(arange(ceil(maximum(cont.shape[1], poisson.isf(1e-4, lambda_1)+1))), lambda_1)
    # Find corresponding maximum entropy distribution
    epmf = reference_pmf(marginal_0, marginal_1, correlation).flatten()
    epmf = epmf[epmf>0.0]
    pmf = cont.flatten() / n
    pmf = pmf[pmf>0.0]
    # Divergence measure: Entropy difference
    test_stat = abs((pmf * log2(pmf)).sum() - (epmf * log2(epmf)).sum())
    # Find critical region via Monte Carlo sampling
    sample_stat = zeros((n_mc,), dtype=float64)
    for i in range(n_mc):
        # Draw a sample from the multinomial distribution
        sample = random.multinomial(n, epmf).flatten() / n
        sample = sample[sample>0.0]
        sample_stat[i] = abs((sample * log2(sample)).sum() - (epmf * log2(epmf)).sum())
    # Correct for ties
    n_ties = (test_stat == sample_stat).sum()
    u = random.rand(n_ties+1)
    # Estimate survival function
    g = ((test_stat < sample_stat).sum() + (u[-1] >= u[1:n_ties]).sum()) / float(n_mc)
    return g

def reference_pmf(marginal_0, marginal_1, correlation):
    ''' Maximum entropy distribution with marginals and correlation coefficient
        as constraints.
    '''
    # Means of the marginals
    n_0 = marginal_0.size
    n_1 = marginal_1.size
    support_0 = arange(n_0)
    support_1 = arange(n_1)
    lambda_0 = dot(marginal_0, support_0)
    lambda_1 = dot(marginal_1, support_1)
    # Standard deviations of the marginals
    sigma_0 = sqrt(dot(marginal_0, (support_0-lambda_0)**2))
    sigma_1 = sqrt(dot(marginal_1, (support_1-lambda_1)**2))
    product_moment = correlation*sigma_0*sigma_1+lambda_0*lambda_1
    # Initial values
    x0 = concatenate((marginal_0, marginal_1, [0.0]))
    # Start with independent distribution
    x = fsolve(maxent_val, x0, args=(marginal_0, marginal_1, product_moment))
    f_0 = x[0:n_0]
    f_1 = x[n_0:(n_0+n_1)]
    mu = x[-1]
    # Compute maximum entropy distribution
    p_me = outer(f_0, f_1) * exp(mu * outer(support_0, support_1))
    # Cut off negative values
    p_me[p_me < 0] = 0
    # Renormalize distribution
    return (p_me / p_me.sum())

def maxent_val(x, marginal_0, marginal_1, product_moment):
    ''' Error function for finding the discrete bivariate maximum entropy
        distribution with marginals and correlation coefficient constraints.
    '''
    n_0 = marginal_0.size
    n_1 = marginal_1.size
    support_0 = arange(n_0)
    support_1 = arange(n_1)
    # Separate input
    f_0 = x[0:n_0]
    f_1 = x[n_0:(n_0+n_1)]
    mu = x[-1]
    # Output vector
    y = zeros(x.size, dtype='float64')
    y[0:n_0] = f_0 * dot(f_1, exp(mu * outer(support_1, support_0))) - marginal_0
    y[n_0:(n_0+n_1)] = f_1 * dot(f_0, exp(mu * outer(support_0, support_1))) - marginal_1
    s = (outer(support_0, support_1) * outer(f_0, f_1) * exp(mu * outer(support_0, support_1))).sum()
    y[-1] = s - product_moment
    return y

def constraints(p):
    ''' Constraints for the maximum entropy distribution.
    '''
    n = p.shape
    # Marginal distributions
    marginal_0 = p.sum(axis=1)
    marginal_1 = p.sum(axis=0)
    support_0 = arange(n[0])
    support_1 = arange(n[1])
    # Expectations
    lambda_0 = dot(marginal_0, support_0)
    lambda_1 = dot(marginal_1, support_1)
    # Standard deviations
    sigma_0 = sqrt(dot(marginal_0, (support_0 - lambda_0)**2))
    sigma_1 = sqrt(dot(marginal_1, (support_1 - lambda_1)**2))
    # Correlation coefficient
    if sigma_0 == 0 or sigma_1 == 0:
        correlation = 0.0
    else:
        correlation = (multiply(p, outer(support_0, support_1)).sum() - lambda_0*lambda_1) / (sigma_0*sigma_1)
    return (lambda_0, lambda_1, correlation)
