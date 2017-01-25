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
from scipy.stats import entropy, poisson
import numpy as np

def order2_poisson_test(counts_0, counts_1, alpha=0.05, nmc=1000, maxiter=1000):
    '''
    Statistical test for a bivariate count distribution that is characterized by
    Poisson marginals and second-order correlation.

    Args:
        counts_0: Array containing samples of the first element of the bivariate
                  distribution as non-negative integer random values.
        counts_1: Array containing corresponding samples of the second element
                  of the bivariate distribution as non-negative integer random
                  values of the same size as `counts_0`.
        alpha: Significance level. Defaults to 0.05.
        nmc: Number of Monte Carlo samples. Defaults to 1000.
        maxiter: Maximum number of iterations for the survival function
                 optimization. Defaults to 1000.

    Returns:
        rejected: 1 indicates rejection of the linear correlation maximum
                  entropy hypothesis at the specified significance level;
                  0 otherwise.
        p_value: p-value of the hypothesis.

    Raises:
        ValueError: If `counts_0` and `counts_1` are different lengths.
    '''
    if len(counts_0) != len(counts_1):
        raise ValueError("Arrays 'counts_0' and 'counts_1' must be same"
                         " lengths.")
    # Generate contingency table and empirical distribution
    cont = contingency_table(counts_0, counts_1)
    emp_pmf = cont / counts_0.size
    # Empirical parameters of the distribution
    emp_par = BivariatePoissonParameters.estimate_from(emp_pmf)
    # Initialize survival function optimization
    initial_point = emp_par.as_list()
    if maxiter < 1:
        # Just use the initial point for the survival function
        max_survival = survival(initial_point, cont, nmc)
    else:
        # Search for maximum of survival function
        fun = lambda theta: -survival(theta.T, cont, nmc)
        # Constraints for means and correlation
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

def contingency_table(counts_0, counts_1):
    '''
    Generate contingency table from counts.

    Args:
        counts_0: Array containing samples of the first element of the bivariate
                  distribution as non-negative integer random values.
        counts_1: Array containing corresponding samples of the second element
                  of the bivariate distribution as non-negative integer random
                  values of the same size as `counts_0`.

    Returns:
        cont: Contingency table of the input count pairs.

    Raises:
        ValueError: If `counts_0` and `counts_1` are different lengths.
    '''
    if len(counts_0) != len(counts_1):
        raise ValueError("Arrays 'counts_0' and 'counts_1' must be same"
                         " lengths.")
    cont = np.zeros((int(counts_0.max())+1, int(counts_1.max())+1), \
            dtype=np.float64)
    for i in range(counts_0.size):
        cont[int(counts_0[i]), int(counts_1[i])] += 1
    return cont

def survival(theta, cont, nmc):
    '''
    Survival function of the test statistic.

    Args:
        theta: The distribution parameters as a list.
        cont: Contingency table of the input counts.
        nmc: Number of Monte Carlo samples.

    Returns:
        value: The survival value at `theta`.
    '''
    pars = BivariatePoissonParameters(theta[0], theta[1], theta[2])
    marginal_0 = poisson.pmf(np.arange(np.ceil(np.maximum(cont.shape[0], \
            poisson.isf(1e-4, pars.mean_0)+1))), pars.mean_0)
    marginal_1 = poisson.pmf(np.arange(np.ceil(np.maximum(cont.shape[1], \
            poisson.isf(1e-4, pars.mean_1)+1))), pars.mean_1)
    # Find corresponding maximum entropy distribution
    model = Order2MaximumEntropyModel(marginal_0, marginal_1, pars.correlation)
    # Calculate empirical test statistic
    test_stat = divergence(cont / cont.sum(), model.pmf())
    # Find critical region via Monte Carlo sampling
    sample_stat = np.zeros((nmc,), dtype=np.float64)
    for i in range(nmc):
        # Draw a sample from the maximum entropy model
        sample = model.rvs(cont.sum())
        sample_stat[i] = divergence(sample / cont.sum(), model.pmf())
    # Correct for ties
    n_ties = (test_stat == sample_stat).sum()
    uni = np.random.rand(n_ties+1)
    # Estimate survival function
    value = ((test_stat < sample_stat).sum() \
            + (uni[-1] >= uni[1:n_ties]).sum()) / float(nmc)
    return value

def divergence(pmf_0, pmf_1):
    '''
    Define distribution divergence as entropy difference. We are deliberately
    not choosing the Kullback-Leibler divergence here, because the difference in
    entropy is what we care about.

    Args:
        pmf_0: Values of the probability mass function of the first
               distribution as an array.
        pmf_1: Values of the probability mass function of the second
               distribution as an array.

    Returns:
        div: The distribution divergence.
    '''
    div = abs(entropy(pmf_0.flatten()) - entropy(pmf_1.flatten()))
    return div

class Order2MaximumEntropyModel(object):
    '''
    This class represents a bivariate maximum entropy distribution with
    marginals and correlation coefficient as constraints.
    '''
    def __init__(self, marginal_0, marginal_1, correlation):
        '''
        Constructs a maximum entropy distribution from given marginals and
        correlation coefficient as contraints.

        Args:
            marginal_0: Values of the first marginal distribution as an array.
            marginal_1: Values of the second marginal distribution as an array.
            correlation: Correlation coefficient as a scalar between -1 and 1.
        '''
        n_0 = marginal_0.size
        n_1 = marginal_1.size
        support_0 = np.arange(n_0)
        support_1 = np.arange(n_1)
        # Means of the marginals
        mean_0 = np.dot(marginal_0, support_0)
        mean_1 = np.dot(marginal_1, support_1)
        # Standard deviations of the marginals
        sigma_0 = np.sqrt(np.dot(marginal_0, (support_0-mean_0)**2))
        sigma_1 = np.sqrt(np.dot(marginal_1, (support_1-mean_1)**2))
        product_moment = correlation*sigma_0*sigma_1+mean_0*mean_1
        # Start with independent distribution
        initial_point = np.concatenate((marginal_0, marginal_1, [0.0]))
        # Find root
        result = root(self.__maxent_errors, initial_point, \
                args=(marginal_0, marginal_1, product_moment))
        # Compute maximum entropy distribution
        self.pmf_me = np.outer(result.x[0:n_0], result.x[n_0:(n_0+n_1)]) \
                * np.exp(result.x[-1] * np.outer(support_0, support_1))
        # Cut off negative values
        self.pmf_me[self.pmf_me < 0] = 0
        # Renormalize distribution
        self.pmf_me /= self.pmf_me.sum()

    def pmf(self):
        '''
        Values of the probability mass function of the maximum entropy
        distribution.

        Returns:
            pmf_me: Values of the maximum entropy distribution as a
                    two-dimensional array.
        '''
        return self.pmf_me

    def rvs(self, size=1):
        '''
        Random variates from the model distribution represented as a contingency
        table with a sum of elements equal to the size argument.

        Args:
            size: Total number of samples corresponding to the sum of elements
                  in the resulting contingency table. Defaults to 1.

        Returns:
            samples: Contingency table of the distribution samples as a
                     two-dimensional array.
        '''
        # Draw a sample from the multinomial distribution
        samples = np.random.multinomial(size, self.pmf_me.flatten())
        samples = samples.reshape(self.pmf_me.shape)
        return samples

    @staticmethod
    def __maxent_errors(point, marginal_0, marginal_1, product_moment):
        '''
        Error function for finding the discrete bivariate maximum entropy
        distribution with marginals and correlation coefficient constraints.

        Args:
            point: The exponential parameters of the maximum entropy
                   distribution for marginals and product moment concatenated
                   into an array.
            marginal_0: Values of the first marginal distribution as an array.
            marginal_1: Values of the second marginal distribution as an array.
            product_moment: The second order product moment.

        Returns:
            errors: The errors of marginal values and product moment when
                    compared to those induced by the parameters in `point`,
                    concatenated into an array.
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
        errors = np.zeros(point.size, dtype=np.float64)
        errors[0:n_0] = f_0 * np.dot(f_1, np.exp(rho \
                * np.outer(support_1, support_0))) - marginal_0
        errors[n_0:(n_0+n_1)] = f_1 * np.dot(f_0, np.exp(rho \
                * np.outer(support_0, support_1))) - marginal_1
        point_moment = (np.outer(support_0, support_1) * np.outer(f_0, f_1) \
                * np.exp(rho * np.outer(support_0, support_1))).sum()
        errors[-1] = point_moment - product_moment
        return errors

class BivariatePoissonParameters(object):
    '''
    This class represents the parameters of a bivariate Poisson distribution
    characterized by marginal means and a correlation coefficient.
    '''
    def __init__(self, mean_0, mean_1, correlation):
        '''
        Constructor setting means and correlation.

        Args:
            mean_0: The mean of the first marginal.
            mean_1: The mean of the second marginal.
            correlation: The correlation coefficient of the count variables.
        '''
        self.mean_0 = mean_0
        self.mean_1 = mean_1
        self.correlation = correlation

    def as_list(self):
        '''
        Concatenates all parameters into a list.

        Returns:
            par_list: Means and correlation as a list.
        '''
        par_list = [self.mean_0, self.mean_1, self.correlation]
        return par_list

    @staticmethod
    def estimate_from(pmf):
        '''
        Estimates parameters from given probability mass function.

        Args:
            pmf: Values of a probability density function of a joint
                 distribution of two count variables represented as a
                 two-dimensional array.

        Returns:
            pars: `BivariatePoissonParameters` object with means and
                  correlations estimated from the input distribution.
        '''
        size = pmf.shape
        # Marginal distributions
        marginal_0 = pmf.sum(axis=1)
        marginal_1 = pmf.sum(axis=0)
        support_0 = np.arange(size[0])
        support_1 = np.arange(size[1])
        # Expectations
        mean_0 = np.dot(marginal_0, support_0)
        mean_1 = np.dot(marginal_1, support_1)
        # Standard deviations
        sigma_0 = np.sqrt(np.dot(marginal_0, (support_0 - mean_0)**2))
        sigma_1 = np.sqrt(np.dot(marginal_1, (support_1 - mean_1)**2))
        # Correlation coefficient
        if sigma_0 == 0 or sigma_1 == 0:
            correlation = 0.0
        else:
            correlation = (np.multiply(pmf, \
                    np.outer(support_0, support_1)).sum() - mean_0*mean_1) \
                    / (sigma_0*sigma_1)
        pars = BivariatePoissonParameters(mean_0, mean_1, correlation)
        return pars
