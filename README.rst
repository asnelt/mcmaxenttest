# mcmaxenttest

Package description
-------------------

This package implements a statistical test that can assess higher-order
correlations of neural population spike counts in terms of an information
theoretic analysis. The test yields reliable results even when the number of
experimental samples is small. If you use this software for publication, please
cite [1].


Files
-----

* mcmaxenttest/demo.py - Demonstration of how to apply the test to count data
* mcmaxenttest/mcmaxenttest.py - Monte Carlo maximum entropy test
* mcmaxenttest/tests/test_mcmaxenttest.py - Tests for the mcmaxenttest module.
* README.rst - This file
* LICENSE - Software license


Test description
----------------

The test consists of two parts: (1) construction of a reference distribution
which is based on the single neuron spike count distributions and the
correlation coefficient and (2) a goodness-of-fit test to calculate a p-value
and eventually reject the reference distribution.

The reference distribution formalizes the linear dependency assumption. For
this purpose, we apply a maximum entropy model subject to a set of constraints.
The constraints contain the complete single neuron spike count distributions
and the linear correlation coefficient. Everything is therefore fixed by the
distribution constraints except for the higher-order correlations. If this
reference distribution can be statistically rejected then we can conclude that
higher-order correlations do matter.

The single neuron spike count distributions and the correlation coefficient are
not known a priori. Instead, they must be estimated from the data. For
simplicity, we assume that the single neuron distributions are Poisson
distributed. This leaves us with the estimation of firing rates and the
correlation coefficient. The test should be applicable even when the number of
samples is very small. Therefore, any estimates of distribution parameters are
not reliable. Instead of relying on specific estimates of these parameters, we
maximize the p-value over these parameters and then use the most conservative
p-value.

Please see [1] for a more detailed description of the test.


References
----------

1. Onken A, Dragoi V, Obermayer K (2012). A Maximum Entropy Test for
Evaluating Higher-Order Correlations in Spike Counts.
PLoS Comput Biol 8(6): e1002539. doi:10.1371/journal.pcbi.1002539


License
-------

```text
Copyright (C) 2012, 2017 Arno Onken

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3, or (at your option)
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.
```
