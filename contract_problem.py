#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:20:35 2020

@author: mckay
"""

import numpy as np
import matplotlib.pyplot as plt



def manager_strategy_linear(T, pmt_slope, interest_rate, max_approp=0.1):
    approp = np.zeros(T+1)
    for t in range(1, T+1):
        if pmt_slope * sum(np.exp(-interest_rate*(s-t)) for s in range(t+1, T+1)) < 1:
            approp[t] = max_approp
    return approp


def manager_utility0_linear(T, mu, y0, pmt_intercept, pmt_slope, interest_rate, approp=None):
    if approp is None:
        approp = manager_strategy_linear(T, pmt_slope, interest_rate)
    cumulative_approp = np.cumsum(approp)
    # set expected y to y0 with exponential growth at rate mu minus cumulated appropriation
    # TODO: deal better with growth rates for continous compounding
    y_exp = np.array([y0 * (1+mu)**t for t in range(T)]) - cumulative_approp[:-1]
    utility = sum(np.exp(-interest_rate*t) * (pmt_intercept + pmt_slope*y_exp[t-1] + approp[t]) \
                  for t in range(1, T+1))
    return utility


def investor_utility0_linear(T, mu, y0, pmt_intercept, pmt_slope, interest_rate, max_approp=0.1):
    approp = manager_strategy_linear(T, pmt_slope, interest_rate, max_approp)
    if manager_utility0_linear(T, mu, y0, pmt_intercept, pmt_slope, interest_rate, approp) <= 0: # or some other threshold
        return -np.inf  # -infinity since cases where manager's utility is non-positive are unacceptable
    # note short and long-term interest rates assumed to be the same here. can change that.
    cumulative_approp = np.cumsum(approp)
    util = sum(np.exp(-interest_rate*t) * (mu*(1+mu)**(t-1) * y0 - approp[t] \
                      - pmt_intercept - pmt_slope*((1+mu)**(t-1) * y0 - cumulative_approp[t-1])) \
                        for t in range(1,T+1))
    return util


# vectorized version of investor's utility function for convenience
vect_investor_util = np.vectorize(investor_utility0_linear,
                                  excluded=['T', 'mu', 'y0', 'interest_rate', 'max_approp'])


def plot_investor_utility(intcpt_a, intcpt_b, slope_a, slope_b,
                          T, mu, y0, interest_rate, max_approp=0.1,
                          resolution=20, show_max=True):
    i = np.linspace(intcpt_a, intcpt_b, resolution)
    s = np.linspace(slope_a, slope_b, resolution)
    ii, ss = np.meshgrid(i, s)
    utility = vect_investor_util(T, mu, y0, ii, ss, interest_rate, max_approp)
    # x axis will be intercept, y axis will be slope
    plt.imshow(np.flip(utility, 0), cmap=plt.cm.Reds, extent=[intcpt_a, intcpt_b, slope_a, slope_b])
    plt.colorbar()
    plt.xlabel('intercept')
    plt.ylabel('slope')
    max_index = np.unravel_index(utility.argmax(), utility.shape)
    max_intercept, max_slope = i[max_index[1]], s[max_index[0]]
    if show_max:
        plt.scatter(max_intercept, max_slope, marker='x')
    plt.show()
    return utility, (max_intercept, max_slope)


def nd_grid_maximize(f, *ranges, midpoints=10, recursions=2):
    n = len(ranges)
    vecs = [np.linspace(r[0], r[1], midpoints+2) for r in ranges]
    meshes = np.meshgrid(*vecs, indexing='ij')
    output = f(*meshes)  # f must be vectorized and take n args
    max_index = np.unravel_index(output.argmax(), output.shape)
    if recursions > 0:
        new_ranges = [(vecs[i][max(j-1, 0)], vecs[i][min(j+1, midpoints+1)]) for i, j in enumerate(max_index)]
        return nd_grid_maximize(f, *new_ranges, midpoints=midpoints, recursions=recursions-1)
    else:
        maxinput = tuple(vecs[i][max_index[i]] for i in range(n))
        maxoutput = f(*maxinput)
        return maxinput, maxoutput


def investor_strategy_linear(T, mu, y0, interest_rate, max_approp=0.1, irange=(-1.5, 0.5), srange=(-0.1, 0.2)):
    def objective(intcpt, slope):
        return vect_investor_util(T, mu, y0, intcpt, slope, interest_rate, max_approp)
    return nd_grid_maximize(objective, irange, srange)



def manager_strategy_modified(T, beta, gamma, interest_rate, max_approp=0.1):
    approp = np.zeros(T+1)
    for t in range(1, T+1):
        if gamma + beta * sum(np.exp(-interest_rate*(s-t)) for s in range(t+1, T+1)) < 1:
            approp[t] = max_approp
    return approp


def manager_utility0_modified(T, mu, y0, alpha, beta, gamma, interest_rate, approp=None):
    if approp is None:
        approp = manager_strategy_modified(T, beta, gamma, interest_rate)
    cumulative_approp = np.cumsum(approp)
    # set expected y to y0 with exponential growth at rate mu minus cumulated appropriation
    # TODO: deal better with growth rates for continous compounding
    y_exp = np.array([y0 * (1+mu)**t for t in range(T)]) - cumulative_approp[:-1]
    utility = sum(np.exp(-interest_rate*t) * (alpha + beta*y_exp[t-1] \
                  + gamma*(y0*mu*(1+mu)**(t-1)-approp[t]) + approp[t]) \
                  for t in range(1, T+1))
    return utility


def investor_utility0_modified(T, mu, y0, alpha, beta, gamma, interest_rate, max_approp=0.1):
    approp = manager_strategy_modified(T, beta, gamma, interest_rate, max_approp)
    manager_util = manager_utility0_modified(T, mu, y0, alpha, beta, gamma, interest_rate, approp)
    if manager_util <= 0: # or some other threshold
        return -np.inf  # -infinity since cases where manager's utility is non-positive are unacceptable
    # note short and long-term interest rates assumed to be the same here. can change that.
    # NOTE! if short and long-term interest rates not the same, this next part needs to be changed
        # since manager and investor utility don't sum to present value of all y
    util = sum(np.exp(-interest_rate*t) * (mu*(1+mu)**(t-1) * y0) for t in range(1,T+1)) - manager_util
    return util


vect_investor_util_mod = np.vectorize(investor_utility0_modified,
                                      excluded=['T', 'mu', 'y0', 'interest_rate', 'max_approp'])


def plot_investor_utility_mod(arange, brange, grange,
                              T, mu, y0, interest_rate, max_approp=0.1,
                              resolution=20, show_max=True):
    '''
    Creates a heatmap of the modified investor's utility function across two variables, with the other held fixed
    
    One of arange, brange, grange must be a scalar; the others must be size-2 tuples representing ranges to plot over
    T, mu, etc. are the same parameters as elsewhere
    resolution is the the number of points to evaluate -- the result will be a resolution x resolution matrix/heatmap
    '''
    if not isinstance(arange, (tuple, list)):
        case = 1
        a1, b1 = brange
        a2, b2 = grange
    elif not isinstance(brange, (tuple, list)):
        case = 2
        a1, b1 = arange
        a2, b2 = grange
    elif not isinstance(grange, (tuple, list)):
        case = 3
        a1, b1 = arange
        a2, b2 = brange
    else:
        print('one of arange, brange, grange must be scalar; the others must be size-2 tuples')
        return np.array(), (0, 0)
    x = np.linspace(a1, b1, resolution)
    y = np.linspace(a2, b2, resolution)
    xx, yy = np.meshgrid(x, y)
    if case == 1:
        utility = vect_investor_util_mod(T, mu, y0, arange, xx, yy, interest_rate, max_approp)
    elif case == 2:
        utility = vect_investor_util_mod(T, mu, y0, xx, brange, yy, interest_rate, max_approp)
    elif case == 3:
        utility = vect_investor_util_mod(T, mu, y0, xx, yy, grange, interest_rate, max_approp)
    plt.imshow(np.flip(utility, 0), cmap=plt.cm.Reds, extent=[a1, b1, a2, b2])
    plt.colorbar()
    plt.xlabel('alpha' if case != 1 else 'beta')
    plt.ylabel('gamma' if case != 3 else 'beta')
    max_index = np.unravel_index(utility.argmax(), utility.shape)
    max1, max2 = x[max_index[1]], y[max_index[0]]
    if show_max:
        plt.scatter(max1, max2, marker='x')
    excluded = 'alpha' if case == 1 else 'beta' if case == 2 else 'gamma'
    excluded_val = arange if case == 1 else brange if case == 2 else grange
    plt.title(f'{excluded}={excluded_val}, max={utility[max_index]:.2f}')
    plt.show()
    return utility, (max1, max2)


def investor_strategy_modified(T, mu, y0, interest_rate, max_approp=0.1,
                               arange=(-1.5, 0.5), brange=(-0.1, 0.2), grange=(-1,1)):
    def objective(alpha, beta, gamma):
        return vect_investor_util_mod(T, mu, y0, alpha, beta, gamma, interest_rate, max_approp)
    return nd_grid_maximize(objective, arange, brange, grange)

# investor_strategy_modified(100, 0.03, 1, 0.02, 0.5) returns \
    # ((-0.9244928625093914, -0.02922614575507137, -0.10743801652892562), array(4.90107199))
    # Why are they all negative? That's bizzare. The objective value seems good though... This function behaves weird.