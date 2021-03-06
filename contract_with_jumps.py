#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:05:10 2020

@author: mckay

works similar to contract_problem.py but treats Yt as a continuous-time process and allows for jumps in Yt
"""

import numpy as np
import matplotlib.pyplot as plt


class Contract:
    '''Contains variables and functions for general case of contract problem
    
    Y0 is the initial value of the firm
    mu is the growth rate of the firm
    T is the lenght of the contract
    rho is the manager's interest rate
    r is the investor's interest rate (riskless rate, should be less than or equal to rho)
    theta is the efficiency with which the manager can appropriate firm value
    lambda_ (optional) is the parameter for the poisson component of jumps in firm value
    m is the mean of jumps in the firm value
    s2 is the variance of jumps in the firm value
    max_approp is the maximum amount that the manager can appropriate in a single period
    manager_threshold is the minimum expected payoff that a manager will accept in a contract
    '''
    
    def __init__(self, Y0, mu, T, rho, r=None, theta=1, lambda_=None, m=None, s2=None, max_approp=0.5, manager_threshold=0):
        self.Y0 = Y0
        self.mu = mu
        self.T = T
        self.rho = rho
        self.r = r if (r is not None) else rho
        self.theta = theta
        self.lambda_ = lambda_
        self.m = m
        self.s2 = s2
        self.max_approp = max_approp
        self.manager_threshold = manager_threshold
        self.Y_exp = None
    
    def expected_Yt(self, t):
        '''computes expected value of Yt given information at time 0
        '''
        if self.lambda_ is None:
            # assume no jumps
            return self.Y0 * np.exp(self.mu * t)
        else:
            # multiply by effect from jumps
            return self.Y0 * np.exp(self.mu * t + self.lambda_ * t * (np.exp(self.m + self.s2/2) - 1))
    
    def get_Y_exp(self):
        '''determines expectation of Y for t = 0, ..., T, based on info at time 0
        '''
        if self.Y_exp is None:
            self.Y_exp = np.array([self.expected_Yt(t) for t in range(self.T+1)])
        return self.Y_exp
    
    def manager_strategy(self, beta, delta):
        '''determines the manager's appropriation strategy for a given beta and delta'''
        approp = np.zeros(self.T+1)
        for t in range(1, self.T+1):
            if delta + beta * sum(np.exp(-self.rho*(s-t)) for s in range(t, self.T+1)) < self.theta:
                approp[t] = self.max_approp
        return approp
    
    def manager_utility0(self, alpha, beta, delta, approp=None):
        '''determines manager's expected utility for a given contract
        '''
        if approp is None:
            approp = self.manager_strategy(beta, delta)
        cumulative_approp = np.cumsum(approp)
        Y_exp = self.get_Y_exp()
        utility = sum(np.exp(-self.rho * t) * (
                    alpha + beta*(Y_exp[t] - cumulative_approp[t]) \
                    + delta*(Y_exp[t] - Y_exp[t-1] - approp[t]) \
                    + self.theta * approp[t]) for t in range(1, self.T+1)
                )
        return utility
    
    def investor_utility0(self, alpha, beta, delta):
        approp = self.manager_strategy(beta, delta)
        cumulative_approp = np.cumsum(approp)
        manager_util = self.manager_utility0(alpha, beta, delta, approp)
        if manager_util <= self.manager_threshold: # these cases are unacceptable contracts
            return -np.inf
        Y_exp = self.get_Y_exp()
        # loss is loss due to payments/appropriation to manager. is the same as manager_util if r == rho
        loss = manager_util if self.r == self.rho else \
                sum(np.exp(-self.r * t) * (
                    alpha + beta*(Y_exp[t] - cumulative_approp[t]) \
                    + delta*(Y_exp[t] - Y_exp[t-1] - approp[t]) \
                    + approp[t]) for t in range(1, self.T+1)
                )
        investor_util = sum(np.exp(-self.r * t) * (Y_exp[t] - Y_exp[t-1]) for t in range(1, self.T+1)) - loss
        return investor_util
    
    def plot_investor_utility(self, arange, brange, drange, resolution=20, show_max=True):
        '''
        Creates a heatmap of the investor's utility function across two variables, with the other held fixed
        
        One of arange, brange, drange must be a scalar; the others must be size-2 tuples representing ranges to plot over
        resolution is the the number of points to evaluate -- the result will be a resolution x resolution matrix/heatmap
        '''
        if not isinstance(arange, (tuple, list)):
            case = 1
            a1, b1 = brange
            a2, b2 = drange
        elif not isinstance(brange, (tuple, list)):
            case = 2
            a1, b1 = arange
            a2, b2 = drange
        elif not isinstance(drange, (tuple, list)):
            case = 3
            a1, b1 = arange
            a2, b2 = brange
        else:
            print('one of arange, brange, drange must be scalar; the others must be size-2 tuples')
            return np.array([]), (0, 0)
        x = np.linspace(a1, b1, resolution)
        y = np.linspace(a2, b2, resolution)
        xx, yy = np.meshgrid(x, y)
        vect_investor_utility0 = np.vectorize(self.investor_utility0, otypes=[float])
        if case == 1:
            utility = vect_investor_utility0(arange, xx, yy)
        elif case == 2:
            utility = vect_investor_utility0(xx, brange, yy)
        elif case == 3:
            utility = vect_investor_utility0(xx, yy, drange)
        plt.imshow(np.flip(utility, 0), cmap=plt.cm.Reds, extent=[a1, b1, a2, b2])
        plt.colorbar()
        plt.xlabel('alpha' if case != 1 else 'beta')
        plt.ylabel('delta' if case != 3 else 'beta')
        max_index = np.unravel_index(utility.argmax(), utility.shape)
        max1, max2 = x[max_index[1]], y[max_index[0]]
        if show_max:
            plt.scatter(max1, max2, marker='x')
        excluded = 'alpha' if case == 1 else 'beta' if case == 2 else 'delta'
        excluded_val = arange if case == 1 else brange if case == 2 else drange
        plt.title(f'{excluded}={excluded_val}, max={utility[max_index]:.2f}')
        plt.show()
        return utility, (max1, max2)