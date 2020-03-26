#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:02:22 2020

@author: mckay
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
    
    def __init__(self, Y0=1, mu=0.01, sigma=0.1, T=20, rho=0.03, r=None, theta=1, lambda_=None, m=0, s2=1,
                 max_approp=0.1, manager_threshold=0,
                 realizations=None):
        self.Y0 = Y0
        self.mu = mu
        self.sigma = sigma
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
        self.realizations = realizations  # can be provided upon init, or just use load_realizations
            
    
    def expected_Yt(self, t, t0=0, Yt0=None):
        '''computes expected value of Yt given informationscharfenaker  at time zero, with Yt0 given
        '''
        if t0 == 0 and Yt0 is None:
            Yt0 = self.Y0
        if self.lambda_ is None:
            # assume no jumps
            return Yt0 * np.exp(self.mu * (t-t0))
        else:
            # multiply by effect from jumps
            return Yt0 * np.exp(self.mu * t + self.lambda_ * t * (np.exp(self.m + self.s2/2) - 1))
        
    def get_Y_exp(self):
        '''determines expectation of Y for t = 0, ..., T, based on info at time 0
        '''
        if self.Y_exp is None:
            self.Y_exp = np.array([self.expected_Yt(t) for t in range(self.T+1)])
        return self.Y_exp

    def load_realizations(self, n=1000, dt=1):  # Increasing dt doesn't help at all how I have it set up currently
        # Simulates n Yt realizations of the form dYt = Yt*(mu*dt + sigma*dZt + dJt)
        steps_per_period = int(1 / dt)
        if steps_per_period != 1 / dt:
            print('Error: dt must evenly divide 1')
            return
        nsteps = int(self.T / dt)
        Yts = np.ones((n, nsteps+1)) * self.Y0
        # dZt is a Brownian motion step, i.e. N(0, dt)
        dZts = np.random.normal(scale=np.sqrt(dt), size=(n, nsteps))
        Zts = np.cumsum(dZts, axis=1)
        Jts = np.zeros((n, nsteps))
        if self.lambda_ is not None:
            # dJt is a sum of dNt N(m, s2) random variables, where dNt ~ Poisson(lambda_*dt)
            dNts = np.random.poisson(self.lambda_*dt, size=(n, nsteps))
            dJts = np.random.normal(loc=dNts*self.m, scale=np.sqrt(dNts*self.s2))
            Jts = np.cumsum(dJts, axis=1)
        ts = np.tile(np.arange(1, nsteps+1)*dt, (n, 1))
        Yts[:,1:] = Yts[:,1:] * np.exp(self.mu * ts - (self.sigma**2 / 2) * ts + self.sigma * Zts + Jts)
        # Compare mean of Yts[:-1] to analytical expected value for diagnostic purposes
        simulated_mean = Yts[:,-1].mean()
        analytical_mean = self.expected_Yt(self.T)
        print(f'error is {simulated_mean - analytical_mean}\n' \
              + f'simulated: {simulated_mean}, analytical: {analytical_mean}')
        # return only the entries that correspond to whole periods
        self.realizations = Yts[:,0::steps_per_period]
        
    def estimate_expected_approp(self, beta, gamma, delta, approp=None):
        '''estimates a time series of the expected value of the manager's appropriation at each step
        works backward on each simulated realization to determine optimal approp for that realization
        '''
        if self.realizations is None:
            self.load_realizations()
        n = len(self.realizations)
        if approp is None:
            approp = np.zeros((n, self.T+1))
        for i in range(n):
            for t in range(1, self.T+1):
                if delta + sum(np.exp(-self.rho*(s-t)) * (beta \
                               + gamma*(2*self.expected_Yt(s-1, t, self.realizations[i,t]) - self.max_approp \
                                        - 2*sum(approp[i,u] for u in range(1, t)) - 2*(s-t-1)*self.max_approp)) \
                               for s in range(t+1, self.T+1)) < self.theta:
                    approp[i,t] = self.max_approp
        return approp.mean(axis=0)
    
    def manager_utility0(self, alpha, beta, gamma, delta, approp=None):
        '''determines manager's expected utility based on information at time 0 for a given contract
        '''
        if approp is None:
            approp = self.estimate_expected_approp(beta, gamma, delta)
        cumulative_approp = np.cumsum(approp)
        Y_exp = self.get_Y_exp()
        Y_reported = Y_exp - cumulative_approp
        utility = sum(np.exp(-self.rho*t) * (
                    alpha + beta*Y_reported[t-1] + gamma*Y_reported[t-1]**2 \
                    + delta*(Y_exp[t] - Y_exp[t-1] - approp[t]) \
                    + self.theta * approp[t]) for t in range(1, self.T+1)
                )
        return utility
    
    def investor_utility0(self, alpha, beta, gamma, delta, approp=None):
        if approp is None:
            approp = self.estimate_expected_approp(beta, gamma, delta)
        cumulative_approp = np.cumsum(approp)
        manager_util = self.manager_utility0(alpha, beta, gamma, delta, approp)
        if manager_util <= self.manager_threshold:  # these cases are unacceptable contracts
            return -np.inf
        Y_exp = self.get_Y_exp()
        Y_reported = Y_exp - cumulative_approp
        # loss is loss due to payments/appropriation to manager. is the same as manager_util if r == rho
        loss = manager_util if self.r == self.rho else \
                sum(np.exp(-self.r*t) * (
                    alpha + beta*Y_reported[t-1] + gamma*Y_reported[t-1]**2 \
                    + delta*(Y_exp[t] - Y_exp[t-1] - approp[t]) \
                    + self.theta * approp[t]) for t in range(1, self.T+1)
                )
        investor_util = sum(np.exp(-self.r * t) * (Y_exp[t] - Y_exp[t-1]) for t in range(1, self.T+1)) - loss
        return investor_util
    
    def plot_investor_utility(self, arange, brange, grange, drange, resolution=20, show_max=True):
        '''
        Creates a heatmap of the investor's utility function across two variables, with the others held fixed
        
        Two of arange, brange, grange, drange must be scalars; the others must be size-2 tuples representing ranges to plot over
        resolution is the the number of points to evaluate -- the result will be a resolution x resolution matrix/heatmap
        '''
        variables = (arange, brange, grange, drange)
        ranges = tuple(np.where([isinstance(x, (tuple, list)) for x in variables])[0])
        fixed = tuple(np.where([not isinstance(x, (tuple, list)) for x in variables])[0])
        print(ranges)
        print(fixed)
        if len(ranges) != 2 or len(fixed) != 2:
            print('two of arange, brange, grange, drange should be ranges, and the other two should be scalars.')
            return np.array([]), (0,0)
        a1, b1 = variables[ranges[0]]
        a2, b2 = variables[ranges[1]]
        x = np.linspace(a1, b1, resolution)
        y = np.linspace(a2, b2, resolution)
        xx, yy = np.meshgrid(x, y)
        vect_investor_utility0 = np.vectorize(self.investor_utility0, otypes=[float])
        alpha = xx if 0 in ranges else arange
        beta = xx if (0 not in ranges and 1 in ranges) else (yy if 1 in ranges else brange)
        gamma = xx if (0 not in ranges and 1 not in ranges) else (yy if 2 in ranges else grange)
        delta = yy if 3 in ranges else drange
        utility = vect_investor_utility0(alpha, beta, gamma, delta)
        plt.imshow(np.flip(utility, 0), cmap=plt.cm.Reds, extent=[a1, b1, a2, b2])
        plt.colorbar()
        labels = ['alpha', 'beta', 'gamma', 'delta']
        plt.xlabel(labels[ranges[0]])
        plt.ylabel(labels[ranges[1]])
        max_index = np.unravel_index(utility.argmax(), utility.shape)
        max1, max2 = x[max_index[1]], y[max_index[0]]
        if show_max:
            plt.scatter(max1, max2, marker='x')
        excluded1 = labels[fixed[0]]
        excluded2 = labels[fixed[1]]
        excluded_val1 = (alpha, beta, gamma, delta)[fixed[0]]
        excluded_val2 = (alpha, beta, gamma, delta)[fixed[1]]
        plt.title(f'{excluded1}={excluded_val1}, {excluded2}={excluded_val2}, max={utility[max_index]:.2f}')
        plt.show()
        return utility, (max1, max2)