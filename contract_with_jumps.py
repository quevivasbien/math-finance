#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:05:10 2020

@author: mckay

works similar to contract_problem.py but treats Yt as a continuous-time process allows for jumps in Yt
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm


class Contract:
    
    def __init__(self, Y0, mu, T, rho, r=None, lambda_=None, m=None, s2=None, max_approp=0.5, manager_threshold=0):
        self.Y0 = Y0
        self.mu = mu
        self.T = T
        self.rho = rho
        self.r = r if (r is not None) else rho
        self.lambda_ = lambda_
        self.m = m
        self.s2 = s2
        self.max_approp = max_approp
        self.manager_threshold = manager_threshold
        self.Y_exp = None
    
    def estimate_exp_Jt(self, t, samples=10000):
        '''Monte Carlo approach to estimate the expected value of exp(J_t)
        J_t = sum of Nt N(m, s2) random variables, where Nt ~ Poisson(lambda_*t)
        '''
        Nts = poisson.rvs(self.lambda_*t, size=samples)
        samples = norm.rvs(loc=Nts*self.m, scale=np.sqrt(Nts*self.s2))
        return np.exp(samples).mean()
    
    def estimate_Yt(self, t):
        '''estimates Yt given information at time 0
        '''
        if self.lambda_ is None:
            return self.Y0 * np.exp(self.mu * t)
        else:
            return self.Y0 * np.exp(self.mu * t) * self.estimate_exp_Jt(t)
    
    def get_Y_exp(self):
        '''determines expectation of Y for t = 0, ..., T, based on info at time 0
        '''
        if self.Y_exp is None:
            self.Y_exp = np.array([self.estimate_Yt(t) for t in range(self.T+1)])
        return self.Y_exp
    
    def manager_strategy(self, beta, gamma):
        '''determines the manager's appropriation strategy for a given beta and gamma'''
        approp = np.zeros(self.T+1)
        for t in range(1, self.T+1):
            if gamma + beta * sum(np.exp(-self.rho*(s-t)) for s in range(t+1, self.T+1)) < 1:
                approp[t] = self.max_approp
        return approp
    
    def manager_utility0(self, alpha, beta, gamma, approp=None, get_y_exp=False):
        '''determines manager's expected utility for a given contract
        '''
        if approp is None:
            approp = self.manager_strategy(beta, gamma)
        cumulative_approp = np.cumsum(approp)
        Y_exp = self.get_Y_exp()
        utility = sum(np.exp(-self.rho * t) * (
                    alpha + beta*(Y_exp[t-1] - cumulative_approp[t-1]) \
                    + gamma*(Y_exp[t] - Y_exp[t-1] - approp[t]) \
                    + approp[t]) for t in range(1, self.T+1)
                )
        return utility
    
    def investor_utility0(self, alpha, beta, gamma):
        approp = self.manager_strategy(beta, gamma)
        cumulative_approp = np.cumsum(approp)
        manager_util = self.manager_utility0(alpha, beta, gamma, approp)
        if manager_util <= self.manager_threshold: # these cases are unacceptable contracts
            return -np.inf
        Y_exp = self.get_Y_exp()
        # loss is loss due to payments/appropriation to manager. is the same as manager_util if r == rho
        loss = manager_util if self.r == self.rho else \
                sum(np.exp(-self.r * t) * (
                    alpha + beta*(Y_exp[t-1] - cumulative_approp[t-1]) \
                    + gamma*(Y_exp[t] - Y_exp[t-1] - approp[t]) \
                    + approp[t]) for t in range(1, self.T+1)
                )
        investor_util = sum(np.exp(-self.r * t) * (Y_exp[t] - Y_exp[t-1]) for t in range(1, self.T+1)) - loss
        return investor_util
    
    def plot_investor_utility(self, arange, brange, grange, resolution=20, show_max=True):
        '''
        Creates a heatmap of the modified investor's utility function across two variables, with the other held fixed
        
        One of arange, brange, grange must be a scalar; the others must be size-2 tuples representing ranges to plot over
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
        vect_investor_utility0 = np.vectorize(self.investor_utility0, otypes=[float])
        if case == 1:
            utility = vect_investor_utility0(arange, xx, yy)
        elif case == 2:
            utility = vect_investor_utility0(xx, brange, yy)
        elif case == 3:
            utility = vect_investor_utility0(xx, yy, grange)
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