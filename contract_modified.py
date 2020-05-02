#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 22:23:35 2020

@author: mckay
"""

import numpy as np
from numpy.polynomial import polynomial
from scipy.optimize import minimize, fminbound, differential_evolution
from scipy.special import expit

from sklearn.preprocessing import PolynomialFeatures

from time import time

DEFAULT_POLYDEGREE = 3



class AppropRule:
    '''A model meant to approximate the manager's optimal appropriation rule.

    The predict function will construct a matrix of features X based on a vector of reported firm values
        and a vector of transition ratios, multiply X by the model coefficients,
        put the output through an inverse logit function, and then scale to the range [0, max_approp].
    The features in X are polynomial features of degree polydegree.
    '''
    
    def __init__(self, max_approp, polydegree=DEFAULT_POLYDEGREE):
        self.max_approp = max_approp
        self.polydegree = polydegree
        ncoeffs = (polydegree + 1)*(polydegree + 2) // 2  # ncoeffs grows exponentially with degree
        self.coeffs = np.ones(ncoeffs)
        self.polyFeatures = PolynomialFeatures(degree=polydegree)
        self.polyFeatures.fit([[0,0]])  # will be handed two features
    
    def predict(self, yt, rt, coeffs=None):
        if coeffs is None:
            coeffs = self.coeffs
        X = self.polyFeatures.transform(np.column_stack((yt, yt*rt)))
        prediction = expit(X.dot(coeffs)) * self.max_approp  # scale to [0, max_approp]
        return prediction
    


class ModifiedContract:
    '''Contains variables and functions for modified version of contract problem
    
    Y0 is the initial value of the firm
    mu is the growth rate of the firm (all rates in terms of growth per period)
    T is the length (in periods) of the contract
    rho is the manager's interest rate
    r is the investor's interest rate (riskless rate, should be less than or equal to rho)
    theta is the efficiency with which the manager can appropriate firm value, in [0,1]
    lambda_ (optional) is the parameter for the poisson component of jumps in firm value (expected #jumps per period)
    m is the mean of jumps in the firm value
    s2 is the variance of jumps in the firm value
    max_approp is the maximum amount that the manager can appropriate in a single period
    manager_threshold is the minimum expected payoff that a manager will accept in a contract
    use_shortcut is Boolean, whether to approximate appropriation strategy with "short-sighted" strategy
        or attempt to find better strategy using fit_approp_rules
    '''
    
    def __init__(self, Y0=1, mu=0.01, sigma=0.1, T=20, rho=0.03, r=None, theta=1, lambda_=None, m=0, s2=1,
                 max_approp=0.1, manager_threshold=0, use_shortcut=False):
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
        self.use_shortcut = use_shortcut
        # ratio is the expected ratio between firm values in subsequent periods
        self.ratio = np.exp(self.mu + self.lambda_*(np.exp(self.m + self.s2/2) - 1)) \
                        if self.lambda_ else np.exp(self.mu)
    
    
    def fit_approp_rules(self, coeffs, npaths=100):
        '''Estimates optimal appropriation rules to maximize the manager's utility.
        '''
        print(f'fitting approp rules with {coeffs}...')
        time0 = time()
        # Draw transition ratios
        Nts = np.random.poisson(self.lambda_, size=(npaths, self.T)) if self.lambda_ else np.zeros((npaths, self.T))
        ratios = np.exp(self.mu - self.sigma**2 / 2 \
                        + np.random.normal(loc=Nts*self.m, scale=np.sqrt(self.sigma**2 + Nts*self.s2)))
        # calculate fixed values
        discounts = np.exp(-self.rho * np.arange(self.T+1))
        # calculate endpoints
        Yt = self.Y0 * ratios.prod(axis=1)
        models = [AppropRule(self.max_approp) for _ in range(self.T)]
        for t in reversed(range(1, self.T+1)):
            # define objective
            def objective(approp_coeffs):
                # will sum up "out", the continuation utility for a given approp
                approp = models[t-1].predict(Yt, ratios[:,t-1], approp_coeffs)
                out = npaths * (self.theta * approp \
                                + polynomial.polyval(Yt*(ratios[:,t-1] - self.ratio) - approp, coeffs))
                for s in range(t+1, self.T+1):
                    new_Yt = np.prod(ratios[:,t:(s-1)], axis=1)*Yt
                    approp = models[s-1].predict(new_Yt, ratios[:,s-1])
                    out += discounts[s-t] * (self.theta * approp \
                            + polynomial.polyval(new_Yt*(ratios[:,s-1] - self.ratio) - approp, coeffs))
                # minimize sum of utilities
                return - np.sum(out)
            models[t-1].coeffs = minimize(objective, x0=models[t-1].coeffs).x
        print(f'models fit in {time()-time0:.2f}')
        return models
    
    
    def simulate_runs(self, coeffs, nruns=100, models=None):
        '''Fits appropriation rule models and simulates optimal runs using those models
        '''
        if models is None:
            models = self.fit_approp_rules(coeffs)
        runs = np.ones((nruns, self.T+1)) * self.Y0
        approps = np.zeros((nruns, self.T+1))
        Nts = np.random.poisson(self.lambda_, size=(nruns, self.T)) if self.lambda_ else np.zeros((nruns, self.T))
        ratios = np.exp(self.mu - self.sigma**2 / 2 \
                             + np.random.normal(loc=Nts*self.m, scale=np.sqrt(self.sigma**2 + Nts*self.s2)))
        for t in range(1, self.T+1):
            approps[:, t] = models[t-1].predict(runs[:,t-1], ratios[:,t-1])
            runs[:, t] = runs[:,t-1]*ratios[:,t-1] - approps[:, t]
        return runs, approps
    
    
    def simulate_runs_shortcut(self, coeffs, nruns=100):
        '''Simulates runs where manager is "short-sighted" (i.e. maximizes current payoff at every step)
        '''
        runs = np.ones((nruns, self.T+1)) * self.Y0
        approps = np.zeros((nruns, self.T+1))
        Nts = np.random.poisson(self.lambda_, size=(nruns, self.T)) if self.lambda_ else np.zeros((nruns, self.T))
        ratios = np.exp(self.mu - self.sigma**2 / 2 \
                             + np.random.normal(loc=Nts*self.m, scale=np.sqrt(self.sigma**2 + Nts*self.s2)))
        def objective(approp, yt, rt):
            return - self.theta * approp - polynomial.polyval(yt*rt - approp - yt*self.ratio, coeffs)
        for t in range(1, self.T+1):
            for n in range(nruns):
                approps[n,t] = fminbound(objective, 0, self.max_approp, args=(runs[n,t-1], ratios[n,t-1]))
            runs[:,t] = runs[:,t-1]*ratios[:,t-1] - approps[:,t]
        return runs, approps
        
    
    def calculate_utilities(self, coeffs):
        '''Calculates utilities of simulated runs under optimal appropriation assumption
        '''
        if self.use_shortcut:
            runs, approps = self.simulate_runs_shortcut(coeffs)
        else:
            runs, approps = self.simulate_runs(coeffs)
        one_step_expectations = runs[:,:-1] * self.ratio
        payments = polynomial.polyval(runs[:,1:] - one_step_expectations, coeffs)
        manager_discounts = np.exp(-self.rho * np.arange(1, self.T+1))
        investor_discounts = np.exp(-self.r * np.arange(1, self.T+1))
        manager_utils = np.sum((payments + self.theta * approps[:,1:]) * manager_discounts, axis=1)
        investor_utils = np.sum((np.diff(runs) - payments) * investor_discounts, axis=1)
        return manager_utils, investor_utils
    
    
    def calculate_expected_utilities(self, coeffs):
        '''Determines manager and investor utility under payment scheme with coeffs.
        '''
        #print(f'calculating utilities with {coeffs}...')
        manager_utils, investor_utils = self.calculate_utilities(coeffs)
        return manager_utils.mean(), investor_utils.mean()
    
    
    def calculate_investor_utility(self, coeffs):
        '''Determines investor utility under payment scheme with coeffs.
        Will return -inf if scheme does not meet manager threshold, since such a contract is unacceptable.
        '''
        manager_util, investor_util = self.calculate_expected_utilities(coeffs)
        return investor_util if manager_util >= self.manager_threshold else -np.inf
    
    
    def find_best_contract(self, bounds, popsize=10, maxiter=10):
        '''Uses differential evolution algorithm to attempt to find best contract for investor.
        
        bounds should be a list of 2-tuples, with length equal to degree of [polynomial] payment function
        the maximum number of iterations required is len(bounds)*popsize*maxiter

        Be aware that this can a take a long time to compute (on the order of multiple hours if not using shortcut)
        '''
        def objective(coeffs):
            return -self.calculate_investor_utility(coeffs)
        optim = differential_evolution(objective, bounds, popsize=popsize, maxiter=maxiter, disp=True)
        return optim
    
