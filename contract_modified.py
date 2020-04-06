#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 22:23:35 2020

@author: mckay
"""

import numpy as np
from numpy.polynomial import polynomial
from scipy.optimize import fminbound
from scipy.special import logit, expit

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class ModifiedContract:
    
    def __init__(self, Y0=1, mu=0.01, sigma=0.1, T=20, rho=0.03, r=None, theta=1, lambda_=None, m=0, s2=1,
                 max_approp=0.1, manager_threshold=0):
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
        self.ratio = np.exp(self.mu + self.lambda_*(np.exp(self.m + self.s2/2) - 1)) \
                        if self.lambda_ else np.exp(self.mu)

    def create_optimal_path(self, YT, coeffs):
        Y = np.zeros(self.T+1)
        Y[self.T] = YT
        approp = np.zeros(self.T+1)
        def objective(x, y_t1, ratio_t):
            return -x - polynomial.polyval(y_t1 - (y_t1 + x)*self.ratio / ratio_t, coeffs)
        for t in reversed(range(1, self.T+1)):
            Nt = np.random.poisson(self.lambda_) if self.lambda_ else 0
            ratio_t = np.exp(self.mu - self.sigma**2 / 2 \
                             + np.random.normal(loc=Nt*self.m, scale=np.sqrt(self.sigma**2 + Nt*self.s2)))
            approp[t] = fminbound(objective, 0, self.max_approp, args=(Y[t], ratio_t))
            Y[t-1] = (Y[t] + approp[t]) / ratio_t
        return Y, approp
    
    def create_many_paths(self, coeffs, npaths=100):
        paths = np.zeros((npaths, self.T+1))
        approps = np.zeros((npaths, self.T+1))
        # Guess an endpoint that will give startpoint near Y0
        endpoint = self.Y0 * np.exp(self.mu * self.T + self.lambda_ * self.T * (np.exp(self.m + self.s2/2) - 1)) \
                    if self.lambda_ else self.Y0 * np.exp(self.mu * self.T)
        for i in range(npaths):
            paths[i,:], approps[i,:] = self.create_optimal_path(endpoint, coeffs)
            endpoint /= paths[i, 0]
        return paths, approps
    
    def fit_approp_rules(self, coeffs, npaths=100, poly_degree=3):
        paths, approps = self.create_many_paths(coeffs, npaths)
        models = []
        for t in range(1, self.T+1):
            model = Pipeline([('poly', PolynomialFeatures(degree=poly_degree)),
                              ('linear', LinearRegression())])
            offset = np.zeros((npaths, 2))
            offset[:, 1] = approps[:,t]
            X = paths[:,(t-1):(t+1)] + offset
            y = logit(approps[:,t] / self.max_approp)
            model.fit(X, y)
            models.append(model)
        return models
    
    def simulate_runs(self, coeffs, nruns=100, models=None):
        if models is None:
            models = self.fit_approp_rules(coeffs)
        runs = np.ones((nruns, self.T+1)) * self.Y0
        approps = np.zeros((nruns, self.T+1)) * self.Y0
        Nts = np.random.poisson(self.lambda_, size=(nruns, self.T)) if self.lambda_ else np.zeros((nruns, self.T))
        ratios = np.exp(self.mu - self.sigma**2 / 2 \
                             + np.random.normal(loc=Nts*self.m, scale=np.sqrt(self.sigma**2 + Nts*self.s2)))
        for t in range(1, self.T+1):
            runs[:, t] = runs[:, t-1] * ratios[:, t-1]
            approps[:, t] = expit(models[t-1].predict(runs[:,(t-1):(t+1)]))*self.max_approp
            runs[:, t] -= approps[:, t]
        return runs, approps
    
    def calculate_utilities(self, coeffs):
        runs, approps = self.simulate_runs(coeffs)
        one_step_expectations = runs[:,:-1] * self.ratio
        payments = polynomial.polyval(runs[:,1:] - one_step_expectations, coeffs)
        manager_discounts = np.exp(-self.rho * np.arange(1, self.T+1))
        investor_discounts = np.exp(-self.r * np.arange(1, self.T+1))
        manager_utils = np.sum((payments + approps[:,1:]) * manager_discounts, axis=1)
        investor_utils = np.sum((np.diff(runs) - payments) * investor_discounts, axis=1)
        return manager_utils, investor_utils
    