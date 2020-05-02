#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 11:55:25 2020

@author: mckay
"""

import torch
import torch.nn as nn
import torch.distributions as dist
import matplotlib.pyplot as plt

from time import time

DEFAULT_POLYDEGREE = 3


def poly_eval(x, coeffs):
    """Evaluate 1d polynomial defined by coeffs at point x; coeffs are in order of ascending degree"""
    return torch.stack([coeffs[i] * x ** i for i in range(len(coeffs))]).sum(axis=0)
    


class PolyFeatures:
    """Replicates basic sklearn PolynomialFeatures functionality with torch tensors"""
    def __init__(self, input_size, degree=DEFAULT_POLYDEGREE, include_bias=True):
        self.input_size = input_size
        self.degree = degree
        self.include_bias = include_bias
        self.output_size = (self.degree + 1) * (self.degree + 2) // 2 - int(not self.include_bias)
    
    def transform(self, X):
        assert X.size(-1) == self.input_size, 'Number of input features must match self.input_size'
        XP = torch.empty(X.size(-2), self.output_size, dtype=X.dtype)
        if self.include_bias:
            XP[:, 0] = 1
            current_col = 1
        else:
            current_col = 0
        XP[:, current_col:current_col + self.input_size] = X
        index = list(range(current_col, current_col + self.input_size + 1))
        current_col += self.input_size
        for _ in range(1, self.degree):
            new_index = []
            end = index[-1]
            for feature_index in range(self.input_size):
                start = index[feature_index]
                new_index.append(current_col)
                next_col = current_col + end - start
                if next_col <= current_col:
                    break
                XP[:, current_col:next_col] = XP[:, start:end] * X[:, feature_index].unsqueeze(-1)
                current_col = next_col
            new_index.append(current_col)
            index = new_index
        return XP


class AppropRule(nn.Module):
    """Simple linear predictor with scaled sigmoid output"""
    def __init__(self, max_approp, input_size=2, polydegree=DEFAULT_POLYDEGREE):
        super().__init__()
        self.max_approp = max_approp
        self.polyFeatures = PolyFeatures(input_size, degree=polydegree)
        self.linear = nn.Linear(self.polyFeatures.output_size, 1)
        self.expit = nn.Sigmoid()
    
    def forward(self, yt, rt):
        XP = self.polyFeatures.transform(torch.stack((yt, yt*rt), dim=1))
        prediction = self.max_approp * self.expit(self.linear(XP))
        return prediction
        
    def predict(self, yt, rt):
        with torch.no_grad():
            return self.forward(yt, rt)


class ModifiedContract:
    """Contains variables and functions for modified version of contract problem
    
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
    """
    
    def __init__(self, Y0=1, mu=0.01, sigma=0.1, T=20, rho=0.03, r=None, theta=1, lambda_=None, m=0, s2=1,
                 max_approp=0.1, manager_threshold=0, use_shortcut=False):
        self.Y0 = torch.tensor(Y0)
        self.mu = torch.tensor(mu)
        self.sigma = torch.tensor(sigma)
        self.T = T
        self.rho = torch.tensor(rho)
        self.r = torch.tensor(r) if (r is not None) else self.rho
        self.theta = torch.tensor(theta)
        self.lambda_ = torch.tensor(lambda_) if lambda_ is not None else None
        self.m = torch.tensor(m)
        self.s2 = torch.tensor(s2)
        self.max_approp = torch.tensor(max_approp)
        self.manager_threshold = manager_threshold
        self.use_shortcut = use_shortcut
        # ratio is the expected ratio between firm values in subsequent periods
        self.ratio = torch.exp(self.mu + self.lambda_*(torch.exp(self.m + self.s2/2) - 1)) \
                        if self.lambda_ else torch.exp(self.mu)
                            
    def fit_approp_rules(self, coeffs, npaths=100, lr=0.02, epochs_per_step=30):
        """Estimates optimal appropriation rules to maximize the manager's utility.
        """
        print(f'fitting approp rules with {coeffs}...')
        time0 = time()
        # Draw transition ratios
        Nts = dist.Poisson(self.lambda_).sample((npaths, self.T)) if self.lambda_ else torch.zeros(npaths, self.T)
        ratios = torch.exp(self.mu - self.sigma**2 / 2 \
                           + dist.Normal(loc=Nts*self.m, scale=torch.sqrt(self.sigma**2 + Nts*self.s2)).sample())
        # calculate [fixed] future value discounts
        discounts = torch.exp(-self.rho * torch.arange(self.T+1))
        # calculate endpoints
        Yt = self.Y0 * ratios.prod(axis=1)
        models = [AppropRule(self.max_approp) for _ in range(self.T)]
        for t in reversed(range(1, self.T+1)):
            def loss_fn(approp):
                utils = npaths * (self.theta * approp \
                                  + poly_eval(Yt*(ratios[:, t-1] - self.ratio) - approp, coeffs))
                for s in range(t+1, self.T+1):
                    new_Yt = ratios[: t:(s-1)].prod(axis=1) * Yt
                    approp = models[s-1].predict(new_Yt, ratios[:, s-1])
                    utils += discounts[s-t] * (self.theta * approp \
                                               + poly_eval(new_Yt * (ratios[:, s-1] - self.ratio) - approp, coeffs))
                return -utils.mean()
            optimizer = torch.optim.Adam(models[t-1].parameters(), lr=lr)
            losses = []
            for i in range(epochs_per_step):
                approp = models[t-1].forward(Yt, ratios[:, t-1])
                loss = loss_fn(approp)
                losses.append(loss.detach().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            plt.plot(losses)
        plt.show()
        print(f'models fit in {time()-time0:.2f}')
        self.models = models
        return models
    
    def simulate_runs(self, coeffs, nruns=100, models=None):
        """Fits appropriation rule models and simulates optimal runs using those models
        """
        if models is None:
            models = self.fit_approp_rules(coeffs)
        runs = torch.ones(nruns, self.T+1) * self.Y0
        approps = torch.zeros(nruns, self.T+1)
        Nts = dist.Poisson(self.lambda_).sample((nruns, self.T)) if self.lambda_ else torch.zeros(nruns, self.T)
        ratios = torch.exp(self.mu - self.sigma**2 / 2 \
                           + dist.Normal(loc=Nts*self.m, scale=torch.sqrt(self.sigma**2 + Nts*self.s2)).sample())
        for t in range(1, self.T+1):
            approps[:, t] = models[t-1].predict(runs[:,t-1], ratios[:,t-1]).flatten()
            runs[:, t] = runs[:, t-1] * ratios[:, t-1] - approps[:, t]
        return runs, approps
    
    def calculate_utilities(self, coeffs):
        """Calculates utilities of simulated runs under optimal appropriation assumption
        """
        runs, approps = self.simulate_runs(coeffs)
        one_step_expectations = runs[:, :-1] * self.ratio
        payments = poly_eval(runs[:, 1:] - one_step_expectations, coeffs)
        manager_discounts = torch.exp(-self.rho * torch.arange(1, self.T+1))
        investor_discounts = torch.exp(-self.r * torch.arange(1, self.T+1))
        manager_utils = torch.sum((payments + self.theta * approps[:,1:]) * manager_discounts, axis=1)
        investor_utils = torch.sum((runs[:, 1:] - runs[:, :-1] - payments) * investor_discounts, axis=1)
        return manager_utils, investor_utils
    
    def calculate_expected_utilities(self, coeffs):
        """Determines manager and investor utility under payment scheme with coeffs.
        """
        manager_utils, investor_utils = self.calculate_utilities(coeffs)
        return manager_utils.mean(), investor_utils.mean()
    
    
    def calculate_investor_utility(self, coeffs):
        '''Determines investor utility under payment scheme with coeffs.
        Will return -inf if scheme does not meet manager threshold, since such a contract is unacceptable.
        '''
        manager_util, investor_util = self.calculate_expected_utilities(coeffs)
        return investor_util if manager_util >= self.manager_threshold else torch.tensor(float('-inf'))
    