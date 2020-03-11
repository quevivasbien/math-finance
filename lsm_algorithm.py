#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 08:34:26 2020

@author: mckay
"""

import numpy as np
#import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import sys
sys.path.append('/home/mckay/Documentos/math/time_series')
from models import gen_ARIMA


def lsm(X, strike=0, r=0, maxdegree=None):
    nrows, ncols = X.shape
    if maxdegree is None:
        maxdegree = int(np.sqrt(nrows))
    X = np.maximum(strike - X, 0)
    models = []
    for j in reversed(range(ncols-1)):
        # compute cash flow
        # construct vector of future cash values
        cashnext = np.zeros(nrows)
        for i in range(nrows):
            for k in range(j+1, ncols):
                if X[i,k] > 0:
                    cashnext[i] = X[i,k] * (1-r)**(k-j)
                    break
        # remove all entries where value in column j is zero since we should always continue in that case
        cashnext = cashnext[X[:,j] > 0]
        cashnow = X[:,j][X[:,j] > 0]
        # estimate least squares # make these orthogonal?
        degree = min(maxdegree, len(cashnow)-1)
        if degree < 1:
            # Deal with this more elegantly; the problem is that they all start at zero
            print(f'Stopped at index {j}')
            return X, models
            raise ValueError('Not enough data to compute regression')
        model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                          ('linear', LinearRegression())])
        model = model.fit(cashnow.reshape(-1,1), cashnext)
        models.append(model)
        predictions = model.predict(X[:,j].reshape(-1,1))
        predictions[X[:,j] <= 0] = 0
        continue_ = (predictions >= X[:,j])
        # set values for future periods to zero if we don't continue
        for k in range(j+1, ncols):
            X[:,k] *= continue_
        # set current values to zero if we continue
        X[:,j] *= (1 - continue_)
    return X, models


def boundary_viz(models, resol):
    pass


#X = np.array([gen_ARIMA(10, AR=[1], prerun=1) for _ in range(10)])