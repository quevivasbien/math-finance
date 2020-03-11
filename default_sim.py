#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:35:17 2020

@author: mckay
"""

import numpy as np
from scipy.stats import norm, poisson



def wiener_process(t, n):
    # converges to a wiener process as n -> oo
    return sum(norm.rvs(size=int(n*t))) / np.sqrt(n)


def sim_S(S0, r, beta, sigma, T, dt=0.001):
    '''simulates the process shown in equation 1.1.1'''
    # t = np.arange(0, T, dt)
    n = int(T // dt) + 1 # len(t)
    S = np.zeros(n)
    S[0] = S0
    stdev = np.sqrt(dt)
    for i in range(1, n):
        S[i] = S[i-1] + S[i-1] * ((r-beta)*dt + sigma*norm.rvs(scale=stdev))
    return S


def sim_S_jumps(S0, r, beta, sigma, eta, lambda_, T, dt=0.001):
    '''simulates same process as above, but with poisson jumps'''
    n = int(T // dt) + 1
    S = np.zeros(n)
    S[0] = S0
    stdev = np.sqrt(dt)
    for i in range(1, n):
        S[i] = S[i-1] + S[i-1] * \
            ((r-beta)*dt + sigma*norm.rvs(scale=stdev) + eta*poisson.rvs(shape=lambda_*dt))
    return S


def find_gammas(r, beta, sigma):
    '''finds roots of the characteristic equation 1.3.4'''
    mu = r - beta
    a = sigma**2 / 2
    b = mu - sigma**2 / 2
    c = -r
    # just plug into quadratic equation
    gamma1 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    gamma2 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    return gamma1, gamma2


def find_default_boundary(r, beta, sigma, C):
    '''computes constant default boundary from equation 1.4.10'''
    gamma1, _ = find_gammas(r, beta, sigma)
    return (gamma1/(gamma1-1)) * (C/r)


def calculate_present_value(C, r, T, dt=0.001):
    '''estimates present value of security with constant coupon rate C,
    with interest rate r, that defaults at time T
    
    This doesn't have the meaning that I thought it did! Don't use this.
    '''
    t = np.arange(0, T, dt)
    return sum(C*np.exp(-r*t)*dt)


def get_default_times(S0, r, beta, sigma, C, T, dt=0.001, N=1000):
    '''gets sample of times to default using process from sim_S
    and constant default boundary from find_default_boundary'''
    SB = find_default_boundary(r, beta, sigma, C)
    print(SB)
    samples = (sim_S(S0, r, beta, sigma, T, dt) for _ in range(N))
    default_times = np.array([np.argmax(sample <= SB) for sample in samples])*dt
    return default_times


def estimate_equity_value(samples, default_times, r, beta, C, dt):
    '''estimates equity value as in equation 1.4.5
    assumes tax is 0, but that is irrelevant for optimization
    '''
    N = len(samples)
    total = 0
    total = sum(np.exp(-r*t*dt) * (beta*S[i] - C) * dt \
                    for S, T in zip(samples, default_times) \
                    for i, t in zip(np.arange(0, int(T / dt)+1), np.arange(0, T, dt)))
    return total / N


def recursive_maximizer(func, a, b, midpoints=10, itercount=3):
    # a simple way to try to find the max of a func on the interval [a,b]
    intvl = np.linspace(a, b, midpoints+2)
    f_intvl = func(intvl) # func must be vectorized for this to work
    i_max = np.argmax(f_intvl)
    if itercount == 1:
        return intvl[i_max], f_intvl[i_max]
    if i_max == midpoints+1:
        j_max = i_max-1
    elif i_max == 0:
        j_max = i_max+1
    else:
        j_max = i_max-1 if f_intvl[i_max-1] > f_intvl[i_max+1] else i_max+1
    if i_max > j_max:
        i_max, j_max = j_max, i_max
    return recursive_maximizer(func, intvl[i_max], intvl[j_max],
                               midpoints=midpoints, itercount=itercount-1)
        


def find_optimal_boundary(S0, r, beta, sigma, C, T, dt=0.05, N=1000, a=0, b=100):
    '''attempts to find optimal constant default boundary via crude monte carlo approach'''
    # simulate N random processes
    samples = np.array([sim_S(S0, r, beta, sigma, T, dt) for _ in range(N)]) # is this the optimal way to construct that?
    # define objective function
    def obj(x):
        # return sum instead of average since maxizimizing it is equivalent
        default_times = np.argmax(samples <= x, 1)*dt
        # set to endpoint if never defaults in interval
        # TODO: figure out how to deal with this better
        print(f'{sum(default_times==0)} did not default in {T} with {x}')
        default_times[default_times == 0] = T
        return estimate_equity_value(samples, default_times, r, beta, C, dt)
    # i use my own maximizer function here
    # scipy's optimizer doesn't work great because obj is not continuous
    out = recursive_maximizer(np.vectorize(obj), a, b)[0]
    return out

# Questions:
    # How to choose S0? Should I randomize that as well?
    # How to make extrapolations when time horizon is finite
    # Increasing the time horizon tends to reduce estimates
    # Increasing N doesn't seem to help estimates much
    # I think this is mostly just problems with things not crossing real default boundary in time frame