#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:29:46 2019

@author: sinsakuokazaki
"""



# Newton optimization method for logistic regression

import numpy as np
from scipy import linalg

# Set minimum threshld
THRESHMIN = 1e-10

# Sigmoid function 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Logistic regression
class LogisticRegression:
    def __init__(self, tol=0.001, max_iter=3, random_seed=0):
        # Limit value of difference between weight for previous iteration and current iteration
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_seed)
        self.w_ = None
        
    def fit(self, X, y):
        # initialize weight with random number 
        # X is (n, p), then w is (p+1, 1)
        self.w_ = self.random_state.randn(X.shape[1] + 1)
        # set x(i, 0) to all one for intercept
        # Xtil is (n, p+1)
        Xtil = np.c_[np.ones(X.shape[0]), X]
        # initialize difference value to infinit
        diff = np.inf
        iter = 0
        # if difference value is biigger than limit value or iteration exceed, 
        # escape from loop
        while diff > self.tol and iter < self.max_iter:
            # yhat (expected value for y)
            # yhat is (n, 1)
            yhat = sigmoid(np.dot(Xtil, self.w_))
            # clip set THRESHMIN if yhat * (1 - yhat) is lower than that
            # r is (n, 1)
            r = np.clip(yhat * (1 - yhat),
                        THRESHMIN, np.inf)
            # calculating XR, which R' diagnal is r
            # R is (n, n), then XR is (p+1, n)
            XR = Xtil.T * r
            # XRX is (p+1, p+1)
            XRX = np.dot(Xtil.T * r, Xtil)
            w_prev = self.w_
            # b is (p+1, 1)
            b = np.dot(XR, np.dot(Xtil, self.w_) - 1 / r * (yhat - y))
            # solve Matrix multiplication to get new w (p+1, 1)
            self.w_ = linalg.solve(XRX, b)
            # culculate difference 
            diff = abs(w_prev - self.w_).mean()
            iter += 1
            
    def predict(self, X):
        Xtil = np.c_[np.ones(X.shape[0]), X]
        yhat = sigmoid(np.dot(Xtil, self.w_))
        return np.where(yhat > .5, 1, 0)
    
    