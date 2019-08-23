#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:03:54 2019

@author: sinsakuokazaki
"""

import numpy as np
from scipy import linalg

class RidgeRegression:
    def __init__(self, lambda_ = 1.):
        self.lambda_ = lambda_
        self.w_ = None
    
    def fit(self, X, t):
        Xtil = np.c_[np.ones(X.shape[0]), X]
        I = np.eye(Xtil.shape[1])
        A = np.dot(Xtil.T, Xtil) + self.lambda_ * I
        b = np.dot(Xtil.T, t)
        self.w_ = linalg.solve(A, b)
        
    def predict(self, X):
        Xtil = np.c_[np.ones(X.shape[0]), X]
        return np.dot(Xtil, self.w_)
    
    