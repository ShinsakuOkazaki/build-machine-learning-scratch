#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:03:04 2019

@author: sinsakuokazaki
"""

import linearreg
import numpy as np

class PolynomialRegression:
    def __init__(self, digree):
        self.digree = digree
        
    def fit(self, x, y):
        x_pow = []
        xx = x.reshape(len(x), 1)
        for i in range(1, self.digree + 1):
            x_pow.append(xx**i)
        mat = np.concatenate(x_pow, axis = 1)
        linreg = linearreg.LinearRegression()
        linreg.fit(mat, y)
        self.w_ = linreg.w_
        
    def predict(self, x):
        r = 0
        for i in range(self.digree + 1):
            r += x**i * self.w_[i]
        return r