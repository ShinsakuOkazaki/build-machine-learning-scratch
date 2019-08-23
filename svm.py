#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:39:59 2019

@author: sinsakuokazaki
"""

import numpy as np
from operator import itemgetter




class SVM:
    # X is (n, p), y is (p, 1)
    def fit(self, X, y, selections=None):
        # a is (n, 1)
        a = np.zeros(X.shape[0])
        ay = 0
        # ayx is (p, 1)
        ayx = np.zeros(X.shape[1])
        # yx is (n, p)
        yx = y.reshape(-1, 1) * X
        # indices is (n, 1)
        indices = np.arange(X.shape[0])
        
        while True:
            # ydf is (p, 1)
            ydf = y * (1 - np.dot(yx, ayx.T))
            # iydf is (p, 2)
            iydf = np.c_[indices, ydf]
            
            #get index of mininum value of derivative of function
            #when y = -1 or a >0
            i = int(min(iydf[(y < 0) | (a > 0)], 
                             key=itemgetter(1))[0])
            
            #get index of maximum value of derivative of function
            #when y = 1 or a > 0
            j = int(max(iydf[(y > 0) | (a > 0)], 
                             key=itemgetter(1))[0])
            
            #check otherwise update data
            if ydf[i] >= ydf[j]:
                break
            
            #ay2 is (1, 1)
            ay2 = ay - y[i] * a[i] - y[j] * a[j]
            #ayx2  is (p, 1)
            ayx2 = ayx - y[i] * a[i] * X[i, :] - y[j] * a[j] * X[j, :]
            
            ai = ((1 - y[i] * y[j] + y[i] * np.dot(X[i, :] - X[j, :], X[j,:] * ay2 - ayx2)) \
                                                        / ((X[i, :] - X[j, :])**2).sum())
            
            
            if ai < 0:
                ai = 0
                
            aj = (-ai * y[i] - ay2) * y[j]
            if aj < 0:
                aj = 0
                ai = (-aj*y[j] - ay2) * y[i]
                
            ay += y[i]*(ai - a[i]) + y[j]*(aj - a[j])
            
            ayx += y[i]*(ai - a[i])*X[i, :] + y[j](aj - a[j])*X[j,:]
            
            if ai == a[i]:
                break
            a[i] = ai
            a[j] = aj
            
        self.a_ = a
        ind = a != 0.
        self.w_ = ((a[ind] * y[ind]).reshape(-1, 1) \
                   * X[ind, :].sum(axis=0))
        self.w0_ = (y[ind] - np.dot(X[ind, :], self.w_)).sum() / ind.sum()
        
    def predict(self, X):
        return np.sign(self.w0_ + np.dot(X, self.w_))

    
        
        