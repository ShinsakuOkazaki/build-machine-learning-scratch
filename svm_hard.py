#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:39:59 2019

@author: sinsakuokazaki
"""

import numpy as np
from operator import itemgetter




class SVC:
    
#     def __init__(self):
#         self.a_ = None
#         self.w_ = None
#         self.w0_ = None
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
        
        count = 0
        while True:
            count += 1
            print("Iteration:", count)
            #get y multiple derivative of function
            
            # ydf is (p, 1)
            ydf = y * (1 - np.dot(yx, ayx.T))
            # iydf is (p, 2)
            iydf = np.c_[indices, ydf]
            
            #get index of mininum value of derivative of function
            #when y = -1 or a >0
            i = int(min(iydf[(y < 0) | (a > 0)], 
                             key=itemgetter(1))[0])
            print("i:", i)
            #get index of maximum value of derivative of function
            #when y = 1 or a > 0
            j = int(max(iydf[(y > 0) | (a > 0)], 
                             key=itemgetter(1))[0])
            print("j:", j)  
            #check if we need to update
            if ydf[i] >= ydf[j]:
                break
            
            # ay2 is (1, 1).
            # ay2 is the sumation of multiple of elements of a and y except where are i and j
            ay2 = ay - y[i] * a[i] - y[j] * a[j]
            # ayx2  is (p, 1)
            # ayx2 is the sumation of multiple of elements of a, y, and x except where are i and j
            ayx2 = ayx - y[i] * a[i] * X[i, :] - y[j] * a[j] * X[j, :]
            
            
            # get new ai
            ai = ((1 - y[i] * y[j] + y[i] * np.dot(X[i, :] - X[j, :], X[j,:] * ay2 - ayx2)) \
                                                        / ((X[i] - X[j])**2).sum())
            
            # if ai is smaller than 0, the optimal is ai = 0
            if ai < 0:
                ai = 0
            
            # get new aj using result of ai 
            aj = (-ai * y[i] - ay2) * y[j]
            
            # if aj is smalller than 0, the optimal is aj = 0
            # ai is needed to be culculated according to the aj's result in this time
            if aj < 0:
                aj = 0
                ai = (-aj*y[j] - ay2) * y[i]
            
            # update ay and ayx with only values changed in this iteration
            ay += y[i]*(ai - a[i]) + y[j]*(aj - a[j])
            ayx += y[i]*(ai - a[i])*X[i, :] + y[j]*(aj - a[j])*X[j,:]
            
            #if new ai is equal to last ai,do not assign new ai
            # same for aj, because if ai does not change aj neither.
            if ai == a[i]:
                break
            
            a[i] = ai
            a[j] = aj
        
        # store Lagrange multiplier 
        self.a_ = a
        
        # culculate w and w0 from a, y and x only where a's elements are not 0
        ind = a != 0.
        print("ind", ind)
        # store w (ind.sum(), p)
        self.w_ = ((a[ind] * y[ind]).reshape(-1, 1) \
                   * X[ind, :]).sum(axis=0)
        print("X[ind,:]", X[ind, :].shape)
        print("self.w_", self.w_.shape)
        # store w0 (1, ind.sum())
        self.w0_ = (y[ind] - np.dot(X[ind, :], self.w_)).sum() / ind.sum()
        print("self.w0_", self.w0_)
        
    def predict(self, X):
        return np.sign(self.w0_ + np.dot(X, self.w_))

    
        
        