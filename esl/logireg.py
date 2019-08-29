#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:02:13 2019

@author: sinsakuokazaki
"""

#linear regressin using  Newton methon

import numpy as np
import scipy

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, min_step = 0.001, max_iter = 100,random_seed = 0):
        self.min_step = min_step
        self.random_seed = np.random.RandomState(random_seed)
        self.beta_ = None
        self.max_iter = max_iter
    
    def fit(self, X, y):
        # initialize weight (p+1, 1 )
        self.beta_ = np.random.randn(X.shape[1] + 1)
        # modify X to culculate adding 0 vector for intercect 
        one_vec = scipy.sparse.coo_matrix(np.ones(X.shape[0]).reshape(-1, 1))
        Xtc = scipy.sparse.hstack([one_vec, X])
        diff = np.inf
        ite = 0
        #print("started iteration")
        while diff > self.min_step and ite < self.max_iter :
            p = sigmoid(Xtc.dot(self.beta_))
            W = p * (1 - p)
            XW = Xtc.T.multiply(W)
            XWX = XW.dot(Xtc)
            z = Xtc.dot(self.beta_) + 1/W * (y - p)
            b = XW.dot(z)
            beta_old = self.beta_
            self.beta_ = XWX.dot(b)
            del z, b, XW, XWX
            diff = abs(beta_old - self.beta_).mean()
            ite += 1 
            #print("iteration", ite, "finished")
           
    def predict(self, X):
        Xtc =  np.c_[np.zero(X.shape[0], X)]
        p = sigmoid(np.dot(Xtc, self.beta_))
        return np.where(p > .5, 1, 0)


if __name__ == '__main__':
    
    from sklearn.model_selection import StratifiedKFold
    import datetime;
    from sklearn.metrics import roc_auc_score
    
    X = scipy.sparse.load_npz("sparse_news.npz")
    y = np.load("label_news.npz", allow_pickle = True)
    y = y["arr_0"]
    
    skf = StratifiedKFold(n_splits=2, random_state=0, shuffle=True)
    cv_score =[]
    for train_index, test_index in skf.split(X, y):
        
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = LogisticRegression()
        tsf = datetime.datetime.now().timestamp()
        model.fit(X_train, y_train)
        tse = datetime.datetime.now().timestamp()
        print("Learning time:", tse - tsf)
        score = roc_auc_score(y_test, model.predict(X_test))
        print('ROC AUC score:',score)
        cv_score.append(score)
    