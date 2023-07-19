# -*- coding: utf-8 -*-
"""
===============================================================================
MLFS: Support Vector Machine Algorithm
===============================================================================

Link to video tutorial:
https://www.youtube.com/watch?v=T9UcK-TxQGw&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=10
"""
import numpy as np

class SVM:
    def __init__(self, alpha=0.001, lambda_par = 0.01, n_iters=1000):
        self.alpha = alpha
        self.lambda_par = lambda_par
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samps, n_feats = X.shape
        # Make sure classes are +1/-1
        y_ = np.where(y <=0 , -1, 1)
        
        # Init Weights - better to randomly init
        self.w = np.zeros(n_feats)
        self.b = 0
        
        for ii in range(self.n_iters):
            for idx, xx in enumerate(X):
                cond = y_[idx] * (np.dot(xx, self.w) - self.b) >= 1
                if cond:
                    self.w -= self.alpha * (2 * self.lambda_par * self.w)
                else:
                    self.w -= self.alpha * (2 * self.lambda_par * self.w - np.dot(xx, y_[idx]))
                    self.b -= self.alpha * (y_[idx])                    
    
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        pred = np.sign(approx)
        return pred