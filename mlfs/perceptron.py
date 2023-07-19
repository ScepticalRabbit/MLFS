# -*- coding: utf-8 -*-
"""
===============================================================================
MLFS: Perceptron Algorithm
===============================================================================

Link to video tutorial:
https://www.youtube.com/watch?v=aOEoxyA4uXU&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=9
"""
import numpy as np

def unit_step_fun(x):
    return np.where(x > 0, 1, 0)

class Perceptron:
    def __init__(self, alpha=0.01,n_iters=1000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.act_fun = unit_step_fun
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samps, n_feats = X.shape
        
        # Init the weights and bias
        self.weights = np.zeros(n_feats)
        self.bias = 0
        
        # Make sure class labels y are between 0 and 1
        y_ = np.where(y>0, 1, 0)
        
        for ii in range(self.n_iters):
            for idx, xx in enumerate(X):
                lin_out = np.dot(xx, self.weights) + self.bias
                y_pred = self.act_fun(lin_out)
                
                # Perceptron update rules
                update = self.alpha * (y_[idx] - y_pred)
                self.weights += update * xx
                self.bias += update
                
    def predict(self, X):
        lin_out = np.dot(X, self.weights) + self.bias
        y_pred = self.act_fun(lin_out)
        return y_pred