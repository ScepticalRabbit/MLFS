# -*- coding: utf-8 -*-
"""
===============================================================================
MLFS: Linear Regression
===============================================================================

Link to video tutorial:
https://www.youtube.com/watch?v=ltXSoduiVwY&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=3
"""

'''
Implementing the Linear Regression Algorithm
-------------------------------------------------------------------------------
Terminology: 
    weight = slope, bias = y-intercept
    MSE = mean square error
    MSE = 1/N*sum(y - (w*x + b))**2
    alpha = learning rate

Fit
1. Initialise weights and bias as zero
2. Given a set of data
    a) calculate error
    b) use gradient descent to work out wieght/bias
    c) repeat n times
    
Predict
1. Give a set of data points
2. Calculate prediction based on weights and bias
'''
import numpy as np

class LinReg:
    def __init__(self, alpha = 0.001, n_iters=1000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
            
    def fit(self, X, y):
        n_samps, n_feats = X.shape
        self.weights = np.zeros(n_feats)
        self.bias = 0
        
        for nn in range(self.n_iters):
            # Calculate predictions based on current w,b
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Calculate derivatives of the cost wrt w,b
            dw = (1/n_samps) * np.dot(X.T,(y_pred-y))
            db = (1/n_samps) * np.sum(y_pred-y)
            
            # Update weights and biases
            self.weights = self.weights - self.alpha * dw
            self.bias = self.bias - self.alpha * db
         
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred