# -*- coding: utf-8 -*-
"""
===============================================================================
MLFS: Logistic Regression
===============================================================================

Link to video tutorial:
https://www.youtube.com/watch?v=YYEJ_GUguHw&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=4
"""

'''
Implementing the Logistic Regression Algorithm
-------------------------------------------------------------------------------
Cross entropy cost function for logistic regression.

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

class LogReg:
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
            y_pred = self._sigmoid(X, self.weights, self.bias)
            
            # Calculate derivatives of the cost wrt w,b
            dw = (1/n_samps) * np.dot(X.T,(y_pred-y))
            db = (1/n_samps) * np.sum(y_pred-y)
            
            # Update weights and biases
            self.weights = self.weights - self.alpha * dw
            self.bias = self.bias - self.alpha * db
         
    def predict_prob(self, X):
        y_pred = self._sigmoid(X, self.weights, self.bias)
        return y_pred
    
    def predict_class(self, X):
        y_pred = self.predict_prob(X)
        class_pred = [0 if y <=0.5 else 1 for y in y_pred]
        return class_pred
    
    def cross_entropy(self, X, y, y_pred):
        n_samps, n_feats = X.shape
        #I = np.ones()
        #cross_e = (1/n_samps) * np.sum(np.log(y_pred) + (np.ones() - ))
    
    def _sigmoid(self, X, w, b):
        y_pred = 1 / (1+np.exp(-(np.dot(X,w) + b)))
        return y_pred