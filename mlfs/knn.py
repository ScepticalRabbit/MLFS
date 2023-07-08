# -*- coding: utf-8 -*-
"""
===============================================================================
MLFS: KNN Algorithm
===============================================================================

Link to video tutorial
https://www.youtube.com/watch?v=rTEtEy5o3X0&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=2
"""

'''
Implementing the KNN algorithm
-------------------------------------------------------------------------------
1. Give a data point and calculate its distance to all other data points
2. Get the closest k points
3. Perform a task with the points:
    a) Regression: calculate the average of the k points
    b) Classification: get the majority vote

'''

# Imports
import numpy as np
from collections import Counter

# Global helper functions
def dist_euclid(x1,x2):
    dist = np.sqrt(np.sum(x1-x2)**2)
    return dist
    
# KNN class def
class KNN:
    def __init__(self,k=3):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        self.preds = [self._predict(x) for x in X]
        return self.preds
    
    def _predict(self, x):
        # Compute distance to all training data
        dists = [dist_euclid(x, x_t) for x_t in self.X_train]
        
        # Get the closest k points
        k_inds = np.argsort(dists)[:self.k]
        k_near_labs = [self.y_train[ii] for ii in k_inds]
        
        # Find majority vote
        most_common = Counter(k_near_labs).most_common()
        return most_common[0][0]

        
        

