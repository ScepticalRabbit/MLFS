# -*- coding: utf-8 -*-
"""
===============================================================================
MLFS: PCA Algorithm
===============================================================================
PCA = Principal Component Analysis

Link to video tutorial:
https://www.youtube.com/watch?v=Rjr62b_h7S4&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=8
"""
import numpy as np

class PCA:
    def __init__(self,n_comps):
        self.n_comps = n_comps
        self.comps = None
        self.mean = None
    
    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        
        # Calculate covariance, needs samples as columns
        cov = np.cov(X.T)
        
        # Calculate eigan vectors and values
        eig_vecs, eig_vals = np.linalg.eig(cov)
        # Eiganvectors v = [:,ii], transpose for easy calc
        eig_vecs = eig_vecs.T
        
        # Sort the eigan vectors
        idxs = np.argsort(eig_vals)[::-1] # Decreasing order
        eig_vals = eig_vals[idxs]
        eig_vecs = eig_vecs[idxs]
        
        self.comps = eig_vecs[:self.n_comps]
        
    def transform(self, X):
        # Mean centering
        X = X - self.mean
        # Project the data
        proj = np.dot(X, self.comps.T)
        return proj