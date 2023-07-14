# -*- coding: utf-8 -*-
"""
===============================================================================
MLFS: Random Forest Algorithm
===============================================================================

Link to video tutorial
https://www.youtube.com/watch?v=kFwe2ZZU7yw&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=6
"""

import numpy as np
from collections import Counter
from mlfs import DecTree

class RandForest:
    def __init__(self, n_trees=10, max_depth=10, min_samp_split=2, n_feats=None): 
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samp_split = min_samp_split
        self.n_feats = n_feats
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        for tt in range(self.n_trees):
           tree = DecTree(max_depth=self.max_depth,
                          min_samp_split=self.min_samp_split,
                          n_feats=self.n_feats)
           X_samp, y_samp = self._bootstrap_samps(X, y)
           tree.fit(X_samp, y_samp)
           self.trees.append(tree)
           
    def _bootstrap_samps(self, X, y):
        n_samps = X.shape[0]
        idxs = np.random.choice(n_samps,n_samps,replace=True)
        return X[idxs], y[idxs]
        
    def predict(self, X):
        preds = np.array([tt.predict(X) for tt in self.trees])
        # pred is a list of lists - need predictions for same class in same list
        tree_preds = np.swapaxes(preds, 0, 1)
        final_pred = np.array([self._most_lab(pp) for pp in tree_preds])
        return final_pred
        
    def _most_lab(self,y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    
               
        
    
