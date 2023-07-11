# -*- coding: utf-8 -*-
"""
===============================================================================
MLFS: Decision Tree Algorithm
===============================================================================

Link to video tutorial
https://www.youtube.com/watch?v=NxEHSAfFlK8&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=5
"""

'''
Implementing the algorithm
-------------------------------------------------------------------------------
1. 
'''

import numpy as np
from collections import Counter

class Node:
    def __init__(self,feat=None,thres=None,left=None,right=None,*,value=None):
        self.feat = feat
        self.thres = thres
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None

class DecTree:
    def __init__(self,min_samp_split=2,max_depth=100,n_feats=None): 
        self.min_samp_split = min_samp_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root=None
    
    def fit(self, X, y):
        if not self.n_feats:
            self.n_feats = X.shape[1] 
        else: 
            self.n_feats = min(X.shape[1],self.n_feats)
            
        self.root = self._grow_tree(X, y)
        
    # Recursive function
    def _grow_tree(self, X, y, depth=0):
        n_samps, n_feats = X.shape
        n_labs = len(np.unique(y))
        
        # Check stop criteria
        if (depth>=self.max_depth or n_labs==1 or n_samps<self.min_samp_split):
            leaf_value = self._most_lab(y)
            return Node(value=leaf_value)
        
        # Find the best split
        feat_idxs = np.random.choice(n_feats, self.n_feats,replace=False)
        best_feat, best_thres = self._best_split(X,y,feat_idxs)
        
        # Create child nodes
        left_idxs, right_idxs = self._split(X[:,best_feat],best_thres)
        left = self._grow_tree(X[left_idxs,:], y[left_idxs], depth+1)
        right= self._grow_tree(X[right_idxs,:], y[right_idxs], depth+1)
        return Node(best_feat,best_thres,left,right)
        
    def _most_lab(self,y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thres = None, None
        
        for ff in feat_idxs:
            X_col = X[:,ff]
            thres_all = np.unique(X_col)
            
            for tt in thres_all:
                # Calculate the information gain
                info_gain = self._info_gain(y,X_col,tt)
                
                if info_gain > best_gain:
                    best_gain = info_gain
                    split_idx = ff
                    split_thres = tt
                    
        return split_idx, split_thres
                
                
    def _info_gain(self, y, X_col, thres):
        # Calc parent entropy
        parent_entropy = self._entropy(y)
        
        # Create children
        left_idxs, right_idxs = self._split(X_col,thres)
        
        if len(left_idxs)==0 or len(right_idxs)==0:
            return 0
        
        # Calc weighted average of child entropy
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        # Weighted average of the left and right branch
        child_entropy = n_l/n*e_l + (n_r/n)*e_r
          
        # Calc info gain
        info_gain = parent_entropy - child_entropy
        return info_gain
    
    def _split(self,X_col,thres):
        left_idxs = np.argwhere(X_col<=thres).flatten()
        right_idxs = np.argwhere(X_col>thres).flatten()
        return left_idxs, right_idxs
    
    def _entropy(self, y):
        # Like a histogram
        hist = np.bincount(y)
        ps = hist/len(y)
        entropy = -np.sum([pp*np.log(pp) for pp in ps if pp>0])
        return entropy
          
    def predict(self, X):
        return np.array([self._traverse_tree(xx,self.root) for xx in X])
    
    # Recursive traverse
    def _traverse_tree(self,x,node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feat] <= node.thres:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
            
        
        
    
