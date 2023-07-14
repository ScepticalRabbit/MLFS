# -*- coding: utf-8 -*-
"""
===============================================================================
MLFS: Naive Bayes Algorithm
===============================================================================

Link to video tutorial:
https://www.youtube.com/watch?v=TLInuAorxqE&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=7
"""
import numpy as np

class NaiveBayes:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        n_samps, n_feats = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        # Calc mean, variance and prior for each classes
        self._mean = np.zeros((n_classes, n_feats))
        self._var = np.zeros((n_classes, n_feats))
        self._priors = np.zeros(n_classes)

        for ii, cc in enumerate(self._classes):
            X_c = X[y == cc]
            self._mean[ii, :] = X_c.mean(axis=0)
            self._var[ii, :] = X_c.var(axis=0)
            self._priors[ii] = X_c.shape[0] / float(n_samps)

    def predict(self, X):
        y_pred = np.array([self._predict(xx) for xx in X])
        return y_pred
    
    def _predict(self, x):
        posteriors = []
        
        # Calc posterior probs for each class
        for ii, cc in enumerate(self._classes):
            prior = np.log(self._priors[ii])
            posterior = np.sum(np.log(self._pdf(ii, x)))
            posterior = posterior + prior
            posteriors.append(posterior)
            
        # Return class with highest post prob
        # NOTE: argmax returns index of highest element
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean)**2) / (2* var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator/denominator