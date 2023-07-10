# -*- coding: utf-8 -*-
"""
===============================================================================
MLFS: Logistic Regression Tester
===============================================================================

Link to video tutorial:
https://www.youtube.com/watch?v=YYEJ_GUguHw&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=4
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import mlfs

# Helper functions
def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)
    
print("-----------------------------------------------------------")
print("Logistic Regression Tester")
print("-----------------------------------------------------------")
pp = mlfs.PlotProps()

# Load breast cancer data set from SciKitLearn
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
test_frac = 0.2 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=1234)

logr_clf = mlfs.LogReg(alpha=0.01,n_iters=1000)
logr_clf.fit(X_train,y_train)
pred_prob = logr_clf.predict_prob(X_test)
pred_class = logr_clf.predict_class(X_test)
y_pred = pred_class

print("Predicted probabilities:")
print(pred_prob)
print()
print("Predicted classes:")
print(pred_class)
print()

acc = accuracy(y_pred,y_test)
print("Accuracy = {}".format(acc))

print("")
print("Finished.")
print("-----------------------------------------------------------")