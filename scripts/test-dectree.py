# -*- coding: utf-8 -*-
"""
===============================================================================
MLFS: Decision Tree Tester
===============================================================================

Link to video tutorial:

"""
# Import external packages
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Import the machine learning from scratch package
import mlfs

# Helper functions
def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)
    
print("-----------------------------------------------------------")
print("Decision Tree Tester")
print("-----------------------------------------------------------")
pp = mlfs.PlotProps()

# Load breast cancer data set from SciKitLearn
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
test_frac = 0.2 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=1234)

dt_clf = mlfs.DecTree()
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)

acc = accuracy(y_pred,y_test)
print("Accuracy = {}".format(acc))

print("")
print("Finished.")
print("-----------------------------------------------------------")
