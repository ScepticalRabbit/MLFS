# -*- coding: utf-8 -*-
"""
===============================================================================
MLFS: Temp Tester
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
print("Temp Tester")
print("-----------------------------------------------------------")
pp = mlfs.PlotProps()

# Load breast cancer data set from SciKitLearn
X, y = datasets.make_classification(
    n_samples=100, n_features=10, n_classes=2, random_state = 123
    )
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
    )

clf = mlfs.NaiveBayes()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy(y_pred,y_test)
print("Accuracy = {}".format(acc))

print("")
print("Finished.")
print("-----------------------------------------------------------")
