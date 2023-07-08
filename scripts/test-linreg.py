# -*- coding: utf-8 -*-
"""
===============================================================================
MLFS: Linear Regression Tester
===============================================================================

Link to video tutorial:
https://www.youtube.com/watch?v=ltXSoduiVwY&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=3
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import mlfs

# Helper functions:
def mse(y_test, preds):
    return np.mean((y_test-preds)**2)
    

print("-----------------------------------------------------------")
print("Linear Regression Tester")
print("-----------------------------------------------------------")

# Make a fake linear dataset with noise
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
test_frac = 0.25 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=1234)

# Plot the data
fig = plt.figure()
plt.scatter(X[:,0], y, color="b", marker="o", s=30)
plt.show()

# Create a linear regressor and fit the data
linr = mlfs.LinReg(alpha=0.001,n_iters=1000)
linr.fit(X_train,y_train)
preds = linr.predict(X_test)

acc_mse = mse(y_test,preds)
print()
print("MSE = {:.4f}".format(acc_mse))


print("")
print("Finished.")
print("-----------------------------------------------------------")