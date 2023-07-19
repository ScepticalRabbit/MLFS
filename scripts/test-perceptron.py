# -*- coding: utf-8 -*-
"""
===============================================================================
MLFS: Perceptron Tester
===============================================================================

Link to video tutorial:
https://www.youtube.com/watch?v=aOEoxyA4uXU&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=9
"""
# Import external packages
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
X, y = datasets.make_blobs(
    n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
    )


p = mlfs.Perceptron(alpha=0.01, n_iters=1000)
p.fit(X_train, y_train)
y_pred = p.predict(X_test)

acc = accuracy(y_pred,y_test)
print("Accuracy = {}".format(acc))

#----------------------------------------------------------
# Create figure to show decision boundary
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])

x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])
ax.set_ylim([ymin - 3, ymax + 3])

plt.show()
#----------------------------------------------------------

print("")
print("Finished.")
print("-----------------------------------------------------------")
