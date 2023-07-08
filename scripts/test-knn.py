# -*- coding: utf-8 -*-
"""
===============================================================================
MLFS: KNN Algorithm Tester
===============================================================================

Link to video tutorial:
https://www.youtube.com/watch?v=rTEtEy5o3X0&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=2
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import mlfs

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

print("-----------------------------------------------------------")
print("KNN Tester")
print("-----------------------------------------------------------")

iris = datasets.load_iris()
X,y = iris.data, iris.target

test_frac = 0.25 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=1234)
                                                    
# Plot the data
plt.figure()
plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()

# Create a classifier KNN class
clf = mlfs.KNN(k=3)
clf.fit(X_train,y_train)
preds = clf.predict(X_test)
print("Predictions on test data:")
print(preds)

accuracy = np.sum(preds == y_test) / len(y_test)
print()
print("Accuracy = {:.4f}".format(accuracy))

print()
print("Finished.")
print("-----------------------------------------------------------")

