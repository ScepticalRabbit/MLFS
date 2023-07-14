# -*- coding: utf-8 -*-
"""
===============================================================================
MLFS: PCA Tester
===============================================================================

Link to video tutorial:
https://www.youtube.com/watch?v=Rjr62b_h7S4&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=8
"""
# Import external packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Import the machine learning from scratch package
import mlfs

# Helper functions
def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)
    
print("-----------------------------------------------------------")
print("PCA Tester")
print("-----------------------------------------------------------")
pp = mlfs.PlotProps()

# Load breast cancer data set from SciKitLearn
data = datasets.load_iris()
X, y = data.data, data.target
test_frac = 0.2 

# Create the PCA object
pca = mlfs.PCA(2)
pca.fit(X)
X_proj = pca.transform(X)

print("Shape of X:", X.shape)
print("Shape of X transformed:", X_proj.shape)

x1 = X_proj[:, 0]
x2 = X_proj[:, 1]

fig, ax = plt.subplots(figsize=pp.single_fig_size, layout='constrained')
fig.set_dpi(pp.resolution)
cmap = plt.cm.get_cmap("viridis",3)
plt.scatter(x1,x2, c=y, alpha=0.8, cmap=cmap)
plt.title("PCA",fontsize=pp.font_head_size,fontname=pp.font_name)
plt.xlabel("Princ. Comp. 1",fontsize=pp.font_ax_size,fontname=pp.font_name)
plt.ylabel("Princ. Comp. 2",fontsize=pp.font_ax_size,fontname=pp.font_name)
plt.colorbar()
plt.show()

print("")
print("Finished.")
print("-----------------------------------------------------------")
