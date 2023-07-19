# -*- coding: utf-8 -*-
"""
TODO
"""

from .knn import KNN
from .linreg import LinReg
from .logreg import LogReg
from .dectree import DecTree
from .randforest import RandForest
from .naivebayes import NaiveBayes
from .pca import PCA
from .perceptron import Perceptron

from .plotprops import PlotProps


__all__ = ["KNN" "LinReg" "LogReg" "DecTree" "RandForest" "NaiveBayes" "Perceptron" "PlotProps"]
