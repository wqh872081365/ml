#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv
import time

from sklearn.svm import SVC


# 数据加载

# load the CSV file as a numpy matrix
dataset = np.loadtxt("train.csv", dtype=str, delimiter=",")
test = np.loadtxt("test.csv", dtype=str, delimiter=",")
# separate the data from the target attributes
X = dataset[1:, 1:].astype(float)  # 数据
y = dataset[1:, 0].astype(float)  # 标签
test_X = test[1:, :].astype(float)


# 数据归一化

from sklearn import preprocessing
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
test_n_X = preprocessing.normalize(test_X)


from sklearn.decomposition import PCA


pca=PCA(n_components=2)
pca.fit(X)
pca_X = pca.transform(X)

color_list = ["r", "b", "g", "c", "m", "y", "k", "tan", "pink", "violet"]

marker_list = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

for j in range(pca_X.shape[0]):
    i = int(y[j])
    plt.scatter(pca_X[j, 0], pca_X[j, 1], marker='o', color=color_list[i])
plt.show()



