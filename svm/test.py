#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca=PCA(n_components=2)
pca.fit(X)
pca_X = pca.transform(X)

plt.scatter(pca_X[:, 0], pca_X[:, 1],marker='o', color="g")
plt.scatter(pca_X[:, 0], pca_X[:, 1],marker='v', color="r")
plt.show()
