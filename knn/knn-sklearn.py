#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv
import time

# 8/28/2016
# 分类 相邻算法
# submit 1
# 还可以使用其他算法 n_neighbors=10, algorithm="kd_tree"
# result : kaggle  447	new	 wangqihui 0.97286 1	Sun, 28 Aug 2016 07:22:05

# 数据加载

# load the CSV file as a numpy matrix
dataset = np.loadtxt("train.csv", dtype=str, delimiter=",")
test = np.loadtxt("test.csv", dtype=str, delimiter=",")
# separate the data from the target attributes
X = dataset[1:, 1:].astype(float)  # 数据
y = dataset[1:, 0].astype(float)  # 标签
test_X = test[1:, :].astype(float)

# print X.dtype

# print X, y
# print X.shape, y.shape

# 数据归一化

from sklearn import preprocessing
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
test_n_X = preprocessing.normalize(test_X)

# standardize the data attributes
# standardized_X = preprocessing.scale(X)

# print normalized_X
# print normalized_X.shape

# print standardized_X
# print standardized_X.shape

"""
# 特征选择  树算法
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(normalized_X, y)
# display the relative importance of each attribute
print(model.feature_importances_)
"""


# k-最相邻 用于复杂分类算法，回归问题；用估计值作为特征

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
# fit a k-nearest neighbor model to the data

def knn_predict(number):
    model = KNeighborsClassifier(n_neighbors=number, weights='distance',n_jobs=-1)
    model.fit(normalized_X, y)
    print(model)
    # make predictions
    # expected = y[:100]
    # predicted = model.predict(normalized_X[:100, :])
    submit = np.array([])
    for i in range(280):
        start_time = time.time()
        predicted = model.predict(test_n_X[i*100:(i*100+100), :])
        print i, predicted
        print time.time() - start_time
        # print predicted.shape
        submit = np.concatenate((submit, predicted.astype(int)), axis=0).astype(int)
        print submit.shape

    # summarize the fit of the model
    # print(metrics.classification_report(expected, predicted))
    # print(metrics.confusion_matrix(expected, predicted))

    # 保存数据csv

    with open('knn_submit'+str(number)+'_distance'+'_sklearn.csv', 'w') as f:
        myWriter = csv.writer(f)
        myWriter.writerow(['ImageId', 'Label'])
        for k, p in enumerate(list(submit)):
            myWriter.writerow([str(k+1), str(p)])


from sklearn.model_selection import train_test_split

# 结果是n=4,weights='distance'的效果最佳，为0.97586
def knn_train():
    # 第一步：将X和y分割成训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.4, random_state=4)
    # 这里的random_state参数根据给定的给定的整数，得到伪随机生成器的随机采样

    # 测试从K=1到K=15，记录测试准确率
    k_range = range(1, 20)
    test_accuracy = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)
        print(knn)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        test_accuracy.append(metrics.accuracy_score(y_test, y_pred))
        print test_accuracy

    print test_accuracy
    plt.plot(k_range, test_accuracy)
    plt.xlabel("Value of K for KNN")
    plt.ylabel("Testing Accuracy")
    plt.show()



def main():
    knn_predict(4)
    knn_train()


if __name__ == '__main__':
    main()

