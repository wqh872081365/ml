#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv
import time

from sklearn.neural_network import MLPClassifier


# 实例
# Digit Recognizer

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


def nn_predict():

    clf = MLPClassifier(solver='adam', hidden_layer_sizes=50, alpha=1e-5,
                        max_iter=150, random_state=0, activation='logistic',
                        learning_rate_init=0.2)
    clf.fit(normalized_X,y)
    print clf
    # print clf.predict(test_n_X)

    # make predictions
    # expected = y[:100]
    # predicted = model.predict(normalized_X[:100, :])
    submit = np.array([])
    for i in range(280):
        start_time = time.time()
        predicted = clf.predict(test_n_X[i * 100:(i * 100 + 100), :])
        print i, predicted
        print time.time() - start_time
        # print predicted.shape
        submit = np.concatenate((submit, predicted.astype(int)), axis=0).astype(int)
        print submit.shape

    # summarize the fit of the model
    # print(metrics.classification_report(expected, predicted))
    # print(metrics.confusion_matrix(expected, predicted))

    # 保存数据csv

    with open('nn_submit' + '_sklearn.csv', 'w') as f:
        myWriter = csv.writer(f)
        myWriter.writerow(['ImageId', 'Label'])
        for k, p in enumerate(list(submit)):
            myWriter.writerow([str(k + 1), str(p)])


def main():
    nn_predict()


if __name__ == '__main__':
    main()



