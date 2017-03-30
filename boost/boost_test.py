#!/usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np
import csv
import time


# load the CSV file as a numpy matrix
dataset = np.loadtxt("train.csv", dtype=str, delimiter=",")
test = np.loadtxt("test.csv", dtype=str, delimiter=",")
# separate the data from the target attributes
X = dataset[1:, 1:].astype(float)  # 数据
y = dataset[1:, 0].astype(float)  # 标签
test_X = test[1:, :].astype(float)


# adaboost

from sklearn.ensemble import AdaBoostClassifier #For Classification
from sklearn.svm import SVC
svc = SVC(kernel='linear', decision_function_shape='ovo')
clf = AdaBoostClassifier(n_estimators=10,  # 弱学习者的数量
                         base_estimator=svc,  # 指定ml算法
                         learning_rate=1,  # 学习率
                         algorithm='SAMME'  # 默认SAMME.R 效率高
                         )
#Above I have used decision tree as a base estimator, you can use any ML learner as base estimator if it ac# cepts sample weight
clf.fit(X, y)
print clf


# Gradient Boosting 渐变增强




# make predictions
# expected = y[:100]
# predicted = model.predict(normalized_X[:100, :])
submit = np.array([])
for i in range(280):
    start_time = time.time()
    predicted = clf.predict(test_X[i * 100:(i * 100 + 100), :])
    print i, predicted
    print time.time() - start_time
    # print predicted.shape
    submit = np.concatenate((submit, predicted.astype(int)), axis=0).astype(int)
    print submit.shape

# summarize the fit of the model
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))

# 保存数据csv

with open('adaboost_svc_submit' + '_sklearn.csv', 'w') as f:
    myWriter = csv.writer(f)
    myWriter.writerow(['ImageId', 'Label'])
    for k, p in enumerate(list(submit)):
        myWriter.writerow([str(k + 1), str(p)])

