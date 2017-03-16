#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv
import time

# 分类
# SVC NuSVC LinearSVC


# SVC
"""
SVC参数解释
（1）C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default
C = 1.0；
（2）kernel：参数选择有RBF, Linear, Poly, Sigmoid, 默认的是
"RBF";
（3）degree：if you choose 'Poly' in param 2, this is effective, degree决定了多项式的最高次幂；
（4）gamma：核函数的系数('Poly', 'RBF' and 'Sigmoid'), 默认是gamma = 1 / n_features;
（5）coef0：核函数中的独立项，'RBF' and 'Poly'
有效；
（6）probablity: 可能性估计是否使用(true or false)；
（7）shrinking：是否进行启发式；
（8）tol（default = 1
e - 3）: svm结束标准的精度;
（9）cache_size: 制定训练所需要的内存（以MB为单位）；
（10）class_weight: 每个类所占据的权重，不同的类设置不同的惩罚参数C, 缺省的话自适应；
（11）verbose: 跟多线程有关，不大明白啥意思具体；
（12）max_iter: 最大迭代次数，default = 1， if max_iter = -1, no limited;
（13）decision_function_shape ： ‘ovo’ 一对一, ‘ovr’ 多对多 or None
无, default = None
（14）random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。
ps：7, 8, 9
一般不考虑。
"""

# from sklearn.svm import SVC
#
# X= np.array([[-1,-1],[-2,-1],[1,1],[2,1]])
# y = np.array([1,1,2,2])
#
# clf = SVC()
# clf.fit(X,y)
# print clf.fit(X,y)
# print clf.predict([[-0.8,-1]])

# NuSVM
"""
NuSVC参数
nu：训练误差的一个上界和支持向量的分数的下界。应在间隔（0，1 ]。
其余同SVC
"""

# from sklearn.svm import NuSVC
#
# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, 2, 2])
# clf = NuSVC()
# clf.fit(X, y)
# print clf.fit(X,y)
# print(clf.predict([[-0.8, -1]]))

"""
LinearSVC 参数解释
C：目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0；
loss ：指定损失函数
penalty ：
dual ：选择算法来解决对偶或原始优化问题。当n_samples > n_features 时dual=false。
tol ：（default = 1e - 3）: svm结束标准的精度;
multi_class：如果y输出类别包含多类，用来确定多类策略， ovr表示一对多，“crammer_singer”优化所有类别的一个共同的目标
如果选择“crammer_singer”，损失、惩罚和优化将会被被忽略。
fit_intercept ：
intercept_scaling ：
class_weight ：对于每一个类别i设置惩罚系数C = class_weight[i]*C,如果不给出，权重自动调整为 n_samples / (n_classes * np.bincount(y))
verbose：跟多线程有关，不大明白啥意思具体
"""

#一对一查询

from sklearn.svm import SVC
#
# # X = [[0], [1], [2], [3]]
# # Y = [0, 1, 2, 3]
# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# Y = np.array([1, 1, 2, 2])
#
# clf = SVC(decision_function_shape='ovo')  # ovo为一对一
# clf.fit(X, Y)
# print clf.fit(X, Y)
#
# dec = clf.decision_function(X)  # 返回的是样本距离超平面的距离
# print dec
# print dec.shape
#
# # 预测
# print clf.predict([[-0.8, -1]])


X = [[0], [1], [2], [3]]  # 一个？
Y = [0, 1, 2, 3]
clf = SVC(decision_function_shape='ovo')
clf.fit(X, Y)

dec = clf.decision_function([[1]])
print dec.shape[1] # 4 classes: 4*3/2 = 6

clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
print dec.shape[1] # 4 classes
