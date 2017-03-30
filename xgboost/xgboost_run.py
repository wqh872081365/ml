# -*- coding: utf-8 -*-

import xgboost as xgb
import numpy as np


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


params = {
    'booster':'gbtree',  # 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器; 选择：gbtree, gblinear, dart
    'objective': 'multi:softmax',  # 多类分类的模型
    'num_class':10,  # 类数，与 multisoftmax 并用
    'gamma':0.01,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
    'max_depth':6,  # 构建树的深度 [1:]
    #'lambda':450,   # L2 正则项权重
    'subsample':0.4,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
    'colsample_bytree':0.7, # 构建树树时的采样比率 (0:1]
    #'min_child_weight':12,  # 节点的最少特征数
    'silent':1 ,
    'eta': 0.005,  # 如同学习率
    'seed':710,
    'nthread':4,  # cpu 线程数,根据自己U的个数适当调整
    }

#Using 10000 rows for early stopping.
offset = 30000  # 训练集中数据42000，划分30000用作训练，12000用作验证

num_rounds = 500 # 迭代你次数
xgtest = xgb.DMatrix(test_n_X)

# 划分训练集与验证集
xgtrain = xgb.DMatrix(normalized_X[:offset,:], label=y[:offset])
xgval = xgb.DMatrix(normalized_X[offset:,:], label=y[offset:])

# return 训练和验证的错误率
watchlist = [(xgtrain, 'train'),(xgval, 'val')]


# training model
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(params, xgtrain, num_rounds, watchlist,early_stopping_rounds=100)
#model.save_model('./model/xgb.model') # 用于存储训练出的模型
preds = model.predict(xgtest,ntree_limit=model.best_iteration)

# 将预测结果写入文件，方式有很多，自己顺手能实现即可
np.savetxt('submission_xgb_MultiSoftmax_1.csv',np.c_[range(1,len(test)),preds],
                delimiter=',',header='ImageId,Label',comments='',fmt='%d')

# 模型保存


