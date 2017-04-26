# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl

import numpy as np
from sklearn.model_selection import train_test_split


# Digit Recognizer

sess = tf.InteractiveSession()

# 准备数据
# X_train, y_train, X_val, y_val, X_test, y_test = \
#                                 tl.files.load_mnist_dataset(shape=(-1,784))

# load the CSV file as a numpy matrix
dataset = np.loadtxt("train.csv", dtype=str, delimiter=",")
test = np.loadtxt("test.csv", dtype=str, delimiter=",")
# separate the data from the target attributes
X_dataset = dataset[1:, 1:].astype(float)  # 数据
y_dataset = dataset[1:, 0].astype(float)  # 标签
test_X = test[1:, :].astype(float)

# print X.dtype

# print X, y
# print X.shape, y.shape

# 数据归一化

from sklearn import preprocessing
# normalize the data attributes
normalized_X = preprocessing.normalize(X_dataset)
# X_test = preprocessing.normalize(test_X)

X_train, X_val, y_train, y_val = train_test_split(normalized_X[:32000, :], y_dataset[:32000, :], test_size=0.4, random_state=4)
X_test = normalized_X[32000:, :]
y_test = y_dataset[32000:, :]

# 定义 placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

# 定义模型
network = tl.layers.InputLayer(x, name='input_layer')
network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
network = tl.layers.DenseLayer(network, n_units=800,
                                act = tf.nn.relu, name='relu1')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
network = tl.layers.DenseLayer(network, n_units=800,
                                act = tf.nn.relu, name='relu2')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
network = tl.layers.DenseLayer(network, n_units=10,
                                act = tf.identity,
                                name='output_layer')
# 定义损失函数和衡量指标
# tl.cost.cross_entropy 在内部使用 tf.nn.sparse_softmax_cross_entropy_with_logits() 实现 softmax
y = network.outputs
cost = tl.cost.cross_entropy(y, y_, 'cost')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y), 1)

# 定义 optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                            epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

# 初始化 session 中的所有参数
tl.layers.initialize_global_variables(sess)

# 列出模型信息
network.print_params()
network.print_layers()

# 训练模型
tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
            acc=acc, batch_size=500, n_epoch=4000, print_freq=5,
            X_val=X_val, y_val=y_val, eval_train=False)

# 评估模型
tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)

# 预测模型
# preds = tl.utils.predict(sess, network, X_test, x, y_op)

# 把模型保存成 .npz 文件
# tl.files.save_npz(network.all_params , name='model.npz')

# 保持数据
# np.savetxt('submission_tf_nn.csv',np.c_[range(1,len(test)),preds],
#                 delimiter=',',header='ImageId,Label',comments='',fmt='%d')



sess.close()



