# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np


def digit_recognizer():
    # TEST learn
    sess = tf.InteractiveSession()

    # 准备数据
    X_train, y_train, X_val, y_val, X_test, y_test = \
                                    tl.files.load_mnist_dataset(shape=(-1,784))

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
                acc=acc, batch_size=500, n_epoch=500, print_freq=5,
                X_val=X_val, y_val=y_val, eval_train=False)

    # 评估模型
    tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)

    # 把模型保存成 .npz 文件
    tl.files.save_npz(network.all_params , name='model.npz')
    sess.close()



# Digit Recognizer

def digit_recognizer_two(number=0):
    sess = tf.InteractiveSession()

    # 准备数据
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))

    # load the CSV file as a numpy matrix
    # dataset = np.loadtxt("train.csv", dtype=str, delimiter=",")
    # test = np.loadtxt("test.csv", dtype=str, delimiter=",")
    # # separate the data from the target attributes
    # X_dataset = dataset[1:, 1:].astype(float)  # 数据
    # y_dataset = dataset[1:, 0].astype(int)  # 标签
    # test_X = test[1:, :].astype(float)

    # print X.dtype

    # print X, y
    # print X.shape, y.shape

    # 数据归一化

    # from sklearn import preprocessing
    # # normalize the data attributes
    # normalized_X = preprocessing.normalize(X_dataset)
    # X_test = preprocessing.normalize(test_X)
    #
    # from sklearn.model_selection import train_test_split
    #
    # X_train, X_val, y_train, y_val = train_test_split(normalized_X, y_dataset, test_size=0.4,
    #                                                   random_state=4)

    X_train[X_train <= 0.25] = 0
    X_train[X_train > 0.25] = 1
    X_train = X_train.astype(int)
    X_val[X_val <= 0.25] = 0
    X_val[X_val > 0.25] = 1
    X_val = X_val.astype(int)
    X_test[X_test <= 0.25] = 0
    X_test[X_test > 0.25] = 1
    X_test = X_test.astype(int)

    # 数据分为0和非0,值为0和1；
    if number > 1:
        y_train[y_train != number] = 0
        y_train[y_train == number] = 1
        y_val[y_val != number] = 0
        y_val[y_val == number] = 1
    elif number == 1:
        y_train[y_train != number] = 0
        y_val[y_val != number] = 0
        y_test[y_test != number] = 0
    else:
        y_train += 1
        y_val += 1

        y_train[y_train != number+1] = 0
        y_val[y_val != number+1] = 0

    # 定义 placeholder
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

    # 定义模型
    network = tl.layers.InputLayer(x, name='input_layer_'+str(number))
    network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1_'+str(number))
    network = tl.layers.DenseLayer(network, n_units=800,
                                   act=tf.nn.relu, name='relu1_'+str(number))
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2_'+str(number))
    network = tl.layers.DenseLayer(network, n_units=800,
                                   act=tf.nn.relu, name='relu2_'+str(number))
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3_'+str(number))
    network = tl.layers.DenseLayer(network, n_units=2,
                                   act=tf.identity,
                                   name='output_layer_'+str(number))
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
                 acc=acc, batch_size=500, n_epoch=500, print_freq=5,
                 X_val=X_val, y_val=y_val, eval_train=False)

    # 评估模型
    tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)

    # 把模型保存成 .npz 文件
    # tl.files.save_npz(network.all_params, name='model_two.npz')

    # preds = tl.utils.predict(sess, network, X_test, x, y_op)
    sess.close()
    # tl.layers.clear_layers_name()
    # return preds
    # 保持数据
    # np.savetxt('submission_tf_nn.csv',np.c_[range(1,len(X_test)+1),preds],
    #                 delimiter=',',header='ImageId,Label',comments='',fmt='%d')


def main():
    # digit_recognizer()

    # preds_0 = digit_recognizer_two(0)
    digit_recognizer_two(1)
    # preds_2 = digit_recognizer_two(2)
    # preds_3 = digit_recognizer_two(3)
    # preds_4 = digit_recognizer_two(4)
    # preds_5 = digit_recognizer_two(5)
    # preds_6 = digit_recognizer_two(6)
    # preds_7 = digit_recognizer_two(7)
    # preds_8 = digit_recognizer_two(8)
    # preds_9 = digit_recognizer_two(9)

    # preds = np.empty([28000], dtype=int)
    # preds.fill(-1)
    # for i in range(preds.shape[0]):
    #     if preds_0[i] == 1:
    #         preds[i] = 0
    #     if preds_1[i] == 1:
    #         preds[i] = 1
    #     if preds_2[i] == 1:
    #         preds[i] = 2
    #     if preds_3[i] == 1:
    #         preds[i] = 3
    #     if preds_4[i] == 1:
    #         preds[i] = 4
    #     if preds_5[i] == 1:
    #         preds[i] = 5
    #     if preds_6[i] == 1:
    #         preds[i] = 6
    #     if preds_7[i] == 1:
    #         preds[i] = 7
    #     if preds_8[i] == 1:
    #         preds[i] = 8
    #     if preds_9[i] == 1:
    #         preds[i] = 9
    #
    # print "-1 number: "
    # print (preds==-1).sum()
    #
    # print "0 start number: "
    # print (preds_0 == 1).sum()
    # print "0 number: "
    # print (preds == 0).sum()
    #
    # print "1 start number: "
    # print (preds_1 == 1).sum()
    # print "1 number: "
    # print (preds == 1).sum()
    #
    # print "2 start number: "
    # print (preds_2 == 1).sum()
    # print "2 number: "
    # print (preds == 2).sum()
    #
    # print "3 start number: "
    # print (preds_3 == 1).sum()
    # print "3 number: "
    # print (preds == 3).sum()
    #
    # print "4 start number: "
    # print (preds_4 == 1).sum()
    # print "4 number: "
    # print (preds == 4).sum()
    #
    # print "5 start number: "
    # print (preds_5 == 1).sum()
    # print "5 number: "
    # print (preds == 5).sum()
    #
    # print "6 start number: "
    # print (preds_6 == 1).sum()
    # print "6 number: "
    # print (preds == 6).sum()
    #
    # print "7 start number: "
    # print (preds_7 == 1).sum()
    # print "7 number: "
    # print (preds == 7).sum()
    #
    # print "8 start number: "
    # print (preds_8 == 1).sum()
    # print "8 number: "
    # print (preds == 8).sum()
    #
    # print "9 start number: "
    # print (preds_9 == 1).sum()
    # print "9 number: "
    # print (preds == 9).sum()
    #
    # np.savetxt('submission_tf_nn.csv',np.c_[range(1,28001),preds],
    #                 delimiter=',',header='ImageId,Label',comments='',fmt='%d')


if __name__ == '__main__':
    main()

