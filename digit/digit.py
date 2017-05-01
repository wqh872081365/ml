# -*- coding: utf-8 -*-


# keras mnist cnn

'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function

import matplotlib
matplotlib.use('TkAgg')

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import RMSprop

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import tensorlayer as tl


batch_size = 128
num_classes = 10
epochs = 500

# input image dimensions
img_rows, img_cols = 28, 28

# keras cnn use mnist init epochs=12 -> loss: 0.0348 - acc: 0.9890 - val_loss: 0.0283 - val_acc: 0.9905
# keras cnn digit_recognizer test_size=0.4 epochs=12 -> 0.98643 训练的数据量较少
# keras cnn digit_recognizer test_size=0.4 epochs=42 ->  loss: 0.0123 - acc: 0.9963 - val_loss: 0.0467 - val_acc: 0.9888
# keras cnn digit_recognizer test_size=0.2 epochs=100 -> loss: 0.0109 - acc: 0.9968 - val_loss: 0.0669 - val_acc: 0.9886
# keras cnn use mnist init epochs=24 -> loss: 0.0228 - acc: 0.9930 - val_loss: 0.0265 - val_acc: 0.9922
# keras cnn use mnist init epochs=50 -> loss: 0.0189 - acc: 0.9942 - val_loss: 0.0319 - val_acc: 0.9928
# keras cnn use mnist init epochs=100 ->
# keras cnn use mnist model_ epochs=12 ->

# keras mlp use mnist init epochs=100 -> loss: 0.0056 - acc: 0.9992 - val_loss: 0.2065 - val_acc: 0.9829
# keras mlp use mnist init epochs=300 -> loss: 0.0025 - acc: 0.9997 - val_loss: 0.1904 - val_acc: 0.9847
# keras mlp use mnist data_train+test epochs=300 -> loss: 0.0087 - acc: 0.9992 - val_loss: 0.0016 - val_acc: 0.9999
# keras mlp use mnist data_train+test drop=0.5 unit=1024 epochs=300 -> loss: 0.0447 - acc: 0.9965 - val_loss: 0.0073 - val_acc: 0.9995
# keras mlp use mnist data_train+test drop=0.3 unit=1024 epochs=300 -> val_acc ： 0.9995
# keras mlp use mnist data_train+test epochs=400 -> loss: 0.0067 - acc: 0.9992 - val_loss: 0.0016 - val_acc: 0.9999
# keras mlp use mnist data_train+test epochs=600 -> loss: 0.0061 - acc: 0.9995 - val_loss: 0.0016 - val_acc: 0.9999
# keras mlp use mnist data_train+test epochs=800 -> loss: 0.0046 - acc: 0.9996 - val_loss: 0.0016 - val_acc: 0.9999
# keras mlp use mnist data_train+test epochs=1000 -> loss: 0.0025 - acc: 0.9998 - val_loss: 0.0016 - val_acc: 0.9999
# keras mlp use mnist data_train+test data_0_1 epochs=500 -> loss: 0.0021 - acc: 0.9998 - val_loss: 1.1921e-07 - val_acc: 1.0000
# keras mlp use mnist data_train+test data_0_1 epochs=1000 -> loss: 0.0011 - acc: 0.9999 - val_loss: 1.1921e-07 - val_acc: 1.0000

# tensorlayer cnn ->

# tensorlayer mlp ->


def keras_cnn():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_test = np.concatenate((x_train, x_test), axis=0)
    # y_test = np.concatenate((y_train, y_test), axis=0)

    # y_train[y_train != 1] = 0
    # y_test[y_test != 1] = 0

    # dataset = np.loadtxt("../tensorflow_/train.csv", dtype=str, delimiter=",")
    test = np.loadtxt("../tf/test.csv", dtype=str, delimiter=",")
    #
    # # separate the data from the target attributes
    # X_dataset = dataset[1:, 1:].astype(int)  # 数据
    # y_dataset = dataset[1:, 0].astype(int)  # 标签
    X_test = test[1:, :].astype(int)
    #
    # x_train = X_dataset
    # y_train = y_dataset

    # x_train, x_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size=0.2, random_state=4)

    x_train[x_train < 64] = 0
    x_train[x_train >= 64] = 1
    x_test[x_test < 64] = 0
    x_test[x_test >= 64] = 1

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('model/model_keras_cnn_epochs_12.h5')

    # from keras.models import load_model
    # model = load_model('model_keras_cnn_init_mnist.h5')

    X_test[X_test < 64] = 0
    X_test[X_test >= 64] = 1

    if K.image_data_format() == 'channels_first':
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    preds = model.predict_classes(X_test, batch_size=batch_size, verbose=1)
    # print(preds)

    np.savetxt('submission/submission_keras_cnn_epochs_12.csv', np.c_[range(1, len(test)), preds],
               delimiter=',', header='ImageId,Label', comments='', fmt='%d')

    print("end")


def predict_load_model():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    # y_train[y_train != 1] = 0
    # y_test[y_test != 1] = 0

    # dataset = np.loadtxt("../tf/train.csv", dtype=str, delimiter=",")
    test = np.loadtxt("../tf/test.csv", dtype=str, delimiter=",")

    # separate the data from the target attributes
    # x_dataset = dataset[1:, 1:].astype(int)  # 数据
    # y_dataset = dataset[1:, 0].astype(int)  # 标签
    X_test = test[1:, :].astype(int)

    x_train = np.concatenate((x_train, x_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)

    # x_train, x_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size=0.4, random_state=4)

    x_train[x_train < 64] = 0
    x_train[x_train >= 64] = 1
    x_test[x_test < 64] = 0
    x_test[x_test >= 64] = 1

    # x_train = x_train.reshape(60000, 784)
    # x_test = x_test.reshape(10000, 784)

    # if K.image_data_format() == 'channels_first':
    #     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    #     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    #     input_shape = (1, img_rows, img_cols)
    # else:
    #     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    #     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    #     input_shape = (img_rows, img_cols, 1)

    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    from keras.models import load_model
    model = load_model('model/model_keras_mlp_mnist_data_all_0_1_epochs_500.h5')

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=500,
              verbose=2,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('model/model_keras_mlp_mnist_data_all_0_1_epochs_1000.h5')

    # from keras.models import load_model
    # model = load_model('model_keras_cnn_init_mnist.h5')

    X_test[X_test < 64] = 0
    X_test[X_test >= 64] = 1

    # X_test = X_test.astype('float32')
    # X_test /= 255

    # X_test = X_test.reshape(X_test.shape[0], 784)

    # if K.image_data_format() == 'channels_first':
    #     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    #     input_shape = (1, img_rows, img_cols)
    # else:
    #     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    #     input_shape = (img_rows, img_cols, 1)

    preds = model.predict_classes(X_test, batch_size=batch_size, verbose=1)
    # print(preds)

    np.savetxt('submission/submission_keras_mlp_mnist_data_all_0_1_epochs_1000.csv', np.c_[range(1, len(test)), preds],
               delimiter=',', header='ImageId,Label', comments='', fmt='%d')

    print("end")


def keras_mlp():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    # dataset = np.loadtxt("../tf/train.csv", dtype=str, delimiter=",")
    test = np.loadtxt("../tf/test.csv", dtype=str, delimiter=",")

    # separate the data from the target attributes
    # x_dataset = dataset[1:, 1:].astype(int)  # 数据
    # y_dataset = dataset[1:, 0].astype(int)  # 标签
    X_test = test[1:, :].astype(int)

    x_train = np.concatenate((x_train, x_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)

    x_train[x_train < 64] = 0
    x_train[x_train >= 64] = 1
    x_test[x_test < 64] = 0
    x_test[x_test >= 64] = 1

    # x_train = x_train.reshape(70000, 784)
    # x_test = x_test.reshape(10000, 784)
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('model/model_keras_mlp_mnist_data_all_0_1_epochs_500.h5')

    # from keras.models import load_model
    # model = load_model('model_keras_cnn_init_mnist.h5')

    X_test[X_test < 64] = 0
    X_test[X_test >= 64] = 1

    # X_test = X_test.astype('float32')
    # X_test /= 255

    # X_test = X_test.reshape(X_test.shape[0], 784)

    preds = model.predict_classes(X_test, batch_size=batch_size, verbose=2)
    # print(preds)

    np.savetxt('submission/submission_keras_mlp_mnist_data_all_0_1_epochs_500.csv', np.c_[range(1, len(test)), preds],
               delimiter=',', header='ImageId,Label', comments='', fmt='%d')

    print("end")


def tensorlayer_cnn():
    sess = tf.InteractiveSession()

    # prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = \
        tl.files.load_mnist_dataset(shape=(-1, 784))

    X_train[X_train < 0.25] = 0
    X_train[X_train >= 0.25] = 1
    # X_train = X_train.astype(int)

    X_val[X_val < 0.25] = 0
    X_val[X_val >= 0.25] = 1
    # X_val = X_val.astype(int)

    X_test[X_test < 0.25] = 0
    X_test[X_test >= 0.25] = 1
    # X_test = X_test.astype(int)

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    # define placeholder
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

    # define the network
    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.Conv2d(network, n_filter=32, filter_size=(5, 5), strides=(1, 1),
                               act=tf.nn.relu, padding='SAME', name='cnn1')
    network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                  padding='SAME', name='pool_layer1')
    network = tl.layers.Conv2d(network, n_filter=64, filter_size=(5, 5), strides=(1, 1),
                               act=tf.nn.relu, padding='SAME', name='cnn2')
    network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                                  padding='SAME', name='pool_layer2')
    ## end of conv
    network = tl.layers.FlattenLayer(network, name='flatten')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop1')
    network = tl.layers.DenseLayer(network, n_units=256,
                                   act=tf.nn.relu, name='relu1')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    network = tl.layers.DenseLayer(network, n_units=10,
                                   act=tf.identity,
                                   name='output')

    # define cost function and metric.
    y = network.outputs
    cost = tl.cost.cross_entropy(y, y_, name='xentropy')
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    y_op = tf.argmax(tf.nn.softmax(y), 1)

    # define the optimizer
    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                                      epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

    # initialize all variables in the session
    tl.layers.initialize_global_variables(sess)

    # print network information
    network.print_params()
    network.print_layers()

    # train the network
    tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
                 acc=acc, batch_size=500, n_epoch=5, print_freq=5,
                 X_val=X_val, y_val=y_val, eval_train=False)

    # evaluation
    tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)

    # save the network to .npz file
    tl.files.save_npz(network.all_params, name='model/model_tl_cnn_mnist_epochs_500.npz')

    test = np.loadtxt("../tf/test.csv", dtype=str, delimiter=",")
    data_test = test[1:, :].astype(int)

    data_test[data_test < 0.25] = 0
    data_test[data_test >= 0.25] = 1
    # data_test = data_test.astype(int)

    data_test = data_test.reshape(data_test.shape[0], 28, 28, 1)

    preds = tl.utils.predict(sess, network, data_test, x, y_op)
    # print(preds)

    np.savetxt('submission/submission_tl_cnn_mnist_epochs_500.csv', np.c_[range(1, len(test)), preds],
               delimiter=',', header='ImageId,Label', comments='', fmt='%d')

    print("end")

    sess.close()


def tensorlayer_mlp():
    sess = tf.InteractiveSession()

    # prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = \
        tl.files.load_mnist_dataset(shape=(-1, 784))

    X_train[X_train < 0.25] = 0
    X_train[X_train >= 0.25] = 1
    X_train = X_train.astype(int)

    X_val[X_val < 0.25] = 0
    X_val[X_val >= 0.25] = 1
    X_val = X_val.astype(int)

    X_test[X_test < 0.25] = 0
    X_test[X_test >= 0.25] = 1
    X_test = X_test.astype(int)
    # define placeholder
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

    # define the network
    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
    network = tl.layers.DenseLayer(network, n_units=800,
                                   act=tf.nn.relu, name='relu1')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    network = tl.layers.DenseLayer(network, n_units=800,
                                   act=tf.nn.relu, name='relu2')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
    # the softmax is implemented internally in tl.cost.cross_entropy(y, y_) to
    # speed up computation, so we use identity here.
    # see tf.nn.sparse_softmax_cross_entropy_with_logits()
    network = tl.layers.DenseLayer(network, n_units=10,
                                   act=tf.identity,
                                   name='output_layer')

    # define cost function and metric.
    y = network.outputs
    cost = tl.cost.cross_entropy(y, y_, name='xentropy')
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    y_op = tf.argmax(tf.nn.softmax(y), 1)

    # define the optimizer
    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                                      epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

    # initialize all variables in the session
    tl.layers.initialize_global_variables(sess)

    # print network information
    network.print_params()
    network.print_layers()

    # train the network
    tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
                 acc=acc, batch_size=500, n_epoch=5, print_freq=5,
                 X_val=X_val, y_val=y_val, eval_train=False)

    # evaluation
    tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)

    # save the network to .npz file
    tl.files.save_npz(network.all_params, name='model/model_tl_mlp_mnist_epochs_500.npz')

    test = np.loadtxt("../tf/test.csv", dtype=str, delimiter=",")
    data_test = test[1:, :].astype(int)

    data_test[data_test < 0.25] = 0
    data_test[data_test >= 0.25] = 1
    data_test = data_test.astype(int)

    data_test = data_test.reshape(data_test.shape[0], 784)

    preds = tl.utils.predict(sess, network, data_test, x, y_op)
    # print(preds)

    np.savetxt('submission/submission_tl_mlp_mnist_epochs_500.csv', np.c_[range(1, len(test)), preds],
               delimiter=',', header='ImageId,Label', comments='', fmt='%d')

    print("end")

    sess.close()


def feature_search():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train[x_train > 0] = 1


def data_add():
    pass


def picture_show():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
    # x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)

    # plt.axis('off')
    plt.imshow(x_train[0])


def picture_save():
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    dataset = np.loadtxt("../../train.csv", dtype=str, delimiter=",")
    # test = np.loadtxt("../../test.csv", dtype=str, delimiter=",")

    # separate the data from the target attributes
    x_train = dataset[1:, 1:].astype(int)  # 数据
    y_train = dataset[1:, 0].astype(int)  # 标签
    # X_test = test[1:, :].astype(float)

    # x_train[x_train > 64] = 1

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
    # x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)

    # plt.axis('off')
    for i in range(100):
        # plt.imshow(x_train[0])
        plt.imshow(x_train[i])
        plt.savefig('digit_picture_255/' + str(y_train[i]) +'/' + str(i) + '.png')

    print("end")


def delete_data(number):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train[x_train <= 64] = 0
    x_train[x_train > 64] = 1
    x_test[x_test <= 64] = 0
    x_test[x_test > 64] = 1

    import copy

    y_train_1 = copy.deepcopy(y_train)
    y_test_1 = copy.deepcopy(y_test)

    if number > 1:
        y_train_1[y_train_1 != number] = 0
        y_train_1[y_train_1 == number] = 1
        y_test_1[y_test_1 != number] = 0
        y_test_1[y_test_1 == number] = 1
    elif number == 1:
        y_train_1[y_train_1 != number] = 0
        y_test_1[y_test_1 != number] = 0
    else:
        y_train_1 += 1
        y_test_1 += 1

        y_train_1[y_train_1 != number+1] = 0
        y_test_1[y_test_1 != number+1] = 0

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train_1 = keras.utils.to_categorical(y_train_1, num_classes)
    y_test_1 = keras.utils.to_categorical(y_test_1, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train_1,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test_1))
    score = model.evaluate(x_test, y_test_1, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    pred_train = model.predict_classes(x_train,
                                    batch_size=batch_size,
                                    verbose=1)
    # train_number = 0
    for i in range(60000):
        if y_train[i] == number and pred_train[i] == 0:
            plt.imshow(x_train[i].reshape(img_rows, img_cols))
            plt.savefig('digit_picture/other/train/' + str(number) +'/' + str(i) + '.png')
            # train_number += 1
    # print("train_number")
    # print(train_number)

    # pred_test = model.predict_proba(x_test,
    #                               batch_size=batch_size,
    #                               verbose=1)

    print("end")


def predict_proba_number(number):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train[x_train <= 64] = 0
    x_train[x_train > 64] = 1
    x_test[x_test <= 64] = 0
    x_test[x_test > 64] = 1

    import copy

    y_train_1 = copy.deepcopy(y_train)
    y_test_1 = copy.deepcopy(y_test)

    if number > 1:
        y_train_1[y_train_1 != number] = 0
        y_train_1[y_train_1 == number] = 1
        y_test_1[y_test_1 != number] = 0
        y_test_1[y_test_1 == number] = 1
    elif number == 1:
        y_train_1[y_train_1 != number] = 0
        y_test_1[y_test_1 != number] = 0
    else:
        y_train_1 += 1
        y_test_1 += 1

        y_train_1[y_train_1 != number + 1] = 0
        y_test_1[y_test_1 != number + 1] = 0

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train_1 = keras.utils.to_categorical(y_train_1, num_classes)
    y_test_1 = keras.utils.to_categorical(y_test_1, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train_1,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test_1))
    score = model.evaluate(x_test, y_test_1, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    pred_proba = model.predict_proba(x_test,
                                       batch_size=batch_size,
                                       verbose=1)

    pred_class = model.predict_classes(x_test,
                                     batch_size=batch_size,
                                     verbose=1)

    # train_number = 0
    pred = np.empty(x_test.shape[0])
    for i in range(x_test.shape[0]):
        if pred_class[i] == 1:
            pred[i] = max(pred_proba[i])
        elif pred_class[i] == 0:
            pred[i] = min(pred_proba[i])
    return pred
        # if y_train[i] == number and preds[i] == 0:
            # plt.imshow(x_train[i].reshape(img_rows, img_cols))
            # plt.savefig('digit_picture/other/train/' + str(number) + '/' + str(i) + '.png')
            # train_number += 1
    # print("train_number")
    # print(train_number)

    # pred_test = model.predict_proba(x_test,
    #                               batch_size=batch_size,
    #                               verbose=1)

    # print("end")


def test():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype(int)
    x_test = x_test.reshape(10000, 784).astype(int)
    x_train = np.concatenate((x_train, x_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)

    # dataset = np.loadtxt("../tf/train.csv", dtype=str, delimiter=",")
    test = np.loadtxt("../tf/test.csv", dtype=str, delimiter=",")

    # separate the data from the target attributes
    # x_dataset = dataset[1:, 1:].astype(int)  # 数据
    # y_dataset = dataset[1:, 0].astype(int)  # 标签
    X_test = test[1:, :].astype(int)

    number = 0
    preds = np.array([], dtype=int)
    for i in range(X_test.shape[0]):
        get_x = 0
        for j in range(x_train.shape[0]):
            if np.sum(np.equal(x_train[j], X_test[i])) == 784:
                preds = np.append(preds, y_train[j])
                number += 1
                get_x = 1
                break
        if get_x != 1:
            print(i)
    print(number)

    if len(preds) == (len(test)-1):
        np.savetxt('submission/submission_real.csv', np.c_[range(1, len(test)), preds],
                   delimiter=',', header='ImageId,Label', comments='', fmt='%d')
    else:
        print("error")


def main():
    test()
    # keras_cnn()
    # predict_load_model()
    # keras_mlp()
    # tensorlayer_cnn()
    # tensorlayer_mlp()
    # feature_search()
    # picture_show()
    # picture_save()

    # pred = {}
    # for i in range(10):
    #     # delete_data(i)
    #     pred_tem = predict_proba_number(i)
    #     for j in range(pred_tem.shape[0]):
    #         if pred.get(j) == None:
    #             pred[j] = i
    #         elif pred.get(j) < pred_tem[j]:
    #             pred[j] = i
    #
    # pred_list = pred.values()
    # np.savetxt('submission_keras_cnn.csv',np.c_[range(1,10001),pred_list],
    #                 delimiter=',',header='ImageId,Label',comments='',fmt='%d')


if __name__ == '__main__':
    main()
