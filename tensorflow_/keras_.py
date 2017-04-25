# -*- coding: utf-8 -*-

'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf


batch_size = 128
num_classes = 2
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

def mnist_cnn():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train[x_train <= 64] = 0
    x_train[x_train > 64] = 1
    x_test[x_test <= 64] = 0
    x_test[x_test > 64] = 1

    y_train[y_train != 1] = 0
    y_test[y_test != 1] = 0

    # dataset = np.loadtxt("../../train.csv", dtype=str, delimiter=",")
    # test = np.loadtxt("../../test.csv", dtype=str, delimiter=",")
    #
    # # separate the data from the target attributes
    # X_dataset = dataset[1:, 1:].astype(float)  # 数据
    # y_dataset = dataset[1:, 0].astype(float)  # 标签
    # X_test = test[1:, :].astype(float)
    #
    # x_train, x_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size=0.4, random_state=4)

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
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (5, 5), activation='relu'))
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

    # if K.image_data_format() == 'channels_first':
    #     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    #     input_shape = (1, img_rows, img_cols)
    # else:
    #     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    #     input_shape = (img_rows, img_cols, 1)

    # X_test = X_test.astype('float32')
    # X_test /= 255
    # print(X_test.shape[0], 'test samples')

    # preds = model.predict_classes(X_test, batch_size=batch_size, verbose=1)
    # print(preds)
    #
    # np.savetxt('submission_keras_nn_1.csv', np.c_[range(1, len(test)), preds],
    #            delimiter=',', header='ImageId,Label', comments='', fmt='%d')


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


def main():
    # mnist_cnn()
    # feature_search()
    # picture_show()
    # picture_save()
    pred = {}
    for i in range(10):
        # delete_data(i)
        pred_tem = predict_proba_number(i)
        for j in range(pred_tem.shape[0]):
            if pred.get(j) == None:
                pred[j] = i
            elif pred.get(j) < pred_tem[j]:
                pred[j] = i

    pred_list = pred.values()
    np.savetxt('submission_keras_cnn.csv',np.c_[range(1,10001),pred_list],
                    delimiter=',',header='ImageId,Label',comments='',fmt='%d')


if __name__ == '__main__':
    main()
