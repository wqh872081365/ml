#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function

import numpy as np
import tflearn

import tensorflow as tf
import tensorlayer as tl


def tflearn_example():
    # Download the Titanic dataset
    from tflearn.datasets import titanic
    titanic.download_dataset('titanic_dataset.csv')

    # Load CSV file, indicate that the first column represents labels
    from tflearn.data_utils import load_csv
    data, labels = load_csv('titanic_dataset.csv', target_column=0,
                            categorical_labels=True, n_classes=2)


    # Preprocessing function
    def preprocess(data, columns_to_ignore):
        # Sort by descending id and delete columns
        for id in sorted(columns_to_ignore, reverse=True):
            [r.pop(id) for r in data]
        for i in range(len(data)):
          # Converting 'sex' field to float (id is 1 after removing labels column)
          data[i][1] = 1. if data[i][1] == 'female' else 0.
        return np.array(data, dtype=np.float32)

    # Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
    to_ignore=[1, 6]

    # Preprocess data
    data = preprocess(data, to_ignore)

    # Build neural network
    net = tflearn.input_data(shape=[None, 6])
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net)

    # Define model
    model = tflearn.DNN(net)
    # Start training (apply gradient descent algorithm)
    model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

    # Let's create some data for DiCaprio and Winslet
    dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
    winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]
    # Preprocess data
    dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
    # Predict surviving chances (class 1 results)
    pred = model.predict([dicaprio, winslet])
    print("DiCaprio Surviving Rate:", pred[0][1])
    print("Winslet Surviving Rate:", pred[1][1])


def digit_recognizer():

    dataset = np.loadtxt("train.csv", dtype=str, delimiter=",")
    test = np.loadtxt("test.csv", dtype=str, delimiter=",")
    # separate the data from the target attributes
    data = dataset[1:, 1:].astype(float)  # 数据
    labels = dataset[1:, 0].astype(float)  # 标签
    test_X = test[1:, :].astype(float)

    from sklearn import preprocessing
    # normalize the data attributes
    normalized_X = preprocessing.normalize(data)
    X_test = preprocessing.normalize(test_X)

    # Build neural network
    net = tflearn.input_data(shape=[None, 784])
    net = tflearn.fully_connected(net, 64)
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

    # Define model
    model = tflearn.DNN(net)
    # Start training (apply gradient descent algorithm)
    model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

    # # Predict surviving chances (class 1 results)
    # preds = model.predict(X_test)
    #
    # # 保持数据
    # np.savetxt('submission_tflearn_dnn.csv',np.c_[range(1,len(test)),preds],
    #                 delimiter=',',header='ImageId,Label',comments='',fmt='%d')



def main():
    # tflearn_example()
    digit_recognizer()


if __name__ == '__main__':
    main()

