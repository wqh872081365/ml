# -*- coding: utf-8 -*-

# rainforest

# keras cnn data_35000 f2_0.235 data_origin epochs=11 -> loss: 0.1565 - acc: 0.9382 - val_loss: 0.1470 - val_acc: 0.9418 F2=0.85519 score=0.85192
# keras cnn data_35000 f2_ data_origin epochs=200 -> F2= score=

import matplotlib
matplotlib.use('TkAgg')

import tensorflow as tf
import sys
import os
import subprocess
from six import string_types
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
from skimage import io
from scipy import ndimage
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras.optimizers import RMSprop


def make_cooccurence_matrix(df_labels, labels):
    numeric_df = df_labels[labels]
    c_matrix = numeric_df.T.dot(numeric_df)
    c_matrix_35000 = (numeric_df.iloc[0:35000, 0:17]).T.dot(numeric_df.iloc[0:35000, 0:17])
    sns.heatmap(c_matrix, xticklabels=True, yticklabels=True)
    return c_matrix


def test_submission_ver():
    df_labels = pd.read_csv("submission/submission_keras_cnn_init_epochs_1.csv") # nan
    label_list = []
    for tag_str in df_labels.tags.values:
        if tag_str:
            labels = tag_str.split(' ')
            for label in labels:
                if label not in label_list:
                    label_list.append(label)
    print(len(label_list))
    for label in label_list:
        df_labels[label] = df_labels['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)

    make_cooccurence_matrix(df_labels, label_list)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()


def save_image():
    # use cv2
    for i in tqdm(range(40479), miniters=1000):
        img = cv2.imread('data/train-jpg/train_'+str(i)+'.jpg')
        img = cv2.resize(img, (32, 32))
        cv2.imwrite('data/train-jpg-32/train_'+str(i)+'.png', img)

    for i in tqdm(range(40669), miniters=1000):
        img = cv2.imread('data/test-jpg/test_'+str(i)+'.jpg')
        img = cv2.resize(img, (32, 32))
        cv2.imwrite('data/test-jpg-32/test_'+str(i)+'.png', img)


def kears_cnn():
    train_df = pd.read_csv("data/train_list.csv")
    label_list = train_df.columns.values[2:]

    x_train = []
    x_test = []

    for i in tqdm(range(40479), miniters=1000):
        img = cv2.imread('data/train-jpg/train_' + str(i) + '.jpg')
        x_train.append(cv2.resize(img, (32, 32)))

    for i in tqdm(range(40669), miniters=1000):
        img = cv2.imread('data/test-jpg/test_' + str(i) + '.jpg')
        x_test.append(cv2.resize(img, (32, 32)))

    x_train = np.array(x_train, np.float16) / 255.
    x_test = np.array(x_test, np.float16) / 255.
    y_train = train_df.values[:, 2:19]

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)

    x_train, x_valid, y_train, y_valid = x_train[:35000], x_train[35000:], y_train[:35000], y_train[35000:]

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(32, 32, 3)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',
                  # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=10,
              verbose=1,
              validation_data=(x_valid, y_valid))
    score = model.evaluate(x_valid, y_valid, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('model/model_keras_cnn_data_origin_epochs_10.h5')

    from sklearn.metrics import fbeta_score

    p_valid = model.predict(x_valid, batch_size=128)
    np.savetxt('data/data_keras_cnn_data_origin_epochs_10_pred.csv', p_valid,
                   delimiter=',', comments='', fmt='%.5f')

    dict_pred = {}
    for i in range(999):
        dict_pred[((i + 1) / 1000.0)] = fbeta_score(y_valid == 1, p_valid > ((i + 1) / 1000.0), beta=2, average='samples') # 随着模型的完善，0.2这个也可能需要改进，多选几个值输出F2；
    print(max(dict_pred, key=dict_pred.get))
    print(dict_pred[max(dict_pred, key=dict_pred.get)])

    p_test = model.predict(x_test, batch_size=128)
    preds = []
    for i in range(p_test.shape[0]):
        pred_list = []
        for j in range(len(label_list)):
            if p_test[i, j] > dict_pred[max(dict_pred, key=dict_pred.get)]:
                pred_list.append(label_list[j])
        if len(pred_list) == 0:
            pred_list.append(label_list[np.argmax(p_test[i])])
            print(i)
            print(max(p_test[i]))
        preds.append(' '.join(pred_list))

    # python2
    index_preds = map(lambda x: "test_" + str(x), range(len(preds)))
    ## python3
    # index_preds = list(map(lambda x: "test_" + str(x), range(len(preds))))

    print(len(index_preds))
    print(len(preds))
    preds_data = np.c_[index_preds, preds]
    print(preds_data.shape)
    np.savetxt('submission/submission_keras_cnn_data_origin_epochs_10.csv', preds_data,
               delimiter=',', header='image_name,tags', comments='', fmt='%s')


def keras_mlp():
    train_df = pd.read_csv("data/train_list.csv")
    label_list = train_df.columns.values[2:]

    x_train = []
    x_test = []

    for i in tqdm(range(40479), miniters=1000):
        img = cv2.imread('data/train-jpg-32/train_' + str(i) + '.png')
        x_train.append(img)

    for i in tqdm(range(40669), miniters=1000):
        img = cv2.imread('data/test-jpg-32/test_' + str(i) + '.png')
        x_test.append(img)

    x_train = np.array(x_train, np.float16) / 255.
    x_test = np.array(x_test, np.float16) / 255.
    y_train = train_df.values[:, 2:19]

    x_train = x_train.reshape(x_train.shape[0], 3072)
    x_test = x_test.reshape(x_test.shape[0], 3072)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.19049, random_state=4)

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(3072,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(17, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',
                  # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=10,
              verbose=2,
              validation_data=(x_valid, y_valid))
    score = model.evaluate(x_valid, y_valid, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('model/model_keras_mlp_epochs_1.h5')

    from sklearn.metrics import fbeta_score
    p_valid = model.predict(x_valid, batch_size=128)
    print(fbeta_score(y_valid == 1, p_valid > 0.5, beta=2, average='samples'))

    p_test = model.predict(x_test, batch_size=128)
    preds = []
    for i in range(p_test.shape[0]):
        pred_list = []
        for j in range(len(label_list)):
            if p_test[i, j] > 0.5:
                pred_list.append(label_list[j])
        if len(pred_list) == 0:
            pred_list.append(label_list[np.argmax(p_test[i])])
            print(i)
            print(max(p_test[i]))
        preds.append(' '.join(pred_list))

    # python2
    index_preds = map(lambda x: "test_" + str(x), range(len(preds)))
    ## python3
    # index_preds = list(map(lambda x: "test_" + str(x), range(len(preds))))

    print(len(index_preds))
    print(len(preds))
    preds_data = np.c_[index_preds, preds]
    print(preds_data.shape)
    np.savetxt('submission/submission_keras_mlp_epochs_1.csv', preds_data,
               delimiter=',', header='image_name,tags', comments='', fmt='%s')


def predict_load_model():
    train_df = pd.read_csv("data/train_list.csv")
    label_list = train_df.columns.values[2:]

    x_train = []
    x_test = []

    for i in tqdm(range(40479), miniters=1000):
        img = cv2.imread('data/train-jpg/train_' + str(i) + '.jpg')
        x_train.append(cv2.resize(img, (32, 32)))

    for i in tqdm(range(40669), miniters=1000):
        img = cv2.imread('data/test-jpg/test_' + str(i) + '.jpg')
        x_test.append(cv2.resize(img, (32, 32)))

    x_train = np.array(x_train, np.float16) / 255.
    x_test = np.array(x_test, np.float16) / 255.
    y_train = train_df.values[:, 2:19]

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)

    x_train, x_valid, y_train, y_valid = x_train[:35000], x_train[35000:], y_train[:35000], y_train[35000:]

    model = load_model('model/model_keras_cnn_data_origin_epochs_10.h5')

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=190,
              verbose=2,
              validation_data=(x_valid, y_valid))
    score = model.evaluate(x_valid, y_valid, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('model/model_keras_cnn_data_origin_epochs_200.h5')

    from sklearn.metrics import fbeta_score

    p_valid = model.predict(x_valid, batch_size=128)
    np.savetxt('data/data_keras_cnn_data_origin_epochs_200_pred.csv', p_valid,
                   delimiter=',', comments='', fmt='%.5f')
    # p_valid = np.loadtxt('data/data_keras_cnn_data_origin_epochs_10_pred.csv', dtype=float, delimiter=",")

    dict_pred = {}
    for i in range(999):
        dict_pred[((i + 1) / 1000.0)] = fbeta_score(y_valid == 1, p_valid > ((i + 1) / 1000.0), beta=2,
                                                    average='samples')  # 随着模型的完善，0.2这个也可能需要改进，多选几个值输出F2；
    print(max(dict_pred, key=dict_pred.get))
    print(dict_pred[max(dict_pred, key=dict_pred.get)])

    print(fbeta_score(y_valid == 1, p_valid > 0.2, beta=2, average='samples'))

    p_test = model.predict(x_test, batch_size=128)
    preds = []
    for i in range(p_test.shape[0]):
        pred_list = []
        for j in range(len(label_list)):
            if p_test[i, j] > max(dict_pred, key=dict_pred.get):
                pred_list.append(label_list[j])
        if len(pred_list) == 0:
            pred_list.append(label_list[np.argmax(p_test[i])])
            print(i)
            print(max(p_test[i]))
            print(label_list[np.argmax(p_test[i])])
        preds.append(' '.join(pred_list))

    # python2
    index_preds = map(lambda x: "test_" + str(x), range(len(preds)))
    ## python3
    # index_preds = list(map(lambda x: "test_" + str(x), range(len(preds))))

    print(len(index_preds))
    print(len(preds))
    preds_data = np.c_[index_preds, preds]
    print(preds_data.shape)
    np.savetxt('submission/submission_keras_cnn_data_origin_epochs_200.csv', preds_data,
               delimiter=',', header='image_name,tags', comments='', fmt='%s')


def get_train_test_data():
    train_df = pd.read_csv("data/train_list.csv")
    label_list = train_df.columns.values[2:]
    # y_train = train_df.values[:, 2:19]
    # y_train_str = train_df.values[:, 1]
    # data_dict = {}
    # for i in range(y_train_str.shape[0]):
    #     if data_dict.get(y_train_str[i]):
    #         data_dict[y_train_str[i]] += 1
    #     else:
    #         data_dict[y_train_str[i]] = 1
    # print(sorted(data_dict.values()))

    make_cooccurence_matrix(train_df, label_list)
    plt.show()



def main():
    # test_submission_ver()
    # save_image()
    # kears_cnn()
    predict_load_model()
    # keras_mlp()
    # get_train_test_data()


if __name__ == '__main__':
    main()

