# -*- coding: utf-8 -*-

# rainforest

# keras cnn data_35000 f2_0.235 data_origin epochs=11 -> loss: 0.1565 - acc: 0.9382 - val_loss: 0.1470 - val_acc: 0.9418 F2=0.85519 score=0.85192
# keras cnn data_35000 f2_ data_origin epochs=200 -> F2= score=

# v2
# keras cnn data_35000 f2_0.188 data_origin optimizer=adam lr=0.001 epochs=10 -> loss: 0.1651 - acc: 0.9354 - val_loss: 0.1529 - val_acc: 0.9399 F2=0.85493 score=0.85520
# keras cnn data_35000 f2_0.144 data_origin optimizer=adam lr=0.0001 epochs=10 -> loss: 0.2032 - acc: 0.9209 - val_loss: 0.1866 - val_acc: 0.9251 F2=0.81318
# keras cnn data_35000 f2_0.216 data_origin optimizer=adam lr=0.001 epochs=40 -> loss: 0.1164 - acc: 0.9515 - val_loss: 0.1414 - val_acc: 0.9471 F2=0.87793 score=
# keras cnn data_0.9 f2_0.208 data_origin data*4 optimizer=adam lr=0.001 epochs=10 -> loss: 0.1350 - acc: 0.9463 - val_loss: 0.1268 - val_acc: 0.9484 F2=0.88531 score=0.88291
# keras cnn data_0.9 f2_0.209 data_origin drop_0.5 optimizer=adam lr=0.001 epochs=10 -> loss: 0.1582 - acc: 0.9378 - val_loss: 0.1505 - val_acc: 0.9389 F2=0.86078 score=0.86417
# keras cnn data_0.9 f2_0.189 data_origin data*4 drop_0.5 optimizer=adam lr=0.001 epochs=10 -> F2=0.8868 score=0.88490
# keras cnn data_0.9 f2_0.227 data_origin data*4 drop_0.5 optimizer=adam lr=0.001 reduce epochs=178 -> loss: 0.1019 - acc: 0.9590 - val_loss: 0.1108 - val_acc: 0.9566 F2=0.904966 score=0.90191
# keras cnn data_0.9 f2_0.182 data_origin ImageDataGenerator drop_0.5 optimizer=adam lr=0.001 reduce epochs=280 -> loss: 0.1270 - acc: 0.9506 - val_loss: 0.1165 - val_acc: 0.9526 F2=0.8972 score=0.89957
# keras cnn data_0.9 f2_0.2 data_origin data*4 ImageDataGenerator drop_0.25 optimizer=adam lr=0.001 reduce epochs=199 -> acc:0.95796 loss:0.10678 val_acc: 0.96189 val_loss: 0.095187 F2=0.9166 score=0.90697


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
import json

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras.optimizers import RMSprop, Adadelta, Adam
from keras import losses
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from keras.applications import InceptionV3, ResNet50, VGG16


def make_cooccurence_matrix(df_labels, labels):
    numeric_df = df_labels[labels]
    c_matrix = numeric_df.T.dot(numeric_df)
    c_matrix_35000 = (numeric_df.iloc[0:35000, 0:17]).T.dot(numeric_df.iloc[0:35000, 0:17])
    sns.heatmap(c_matrix, xticklabels=True, yticklabels=True)
    return c_matrix


def get_train_list():
    labels_df = pd.read_csv("data/train_v2.csv")
    # print(labels_df.head())
    #
    # # Build list with unique labels
    label_list = []
    for tag_str in labels_df.tags.values:
        labels = tag_str.split(' ')
        for label in labels:
            if label not in label_list:
                label_list.append(label)

    # datas_df = pd.read_csv("data/sample_submission.csv")

    # Add onehot features for every label
    for label in label_list:
        labels_df[label] = labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
    # # Display head
    print(labels_df.head())
    labels_df.to_csv("data/train_v2_list.csv", index=False)


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
    train_df = pd.read_csv("data/train_v2_list.csv")
    label_list = train_df.columns.values[2:]
    train_name_list = train_df.values[:, 0]

    test_df = pd.read_csv("data/sample_submission_v2.csv")
    test_name_list = test_df.values[:, 0]
    print(test_name_list.shape)

    y_train_value = train_df.values[:, 2:19]

    x_train = []
    y_train = []
    x_test = []
    x_file = []

    for i in tqdm(range(40479), miniters=1000):
        img = cv2.imread('data/train-jpg/' + train_name_list[i] + '.jpg')
        img_32 = cv2.resize(img, (48, 48))
        x_train.append(img_32)
        x_train.append(img_32[::-1])
        x_train.append(img_32[:, ::-1])
        x_train.append(img_32[:, ::-1][::-1])

        y_train.extend([y_train_value[i], y_train_value[i], y_train_value[i], y_train_value[i]])
        # y_train.extend([y_train_value[i]])

    for i in tqdm(range(40669), miniters=1000):
        img = cv2.imread('data/test-jpg/' + test_name_list[i] + '.jpg')
        x_test.append(cv2.resize(img, (48, 48)))

    for i in tqdm(range(20522), miniters=1000):
        img = cv2.imread('data/test-jpg-additional/' + test_name_list[40669+i] + '.jpg')
        x_file.append(cv2.resize(img, (48, 48)))

    x_train = np.array(x_train, np.float16) / 255.
    y_train = np.array(y_train, np.int)
    x_test = np.array(x_test, np.float16) / 255.
    x_file = np.array(x_file, np.float16) / 255.


    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(x_file.shape)

    # x_train, x_valid, y_train, y_valid = x_train[:35000], x_train[35000:], y_train[:35000], y_train[35000:]
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=4)

    # datagen = ImageDataGenerator(
    #     featurewise_center=False,  # set input mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #     samplewise_std_normalization=False,  # divide each input by its std
    #     zca_whitening=False,  # apply ZCA whitening
    #     rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    #     width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
    #     height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
    #     horizontal_flip=True,  # randomly flip images
    #     vertical_flip=False)  # randomly flip images

    # datagen.fit(x_train)

    model = VGG16(include_top=False, weights='imagenet',
                  input_tensor=None, input_shape=(48, 48, 3),
                  pooling=None,
                  classes=1000)

    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=(32, 32, 3)))
    #
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='sigmoid'))

    model.summary()

    model.compile(loss=losses.binary_crossentropy,
                  # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                  optimizer=Adam(lr=0.001),  # adam 854931990138
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=100)
    reduce_lr_on_Plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=0, mode='auto',
                                      epsilon=0.0001, cooldown=0, min_lr=0)
    model_save = ModelCheckpoint(filepath='model/5-19/{epoch}.h5', monitor='val_loss', verbose=0, save_best_only=False,
                                    save_weights_only=False, mode='auto', period=10)

    # hist = model.fit_generator(datagen.flow(x_train, y_train,
    #                                  batch_size=128),
    #                     steps_per_epoch=x_train.shape[0]//128,
    #                     epochs=300,
    #                     verbose=1,
    #                     validation_data=(x_valid, y_valid),
    #                     callbacks=[early_stopping, reduce_lr_on_Plateau, model_save])

    hist = model.fit(x_train, y_train,
              batch_size=128,
              epochs=300,
              verbose=2,
              validation_data=(x_valid, y_valid),
              callbacks=[early_stopping, reduce_lr_on_Plateau, model_save])
    print(hist.history)
    try:
        with open('model/model_keras_cnn_vgg_data_4d_v2_optimizer_adam_drop_025_lr_reduce_epochs_300.json', 'w') as f:
            json.dump(hist.history, f)
    except:
        pass

    score = model.evaluate(x_valid, y_valid, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('model/model_keras_cnn_vgg_data_4d_v2_optimizer_adam_drop_025_lr_reduce_epochs_300.h5')

    from sklearn.metrics import fbeta_score

    p_valid = model.predict(x_valid, batch_size=128)
    # np.savetxt('data/data_keras_cnn_data_origin_epochs_10_pred.csv', p_valid,
    #                delimiter=',', comments='', fmt='%.5f')

    dict_pred = {}
    for i in range(999):
        dict_pred[((i + 1) / 1000.0)] = fbeta_score(y_valid == 1, p_valid > ((i + 1) / 1000.0), beta=2, average='samples') # 随着模型的完善，0.2这个也可能需要改进，多选几个值输出F2；
    print(max(dict_pred, key=dict_pred.get))
    print(dict_pred[max(dict_pred, key=dict_pred.get)])

    p_test = model.predict(np.concatenate((x_test, x_file)), batch_size=128)
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
        preds.append(' '.join(pred_list))
    print(len(preds))
    # python2
    # index_preds = map(lambda x: "test_" + str(x), range(len(preds)))
    ## python3
    # index_preds = list(map(lambda x: "test_" + str(x), range(len(preds))))

    # print(len(index_preds))
    # print(len(preds))
    preds_data = np.c_[test_name_list, preds]
    print(preds_data.shape)
    np.savetxt('submission/submission_keras_cnn_vgg_data_4d_v2_optimizer_adam_drop_025_lr_reduce_epochs_300.csv', preds_data,
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
    train_df = pd.read_csv("data/train_v2_list.csv")
    label_list = train_df.columns.values[2:]
    train_name_list = train_df.values[:, 0]

    test_df = pd.read_csv("data/sample_submission_v2.csv")
    test_name_list = test_df.values[:, 0]
    print(test_name_list.shape)

    y_train_value = train_df.values[:, 2:19]

    x_train = []
    y_train = []
    x_test = []
    x_file = []

    for i in tqdm(range(40479), miniters=1000):
        img = cv2.imread('data/train-jpg/' + train_name_list[i] + '.jpg')
        img_32 = cv2.resize(img, (32, 32))
        x_train.append(img_32)
        x_train.append(img_32[::-1])
        x_train.append(img_32[:, ::-1])
        x_train.append(img_32[:, ::-1][::-1])

        y_train.extend([y_train_value[i], y_train_value[i], y_train_value[i], y_train_value[i]])
        # y_train.extend([y_train_value[i]])

    for i in tqdm(range(40669), miniters=1000):
        img = cv2.imread('data/test-jpg/' + test_name_list[i] + '.jpg')
        x_test.append(cv2.resize(img, (32, 32)))

    for i in tqdm(range(20522), miniters=1000):
        img = cv2.imread('data/test-jpg-additional/' + test_name_list[40669 + i] + '.jpg')
        x_file.append(cv2.resize(img, (32, 32)))

    x_train = np.array(x_train, np.float16) / 255.
    y_train = np.array(y_train, np.int)
    x_test = np.array(x_test, np.float16) / 255.
    x_file = np.array(x_file, np.float16) / 255.

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(x_file.shape)

    # x_train, x_valid, y_train, y_valid = x_train[:35000], x_train[35000:], y_train[:35000], y_train[35000:]
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=4)

    # datagen = ImageDataGenerator(
    #     featurewise_center=False,  # set input mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #     samplewise_std_normalization=False,  # divide each input by its std
    #     zca_whitening=False,  # apply ZCA whitening
    #     rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    #     width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
    #     height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
    #     horizontal_flip=True,  # randomly flip images
    #     vertical_flip=False)  # randomly flip images

    # datagen.fit(x_train)

    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=(32, 32, 3)))
    #
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(17, activation='sigmoid'))

    model = load_model('model/5-17/199')

    model.summary()

    # model.compile(loss=losses.binary_crossentropy,
    #               # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
    #               optimizer=Adam(lr=0.001),  # adam 854931990138
    #               metrics=['accuracy'])

    # early_stopping = EarlyStopping(monitor='val_loss', patience=100)
    # reduce_lr_on_Plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=0, mode='auto',
    #                                          epsilon=0.0001, cooldown=0, min_lr=0)
    # model_save = ModelCheckpoint(filepath='model/5-17/{epoch}.h5', monitor='val_loss', verbose=0, save_best_only=False,
    #                              save_weights_only=False, mode='auto', period=10)
    #
    # hist = model.fit_generator(datagen.flow(x_train, y_train,
    #                                         batch_size=128),
    #                            steps_per_epoch=x_train.shape[0] // 128,
    #                            epochs=200,
    #                            verbose=2,
    #                            workers=4,
    #                            validation_data=(x_valid, y_valid),
    #                            callbacks=[early_stopping, reduce_lr_on_Plateau, model_save])

    # hist = model.fit(x_train, y_train,
    #           batch_size=128,
    #           epochs=300,
    #           verbose=1,
    #           validation_data=(x_valid, y_valid),
    #           callbacks=[early_stopping, reduce_lr_on_Plateau])
    # print(hist.history)
    # try:
    #     with open(
    #             'model/model_keras_cnn_data_4d_ImageDataGenerator_v2_optimizer_adam_drop_025_lr_reduce_epochs_200.json',
    #             'w') as f:
    #         json.dump(hist.history, f)
    # except:
    #     pass

    # score = model.evaluate(x_valid, y_valid, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    model.save('model/model_keras_cnn_data_4d_ImageDataGenerator_v2_optimizer_adam_drop_025_lr_reduce_epochs_199.h5')

    from sklearn.metrics import fbeta_score

    p_valid = model.predict(x_valid, batch_size=128)
    # np.savetxt('data/data_keras_cnn_data_origin_epochs_10_pred.csv', p_valid,
    #                delimiter=',', comments='', fmt='%.5f')

    dict_pred = {}
    for i in range(999):
        dict_pred[((i + 1) / 1000.0)] = fbeta_score(y_valid == 1, p_valid > ((i + 1) / 1000.0), beta=2,
                                                    average='samples')  # 随着模型的完善，0.2这个也可能需要改进，多选几个值输出F2；
    print(max(dict_pred, key=dict_pred.get))
    print(dict_pred[max(dict_pred, key=dict_pred.get)])

    p_test = model.predict(np.concatenate((x_test, x_file)), batch_size=128)
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
        preds.append(' '.join(pred_list))
    print(len(preds))
    # python2
    # index_preds = map(lambda x: "test_" + str(x), range(len(preds)))
    ## python3
    # index_preds = list(map(lambda x: "test_" + str(x), range(len(preds))))

    # print(len(index_preds))
    # print(len(preds))
    preds_data = np.c_[test_name_list, preds]
    print(preds_data.shape)
    np.savetxt(
        'submission/submission_keras_cnn_data_4d_ImageDataGenerator_v2_optimizer_adam_drop_025_lr_reduce_epochs_199.csv',
        preds_data,
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
    kears_cnn()
    # predict_load_model()
    # keras_mlp()
    # get_train_test_data()
    # get_train_list()


if __name__ == '__main__':
    main()

