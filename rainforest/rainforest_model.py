# -*- coding: utf-8 -*-

# rainforest_image

import matplotlib
matplotlib.use('TkAgg')

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import tensorflow as tf


import sys
import os
import subprocess

from six import string_types

# Make sure you have all of these packages installed, e.g. via pip
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
from skimage import io
from scipy import ndimage
# from IPython.display import display
# %matplotlib inline


## keras
# keras cnn init epochs=5 ->
# keras cnn init epochs=50 ->
# keras mlp init epochs=5 ->


## data init

# labels_df = pd.read_csv("data/train.csv")
# # print(labels_df.head())
# #
# # # Build list with unique labels
# label_list = []
# for tag_str in labels_df.tags.values:
#     labels = tag_str.split(' ')
#     for label in labels:
#         if label not in label_list:
#             label_list.append(label)
#
# # datas_df = pd.read_csv("data/sample_submission.csv")
#
# # Add onehot features for every label
# for label in label_list:
#     labels_df[label] = labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
# # # Display head
# # # print(labels_df.head())
# # labels_df.to_csv("data/train_list.csv", index=False)
# train_df = labels_df
#
test_df = pd.read_csv("data/sample_submission.csv")
# test_df['tags'] = test_df['tags'].apply(lambda x: '')

# start

# min:98, max 37840, all=17

train_df = pd.read_csv("data/train_list.csv")
label_list = train_df.columns.values[2:]

# Histogram of label instances
# train_list_values = train_df[label_list].sum().sort_values()# .plot(kind="bar")
# plt.show()
# print(train_list_values)


def make_cooccurence_matrix(labels):
    numeric_df = train_df[labels]
    c_matrix = numeric_df.T.dot(numeric_df)
    sns.heatmap(c_matrix, xticklabels=True, yticklabels=True)
    return c_matrix


# Compute the co-ocurrence matrix
# c_matrix = make_cooccurence_matrix(label_list)
# fig, ax = plt.subplots()
# a=ax.get_xticks()
# ax.set_xticklabels(a, rotation=40)

# weather_labels = ['clear', 'partly_cloudy', 'haze', 'cloudy']
# make_cooccurence_matrix(weather_labels)

# land_labels = ['primary', 'agriculture', 'water', 'cultivation', 'habitation', 'road']
# make_cooccurence_matrix(land_labels)
# train_df['land'] = train_df['tags'].apply(lambda x: 1 if ('road' in x.split(' ') or 'primary' in x.split(' ') or 'agriculture' in x.split(' ') or 'water' in x.split(' ') or 'cultivation' in x.split(' ') or 'habitation' in x.split(' ')) else 0)
# print(train_df['land'].sum())

# rare_labels = [l for l in label_list if train_df[label_list].sum()[l] < 1000]
# make_cooccurence_matrix(rare_labels)
#
# plt.xticks(rotation=90)
# plt.yticks(rotation=0)
# plt.show()



# single label

# print(train_df[train_df["tags"] == label_list[9]])

# keras

# mlp

# cnn


import gc

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import cv2
from tqdm import tqdm

x_train = []
x_test = []
# y_train = []

# df_train = pd.read_csv('data/train.csv')

# flatten = lambda l: [item for sublist in l for item in sublist]
# labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

# label_map = {l: i for i, l in enumerate(labels)}
# inv_label_map = {i: l for l, i in label_map.items()}

for f in tqdm(train_df.values[:2], miniters=1000):
    img = cv2.imread('data/train-jpg-sample/{}.jpg'.format(f[0]))
    # targets = np.zeros(17)
    # for t in tags.split(' '):
        # targets[label_map[t]] = 1
    x_train.append(cv2.resize(img, (32, 32)))
    # y_train.append(targets)

for f in tqdm(test_df.values[:1], miniters=1000):
    img = cv2.imread('data/test-jpg-sample/{}.jpg'.format(f[0]))
    # targets = np.zeros(17)
    # for t in tags.split(' '):
        # targets[label_map[t]] = 1
    x_test.append(cv2.resize(img, (32, 32)))
    # y_train.append(targets)

x_train = np.array(x_train, np.float16) / 255.
x_test = np.array(x_test, np.float16) / 255.

# x_train /= 255
# x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
# x_test /= 255
# x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
y_train = train_df.values[0:2, 2:19]

print(x_train.shape)
print(y_train.shape)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

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

# model.save('model_keras_cnn_epochs_5.h5')

# from sklearn.metrics import fbeta_score
# p_valid = model.predict(x_valid, batch_size=128)
# print(fbeta_score(y_valid, p_valid > 0.5, beta=2, average='macro'))

p_test = model.predict(x_test, batch_size=128)
preds = []
for i in range(p_test.shape[0]):
    preds.append(' '.join([label_list[j] for j in range(len(label_list)) if p_test[i, j]>0.5]))

np.savetxt('submission_keras_cnn_epochs_5.csv', np.c_[map(lambda x: "test_" + str(x), range(x_test.shape[0])), preds],
               delimiter=',', header='image_name,tags', comments='', fmt='%s')
