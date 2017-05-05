# -*- coding: utf-8 -*-

# rainforest

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


def make_cooccurence_matrix(df_labels, labels):
    numeric_df = df_labels[labels]
    c_matrix = numeric_df.T.dot(numeric_df)
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
        img = cv2.imread('/users/wangqihui/Downloads/rainforest/train-jpg/train_'+str(i)+'.jpg')
        img = cv2.resize(img, (32, 32))
        cv2.imwrite('/users/wangqihui/Downloads/rainforest/train-jpg-32/train_'+str(i)+'.png', img)

    for i in tqdm(range(40669), miniters=1000):
        img = cv2.imread('/users/wangqihui/Downloads/rainforest/test-jpg/test_'+str(i)+'.jpg')
        img = cv2.resize(img, (32, 32))
        cv2.imwrite('/users/wangqihui/Downloads/rainforest/test-jpg-32/test_'+str(i)+'.png', img)


def kears_cnn():
    pass


def keras_mlp():
    pass


def predict_load_model():
    pass


def main():
    # test_submission_ver()
    # save_image()
    # kears_cnn()
    predict_load_model()
    # keras_mlp()


if __name__ == '__main__':
    main()

