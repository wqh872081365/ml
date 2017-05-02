# -*- coding: utf-8 -*-

# rainforest

# import matplotlib
# matplotlib.use('TkAgg')

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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
from skimage import io
from scipy import ndimage
# from IPython.display import display
# %matplotlib inline


## data init

# labels_df = pd.read_csv("data/train.csv")
# print(labels_df.head())

# Build list with unique labels
# label_list = []
# for tag_str in labels_df.tags.values:
#     labels = tag_str.split(' ')
#     for label in labels:
#         if label not in label_list:
#             label_list.append(label)

# datas_df = pd.read_csv("data/sample_submission.csv")

# Add onehot features for every label
# for label in label_list:
#     datas_df[label] = datas_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
    # datas_df[label] = datas_df[label].astype(int)
# Display head
# print(labels_df.head())
# datas_df.to_csv("data/test_list.csv", index=False)


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


## image


def sample_images(tags, n=None):
    """Randomly sample n images with the specified tags."""
    condition = True
    if isinstance(tags, string_types):
        raise ValueError("Pass a list of tags, not a single tag.")
    for tag in tags:
        condition = condition & train_df[tag] == 1
    if n is not None:
        return train_df[condition].loc[10]# .sample(n)
    else:
        return train_df[condition]


def load_image(filename):
    '''Look through the directory tree to find the image you specified
    (e.g. train_10.tif vs. train_10.jpg)'''
    for dirname in os.listdir('data'):
        path = os.path.abspath(os.path.join('data', 'train-tif-sample', filename))
        if os.path.exists(path):
            print('Found image {}'.format(path))
            return io.imread(path)
    # if you reach this line, you didn't find the image you're looking for
    print('Load failed: could not find image {}'.format(filename))


def sample_to_fname(sample_df, row_idx, suffix='tif'):
    '''Given a dataframe of sampled images, get the
    corresponding filename.'''
    fname = sample_df.get_value(sample_df.index[row_idx], 'image_name')
    return '{}.{}'.format(fname, suffix)


def plot_rgbn_histo(r, g, b, nir, rgbn_image):
    for slice_, name, color in ((r,'r', 'red'),(g,'g', 'green'),(b,'b', 'blue'), (nir, 'nir', 'magenta')):
        plt.hist(slice_.ravel(), bins=100,
                 range=[0,rgbn_image.max()],
                 label=name, color=color, histtype='step')
    plt.legend()


s = sample_images(['primary', 'agriculture', 'clear', 'slash_burn', 'water'], n=1)
fname = sample_to_fname(s, 0)

# find the image in the data directory and load it
rgbn_image = load_image(fname)

# extract the rgb values
rgb_image = rgbn_image[:,:,:3]

# # extract the different bands
# r, g, b, nir = rgbn_image[:, :, 0], rgbn_image[:, :, 1], rgbn_image[:, :, 2], rgbn_image[:, :, 3]

# plot a histogram of rgbn values
# plot_rgbn_histo(r, g, b, nir, rgbn_image)


# Plot the bands
# fig = plt.figure()
# fig.set_size_inches(12, 4)
# for i, (x, c) in enumerate(((r, 'r'), (g, 'g'), (b, 'b'), (nir, 'near-ir'))):
#     a = fig.add_subplot(1, 4, i+1)
#     a.set_title(c)
#     plt.imshow(x)

# plt.imshow(rgb_image)

# plt.show()



# Pull a list of 100 image names
jpg_list = os.listdir('data/train-jpg-sample')
# Select a random sample of 100 among those
# np.random.shuffle(jpg_list)
# jpg_list = jpg_list[:100]


ref_colors = [[], [], []]
for _file in jpg_list:
    # keep only the first 3 bands, RGB
    _img = mpimg.imread(os.path.join('data/train-jpg-sample', _file))[:, :, :3]
    # Flatten 2-D to 1-D
    _data = _img.reshape((-1, 3))
    # Dump pixel values to aggregation buckets
    for i in range(3):
        ref_colors[i] = ref_colors[i] + _data[:, i].tolist()

ref_colors = np.array(ref_colors)


# for i,color in enumerate(['r','g','b']):
#     plt.hist(ref_colors[i], bins=30, range=[0,255], label=color, color=color, histtype='step')
# plt.legend()
# plt.title('Reference color histograms')

# plt.show()


ref_means = [np.mean(ref_colors[i]) for i in range(3)]
ref_stds = [np.std(ref_colors[i]) for i in range(3)]


def calibrate_image(rgb_image):
    # Transform test image to 32-bit floats to avoid
    # surprises when doing arithmetic with it
    calibrated_img = rgb_image.copy().astype('float32')

    # Loop over RGB
    for i in range(3):
        # Subtract mean
        calibrated_img[:,:,i] = calibrated_img[:,:,i]-np.mean(calibrated_img[:,:,i])
        # Normalize variance
        calibrated_img[:,:,i] = calibrated_img[:,:,i]/np.std(calibrated_img[:,:,i])
        # Scale to reference
        calibrated_img[:,:,i] = calibrated_img[:,:,i]*ref_stds[i] + ref_means[i]
        # Clip any values going out of the valid range
        calibrated_img[:,:,i] = np.clip(calibrated_img[:,:,i],0,255)

    # Convert to 8-bit unsigned int
    return calibrated_img.astype('uint8')


test_image_calibrated = calibrate_image(rgb_image)
# for i,color in enumerate(['r','g','b']):
#     plt.hist(test_image_calibrated[:,:,i].ravel(), bins=30, range=[0,255],
#              label=color, color=color, histtype='step')
# plt.legend()
# plt.title('Calibrated image color histograms')

# plt.imshow(test_image_calibrated)
io.imsave('test.tif', test_image_calibrated)
plt.show()


