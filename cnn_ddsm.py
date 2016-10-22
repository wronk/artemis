"""
cnn_ddsm.py

@author: wronk

Classify breast cancer images using convolutional neural network.
"""

import os
from os import path as op
import tensorflow as tf
import numpy as np
import cv2

data_dir = '/media/Toshiba/Code/ddsm_data_preproc/'
diagnosis_classes = ['normal', 'benign', 'cancer']
n_base_classes = [2, 2, 2]
#n_base_classes = [12, 14, 15]
case_base_str = 'case'
base_str_img = ['LEFT_CC.png', 'LEFT_MLO.png',
                'RIGHT_CC.png', 'RIGHT_MLO.png']
IMG_SIZE = (800, 400)

diagnosis_data = {}
for di, diag_class in enumerate(diagnosis_classes):
    print 'Loading diagnosis class: ' + diag_class
    batch_folds = [diag_class + '_%02i' % batch_num
                   for batch_num in range(1, n_base_classes[di] + 1)]

    batch_data = []
    for batch_fold in batch_folds:
        print '  Loading batch: ' + batch_fold
        # Get inidividual case directory names
        batch_dir = op.join(data_dir, batch_fold)
        case_list = [c for c in os.listdir(batch_dir)
                     if op.isdir(op.join(batch_dir, c)) and case_base_str in c]
        case_list.sort()

        cases_arr = -1 * np.ones((len(case_list), 4, IMG_SIZE[0], IMG_SIZE[1]))

        # Loop through each individual case
        for ci, case_fold in enumerate(case_list):
            img_dir = op.join(batch_dir, case_fold)
            img_list = [temp_img_fname for temp_img_fname in os.listdir(img_dir)
                        if '.png' in temp_img_fname]

            # Load and store each image
            for img_i, img_fname in enumerate(img_list):
                img_fpath = op.join(img_dir, img_fname)
                img = cv2.imread(img_fpath, 0)

                if 'RIGHT' in img_fname:
                    img = cv2.flip(img, flipCode=1)

                cases_arr[ci, img_i, :, :] = img

        batch_data.append(cases_arr)

    diagnosis_data[diag_class] = batch_data

#####################
# CNN helper funks
#####################


def weight_variable(shape, name):
    return(tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name))


def bias_variable(shape, name):
    #return(tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name))
    return(tf.Variable(tf.constant(0.1, shape=shape), name=name))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, pool_size):
    return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1],
                          strides=[1, pool_size, pool_size, 1], padding='SAME')


def create_ccn_model(layers_sizes, fullC_size, pool_size, filt_size, act_func, img_size):
    W_list, b_list, h_list = [], [], []

    x_train = tf.placeholder(tf.float32, shape=[None, img_size[0] * image_size[1]],
                             name='x_train')
    x_image = tf.reshape(x_train, [-1, image_size[0], image_size[1], 1])

    for li, l_size in enumerate(layer_sizes[:-1]):
        # Add convolutional layer
        W_list.append(weight_variable([filt_size, filt_size, l_size,
                                       l_size[li + 1]], name='W%i' % li))
        b_list.append(bias_variable([l_size[li + 1]], name='b%i' % li))

        if li == 0:
            conv_temp = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        else:
            conv_temp = tf.nn.relu(conv2d(h_list[-1], W_list[-1]) + b_list[-1])

        h_list.append(max_pool(conv_temp))

    # First fully connected layer
    # Input: W_fc1 is image size x num_features
    n_pool_layers = float(len(cnn_layer_sizes))
    n_fc_vars = int(np.ceil(image_size[0] / pool_size ** n_pool_layers) *
                    np.ceil(image_size[1] / pool_size ** n_pool_layers) *
                    cnn_layer_sizes[-1])

    W_fc1 = weight_variable([n_fc_vars, fullC_layer_size], name='W_fc1')
    b_fc1 = bias_variable([fullC_layer_size], name='b_fc1')

    #h_pool2_flat = tf.reshape(h_pool4, [-1, n_fc_vars])
    h_pool2_flat = tf.reshape(h_pool2, [-1, n_fc_vars])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Apply dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Second fully connected layer (includes output layer)
    W_fc2 = weight_variable([fullC_layer_size, n_classes], name='W_fc2')
    b_fc2 = bias_variable([n_classes], name='b_fc2')

    y_out = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    #y_out = tf.matmul(h_fc1, W_fc2) + b_fc2

    return x_train, y_out, keep_prob


#####################
# CNN model params
#####################

layers_size = [64, 64, 12]
fullC_size = 32
act_func = tf.nn.relu
pool_size = 2
filt_size = 2
dropout_keep_p = 0.5

# Training params
training_prop = 0.75
batch_size = 30
n_training_batches = 3000

######################
# Construct CNN
######################
layers_size.insert(0, 1)  # Add for convenience
