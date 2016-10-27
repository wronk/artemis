"""
cnn_ddsm.py

@author: wronk

Classify breast cancer images using convolutional neural network.
"""

import os
from os import path as op
import tensorflow as tf
import numpy as np
from numpy.random import choice
import cv2
import json

data_dir = os.environ['DDSM_DATA']
diagnosis_classes = ['normal', 'benign', 'cancer']
diagnosis_batch_size = [20, 0, 20]  # Number of samples per batch

n_base_classes = [1, 1, 1]
#n_base_classes = [12, 14, 15]
data_split_props = [0.6, 0.2, 0.2]  # Training, validation, test
case_base_str = 'case'
base_str_img = ['LEFT_CC.png', 'LEFT_MLO.png',
                'RIGHT_CC.png', 'RIGHT_MLO.png']
IMG_SIZE = (800, 400)  # n_rows x n_cols

assert sum(data_split_props) == 1., "Data proportions must add to 1."

# Load image labels (y/n malignant)
with open(op.join(data_dir, 'pathology_labels.json')) as json_file:
    loaded_diag_dict = json.load(json_file)
base_views = [name + '_preproc' for name in loaded_diag_dict['base_views']]

diagnosis_data = {}
diagnosis_labels = {}
for di, diag_class in enumerate(diagnosis_classes):
    print '\nLoading diagnosis class: ' + diag_class
    batch_folds = [diag_class + '_%02i' % batch_num
                   for batch_num in range(1, n_base_classes[di] + 1)]

    batch_data = []
    batch_label = []
    for batch_fold in batch_folds:
        print '  Loading batch: ' + batch_fold
        # Get inidividual case directory names
        batch_dir = op.join(data_dir, batch_fold)
        case_list = [c for c in os.listdir(batch_dir)
                     if op.isdir(op.join(batch_dir, c)) and case_base_str in c]
        case_list.sort()

        cases_arr = -1 * np.ones((len(case_list), 4, IMG_SIZE[0], IMG_SIZE[1]))
        labels_arr = -1 * np.ones((len(case_list), 4))

        # Loop through each individual case
        for ci, case_fold in enumerate(case_list):
            if case_fold in loaded_diag_dict['skip_cases']:
                print 'Skipping case %s' % case_fold
                continue

            img_dir = op.join(batch_dir, case_fold)
            img_list = [temp_img_fname for temp_img_fname in os.listdir(img_dir)
                        if '.png' in temp_img_fname]
            case_labels = loaded_diag_dict[diag_class][batch_fold][case_fold]

            for img_i, img_fname in enumerate(img_list):
                # Load and store each image
                img_fpath = op.join(img_dir, img_fname)
                img = cv2.imread(img_fpath, 0)

                img_base_view = img_fname.split('.')[-2]
                img_4_index = base_views.index(img_base_view)

                # Load and store labels
                # TODO: Check ordering of images/labels
                cases_arr[ci, img_i, :, :] = img
                labels_arr[ci, img_4_index] = int(case_labels[img_4_index])

        batch_data.extend(cases_arr)
        batch_label.extend(labels_arr)

    diagnosis_data[diag_class] = np.asarray(batch_data)
    diagnosis_labels[diag_class] = np.asarray(batch_label)

####################################
# Construct validation and test data
####################################


def split_data(data, splits):
    """Helper to split large dataset into an arbitrary number of proportions"""
    n_imgs = data.shape[0]
    rand_inds = choice(range(n_imgs), size=n_imgs, replace=False)

    data_arrs = []
    split_inds = [int(sum(splits[0:ind]) * n_imgs) for ind in range(len(splits))]
    split_inds.append(n_imgs)

    for si in range(len(splits)):
        start, stop = split_inds[si], split_inds[si + 1]
        rand_slice = rand_inds[start:stop]

        data_arrs.append(data[rand_slice])

    return data_arrs


train_data, valid_data, test_data = {}, {}, {}
for diag_class in diagnosis_classes:
    train_data[diag_class], valid_data[diag_class], test_data[diag_class] = \
        split_data(diagnosis_data[diag_class], data_split_props)
del diagnosis_data

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


def create_ccn_model(layer_sizes, fullC_size, pool_size, filt_size, act_func,
                     img_size, n_classes):
    W_list, b_list, h_list = [], [], []

    # Initialize input vars
    #x_train = tf.placeholder(tf.float32, shape=[None, img_size[0] * img_size[1]],
    #                         name='x_train')
    #x_image = tf.reshape(x_train, [-1, img_size[0], img_size[1], 1])
    x_image = tf.placeholder(tf.float32, shape=[None, img_size[0], img_size[1], 1])

    # Add convolutional layers one by one
    for li, (l_size1, l_size2) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        W_list.append(weight_variable([filt_size, filt_size, l_size1,
                                       l_size2], name='W%i' % li))
        b_list.append(bias_variable([l_size2], name='b%i' % li))

        if li == 0:
            conv_temp = tf.nn.relu(conv2d(x_image, W_list[-1]) + b_list[-1])
        else:
            conv_temp = tf.nn.relu(conv2d(h_list[-1], W_list[-1]) + b_list[-1])

        h_list.append(max_pool(conv_temp, pool_size))

    # First fully connected layer
    # Input: W_fc1 is image size x num_features
    n_pool_layers = float(len(layer_sizes) - 1)
    last_h_shape = [dim.value for dim in h_list[-1].get_shape()[1:]]
    n_fc_vars = int(np.prod(last_h_shape))

    W_fc1 = weight_variable([n_fc_vars, fullC_size], name='W_fc1')
    b_fc1 = bias_variable([fullC_size], name='b_fc1')

    h_pool_last_flat = tf.reshape(h_list[-1], [-1, n_fc_vars])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_last_flat, W_fc1) + b_fc1)

    # Apply dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Second fully connected layer (output layer)
    W_fc2 = weight_variable([fullC_size, n_classes], name='W_fc2')
    b_fc2 = bias_variable([n_classes], name='b_fc2')

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    #return x_train, y_conv, keep_prob
    return x_image, y_conv, keep_prob


def get_batch(data):
    """Helper to randomly return samples from dataset"""
    batch_x, batch_y = [], []

    # Get a designated number of samples from each diagnosis class
    for di, diag_class in enumerate(diagnosis_classes):
        rand_inds = choice(range(data[diag_class].shape[0]),
                           size=diagnosis_batch_size[di], replace=False)
        batch_x.extend(data[diag_class][rand_inds])
        batch_y.extend(diagnosis_labels[diag_class][rand_inds])

    # Reshape for feeding tensorflow. Each row is now an observation
    batch_x_arr = np.array(batch_x).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)
    batch_y_arr = np.array(batch_y).reshape(-1)

    return batch_x_arr, batch_y_arr

#####################
# CNN model params
#####################

layers_sizes = [64, 64, 12]
fullC_size = 32
act_func = tf.nn.relu
pool_size = 2
filt_size = 2
dropout_keep_p = 0.5
n_classes = 2

# Training params
training_prop = 0.75
n_training_batches = 3000

######################
# Construct CNN
######################
layers_sizes.insert(0, 1)  # Add for convenience during construction

with tf.device('/gpu:0'):
    x_train, y_conv, keep_prob = create_ccn_model(
        layers_sizes, fullC_size, pool_size, filt_size, act_func, IMG_SIZE,
        n_classes)
    y_labels = tf.placeholder(tf.int64, shape=[None], name='y_labels')

    # Add objective function and defining training scheme
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        y_conv, y_labels))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
    is_pred_correct = tf.equal(tf.arg_max(y_conv, 1), y_labels)
    accuracy = tf.reduce_mean(tf.cast(is_pred_correct, tf.float32))

# Attach summaries
tf.scalar_summary('loss', loss)
tf.scalar_summary('accuracy', accuracy)
merged_summaries = tf.merge_all_summaries()

#saver = tf.train.Saver()  # create saver for saving network weights
init = tf.initialize_all_variables()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

train_writer = tf.train.SummaryWriter('./train_summaries', sess.graph)
sess.run(init)

######################
# Train CNN
######################

for ti in range(n_training_batches):
    # Get data for training step
    batch_x, batch_y = get_batch(train_data)
    feed_dict = {x_train: batch_x, y_labels: batch_y,
                 keep_prob: dropout_keep_p}

    _, obj, acc, summary = sess.run([train_step, loss, accuracy,
                                     merged_summaries], feed_dict)
    train_writer.add_summary(summary, ind)
    print("\titer: %03d, cost: %.2f, acc: %.2f" % (ti, obj, acc))

    # Sometimes compute validation accuracy
    if ti % 5 == 0:
        valid_acc = accuracy.eval(feed_dict={x_train: valid_x, y_labels:
                                             valid_y, keep_prob: 1.0})
        print 'Validation accuracy: %0.2f' % test_acc

# Compute test data accuracy
test_acc = accuracy.eval(feed_dict={x_train: test_x, y_labels: test_y,
                                    keep_prob: 1.0})
print 'Test accuracy: %0.2f' % test_acc
sess.close()
