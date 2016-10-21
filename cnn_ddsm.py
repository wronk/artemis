"""
cnn_ddsm.py

@author: wronk

Classify breast cancer images using convolutional neural network.
"""

import os
from os import path as op
import tensorflow
import numpy as np
import cv2

data_dir = '/media/Toshiba/Code/ddsm_data_preproc/'
diagnosis_classes = ['normal', 'benign', 'cancer']
n_base_classes = [1, 1, 1]
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

                cases_arr[ci, img_i, :, :] = img
                #import ipdb; ipdb.set_trace()

        batch_data.append(cases_arr)

    diagnosis_data[diag_class] = batch_data
