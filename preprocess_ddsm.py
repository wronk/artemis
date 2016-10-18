"""
preprocess_bc_test.py

@author: wronk

Preprocess (resize, crop, normalize) BC test data and save
"""

import os
from os import path as op
#import cv2


###################
# Set globals
###################
save_data = True
fold_save = 'preprocessed'

data_dir = './test_data/'
base_str_classes = ['normal', 'benign', 'cancer']
case_base_str = 'case_'
base_str_img = ['*LCC.png', '*LMLO.png',
                '*RCC.png', '*RMLO.png']
new_img_size = [400, 300]

####################
# Preprocess images
####################

for class_fold in base_str_classes:
    batch_dir = op.join(data_dir, class_fold)

    # Get inidividual case directory names
    case_list = [c for c in os.listdir(batch_dir)
                 if op.isdir(c) and case_base_str in c]

    # Loop through each individual case
    # TODO: change to use base_str_img
    for case in case_list:
        img_list = [img for img in op.isdir(case)
                    if '.JPG' in img]

        for fname_img in img_list:
            img = cv2.imread(fname_img)
            img_resized = cv2.resize(img, None, new_img_size[0], new_img_size[1],
                                     interpolation='INTER_AREA')

            if save_data:
                fname, fext = op.splitext(fname_img.split)
                fname_save = fname + '_preproc' + fext

                cv2.imwrite(fname_save, img_resize)
