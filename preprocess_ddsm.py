"""
preprocess_ddsm.py

@author: wronk

Preprocess (resize, crop, normalize) BC test data and save
"""

import os
from os import path as op
import cv2

###################
# Set globals
###################
save_data = True
fold_save = 'preprocessed'

data_dir = '/media/Toshiba/Code/ddsm_data/'
save_dir = '/media/Toshiba/Code/ddsm_data_preproc/'
base_str_classes = ['normal', 'benign', 'cancer']
n_base_classes = [12, 14, 15]
#base_str_classes = ['normal']
#n_base_classes = [1]
case_base_str = 'case'
base_str_img = ['LEFT_CC.png', 'LEFT_MLO.png',
                'RIGHT_CC.png', 'RIGHT_MLO.png']
new_img_size = (400, 800)


def check_dir(path):
    if not op.isdir(path):
        os.mkdir(path)

####################
# Preprocess images
####################
#TODO: Once working, convert to functions/reduce for loops

for class_fold, n_base_class in zip(base_str_classes, n_base_classes):
    batch_folds = [class_fold + '_%02i' % batch_num
                   for batch_num in range(1, n_base_class + 1)]

    for batch_fold in batch_folds:
        # Get inidividual case directory names
        batch_dir = op.join(data_dir, batch_fold)
        case_list = [c for c in os.listdir(batch_dir)
                     if op.isdir(op.join(batch_dir, c)) and case_base_str in c]

        check_dir(op.join(save_dir, batch_fold))

        # Loop through each individual case
        # TODO: change to use base_str_img
        for ci, case in enumerate(case_list):
            print 'Processing case %i/%i, %s' % (ci + 1, len(case_list), case)
            case_dir = op.join(batch_dir, case)
            img_dir = os.listdir(op.join(batch_dir, case, 'PNGFiles'))
            img_list = [img for img in img_dir if '.png' in img]

            check_dir(op.join(save_dir, batch_fold, case))

            # Load and resize each image
            for filepath_img in img_list:
                filepath_img = op.join(case_dir, 'PNGFiles', filepath_img)
                img = cv2.imread(filepath_img, 0)
                img_resized = cv2.resize(img, new_img_size,
                                         interpolation=cv2.INTER_AREA)

                img_file_name, ext = op.splitext(op.basename(filepath_img))
                fname_save = op.join(save_dir, batch_fold, case,
                                     img_file_name + '_preproc' + ext)

                if save_data:

                    cv2.imwrite(fname_save, img_resized)
                else:
                    print 'Would have saved to: ' + fname_save
