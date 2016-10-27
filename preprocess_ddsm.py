"""
preprocess_ddsm.py

@author: wronk

Preprocess (resize, crop, normalize) BC test data and save
"""

import os
from os import path as op
import cv2
import json

###################
# Set globals
###################
save_data = True
fold_save = 'preprocessed'

data_dir = '/media/Toshiba/Code/ddsm_data/'
save_dir = '/media/Toshiba/Code/ddsm_data_preproc/'
base_str_classes = ['normal', 'benign_without_callback', 'benign', 'cancer']
n_base_classes = [12, 2, 14, 15]

case_base_str = 'case'
base_str_img = ['LEFT_CC.png', 'LEFT_MLO.png',
                'RIGHT_CC.png', 'RIGHT_MLO.png']
base_views = ['LEFT_CC', 'LEFT_MLO', 'RIGHT_CC', 'RIGHT_MLO']
new_img_size = (400, 800)


skip_cases = ['case3102']
do_flip_images = False
do_get_pathology = True

pathology_dict = dict(base_views=base_views, skip_cases=skip_cases)

def check_dir(path):
    if not op.isdir(path):
        os.mkdir(path)


####################
# Preprocess images
####################
def check_for_abnormality(case_dir, filename):
    """Helper to load .OVERLAYs and get doctor's pathology classification"""

    with open(op.join(case_dir, filename), 'r') as overlay_file:
        # Loop over overlay lines
        for line in overlay_file:
            if 'PATHOLOGY' in line:
                # Check if "Pathology" in line
                if 'MALIGNANT' in line:
                    return True
        return False

    raise RuntimeError('Missing pathology for %s' % case_dir)

#TODO: Once working, convert to functions/reduce for loops

for class_fold, n_base_class in zip(base_str_classes, n_base_classes):
    batch_folds = [class_fold + '_%02i' % batch_num
                   for batch_num in range(1, n_base_class + 1)]

    print 'Processing %s' % (class_fold)
    pathology_dict[class_fold] = {}
    for batch_fold in batch_folds:
        # Get inidividual case directory names
        batch_dir = op.join(data_dir, batch_fold)
        case_list = sorted([c for c in os.listdir(batch_dir)
                            if op.isdir(op.join(batch_dir, c)) and
                            case_base_str in c])

        check_dir(op.join(save_dir, batch_fold))
        pathology_dict[class_fold][batch_fold] = {}

        # Loop through each individual case
        for ci, case in enumerate(case_list):
            print '  Processing case %i/%i, %s' % (ci + 1, len(case_list),
                                                   case)
            if case in skip_cases:
                print '  Skipping case %s' % case
                continue

            case_dir = op.join(batch_dir, case)

            img_dir_files = os.listdir(op.join(batch_dir, case, 'PNGFiles'))
            img_list = [img for img in img_dir_files if '.png' in img]

            check_dir(op.join(save_dir, batch_fold, case))

            if do_flip_images:
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

            #################################
            # Process potential overlay files
            #################################
            if do_get_pathology:
                # Get existing overlay files (which indicate abnormalities)
                overlay_list = [tmp for tmp in os.listdir(case_dir)
                                if '.OVERLAY' in tmp]
                path_4 = [False] * 4

                # For each overlay file, check for 'malignant' diagnosis
                for overlay in overlay_list:
                    temp_pathology = check_for_abnormality(case_dir, overlay)

                    ind = base_views.index(overlay.split('.')[1])
                    path_4[ind] = temp_pathology
                if 'cancer' in batch_fold and not any(path_4):
                    raise RuntimeError('Cancer folder, but no malignancy')
                # Store prognosis for each image of each case
                pathology_dict[class_fold][batch_fold][case] = path_4

if save_data:
    fname_save = op.join(save_dir, 'pathology_labels.json')
    print 'Output saved to %s' % fname_save
    with open(fname_save, 'w') as json_file:
        json.dump(pathology_dict, json_file, indent=2)
