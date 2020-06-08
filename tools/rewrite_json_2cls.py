import os
import argparse
import cv2
import json

import os.path as osp
import numpy as np

from tqdm import tqdm
from pycocotools.coco import COCO


# data_dir = '/Users/ruichen/Data/shanghai_store_img'
data_dir = '/Users/ruichen/Data/real_data_anno_dir/filtered_version'
# file_name = 'annotations_amcrest_skus1-30-hcrop_crop_450.json'
# file_name = 'annotations_combined_1-30_200classes_v2_filtered.json'
# file_name = 'annotations_31_100_200classes_v6_filtered.json'
file_name = 'annotations_skus_101_200_train_v4_filtered.json'

file_path = os.path.join(data_dir, file_name)

with open(file_path, mode='r') as fp:
    ann = json.load(fp)

print('Loading annotations ...')

for key, val in enumerate(ann['annotations']):
    ann['annotations'][key]['category_id'] = 1

two_cls_file_path = os.path.join(data_dir, file_name.split('.')[0] + '_2_cls.json')

with open(two_cls_file_path, mode='w') as fp:
    json.dump(ann, fp)

print('Rewriting annotations ...')
