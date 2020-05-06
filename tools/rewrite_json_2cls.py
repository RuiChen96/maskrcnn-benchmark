import os
import argparse
import cv2
import json

import os.path as osp
import numpy as np

from tqdm import tqdm
from pycocotools.coco import COCO


data_dir = '/Users/ruichen/Data/test_data_31-100'
file_name = 'ann_test_31_100_cam_100.json'

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
