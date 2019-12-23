import argparse
import cv2
import os
import torch

import os.path as osp
import numpy as np

# for COCO annotations
from pycocotools.coco import COCO
from PIL import Image

import time
import math
import json

import datetime
import re
import fnmatch
from natsort import natsorted
import random
import sys


def xywh2xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


# read in annotations
img_dir = '/home/ruichen/Documents/Documents_from_ubuntu_1604/Cityscapes_DATASET_and_MODEL/CityPersons_DATASET/leftImg8bit/train'
anno_dir = '/home/ruichen/Documents/Cityscapes_COCO_Annotations/annotations_coco_format'
save_dir = '/home/ruichen/Documents/Cityscapes_COCO_Annotations/crop_train'

anno_path = osp.join(anno_dir, 'instancesonly_filtered_gtFine_train.json')
coco = COCO(anno_path)
ids = list(coco.imgs.keys())
length = len(ids)

for index in range(length):
    print(index)
    img_id = ids[index]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    target = coco.loadAnns(ann_ids)

    path = coco.loadImgs(img_id)[0]['file_name']
    sub_folder = path.split('_')[0]
    img = Image.open(osp.join(img_dir, sub_folder, path)).convert('RGB')

    if not osp.exists(osp.join(save_dir, sub_folder)):
        os.mkdir(osp.join(save_dir, sub_folder))

    if not osp.exists(osp.join(save_dir, sub_folder, path.split('.')[0])):
        os.mkdir(osp.join(save_dir, sub_folder, path.split('.')[0]))

    boxes = [obj["bbox"] for obj in target]
    # print(len(boxes))
    for idx, box in enumerate(boxes):
        # print(box)
        box = xywh2xyxy(box)
        # print(box)
        crop_area = img.crop((box[0], box[1], box[2], box[3]))
        crop_area.save(osp.join(save_dir, sub_folder, path.split('.')[0], path.split('.')[0] + '_{}.jpg'.format(idx)))
    # boxes = torch.as_tensor(boxes).reshape(-1, 4)



# crop the image patch


# re-write the annotation


