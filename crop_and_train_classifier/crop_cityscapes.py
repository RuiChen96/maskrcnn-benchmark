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


INFO = {
    "description": "Cityscapes Cropped Image Patches for Image Classification",
    "url": "",
    "version": "0.1.0",
    "year": 2019,
    "contributor": "Rui Chen",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

# read in annotations
img_dir = '/home/ruichen/Documents/Documents_from_ubuntu_1604/Cityscapes_DATASET_and_MODEL/CityPersons_DATASET/leftImg8bit/train'
anno_dir = '/home/ruichen/Documents/Cityscapes_COCO_Annotations/annotations_coco_format'
save_dir = '/home/ruichen/Documents/Cityscapes_COCO_Annotations/crop_train'

anno_path = osp.join(anno_dir, 'instancesonly_filtered_gtFine_train.json')
coco = COCO(anno_path)
ids = list(coco.imgs.keys())
length = len(ids)

dataset = json.load(open(anno_path, 'r'))
CATEGORIES = dataset['categories']
coco_output_train = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
}
annotations_train = []

img_id_save = 0

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

    # bbox locations for cropping
    boxes = [obj["bbox"] for obj in target]
    # bbox labels for re-writing annotations
    labels = [obj["category_id"] for obj in target]

    for idx, box in enumerate(boxes):

        box = xywh2xyxy(box)

        crop_area = img.crop((box[0], box[1], box[2], box[3]))
        save_path = osp.join(save_dir, sub_folder, path.split('.')[0], path.split('.')[0] + '_{}.jpg'.format(idx))
        crop_area.save(save_path)

        file_name = path.split('.')[0] + '_{}.jpg'.format(idx)
        width = box[2] - box[0]
        height = box[3] - box[1]

        image_info = {'id': img_id_save, 'file_name': file_name, 'width': width,
                      'height': height, 'licence': 1, 'coco_url': ''}
        coco_output_train["images"].append(image_info)

        annotation = {"image_id": img_id_save, "iscrowd": 0, "area": int(width * height),
                      "bbox": [], "segmentation": [], "id": [], "category_id": labels[idx]}
        annotations_train.append(annotation)

        img_id_save += 1

coco_output_train["annotations"] = annotations_train

with open('{}/annotations_cityscapes_cropped_patches.json'.format(save_dir), 'w') as output_json_file:
    json.dump(coco_output_train, output_json_file)

print('--- --- --- END --- --- ---')

# crop the image patch -- done


# re-write the annotation -- save it using COCO format -- TODO


# prepare the negative example

