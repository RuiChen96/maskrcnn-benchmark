import os
import argparse
import cv2

import os.path as osp
import numpy as np

from tqdm import tqdm
from pycocotools.coco import COCO


def scoring(item):
    return item['score']


real_annotation_dir = '/Users/ruichen/train_real_pilot1_photobox'
pseudo_label_dir = '/Users/ruichen/'

real_path = osp.join(real_annotation_dir, 'annotations.json')
pseudo_path = osp.join(pseudo_label_dir, 'annotations_photobox2real_new_18skus_iter1_0.98.json')

coco_real = COCO(real_path)
coco_pseudo = COCO(pseudo_path)
print ('--- --- End Loading Annotations and Predictions --- ---')

imgs = coco_pseudo.imgs
cats = coco_pseudo.cats
pseudo_anns = coco_pseudo.imgToAnns

real_imgs = coco_real.imgs
real_anns = coco_real.imgToAnns

good_pred = dict()
pseudo_pred = dict()
total_pred = dict()

for i in range(1, 200):
    good_pred[i] = 0
    pseudo_pred[i] = 0
    total_pred[i] = 0

pbar = tqdm(total=len(imgs))
for num, pseudo_ann in pseudo_anns.items():
    img_id = pseudo_ann[0]['image_id']
    file_name = imgs[img_id]['file_name']
    for real_num, real_img in real_imgs.items():
        if real_img['file_name'] == file_name:
            real_id = real_img['id']
            break
    real_ann = real_anns[real_id]
    for ann in pseudo_ann:
        cat_id = ann['category_id']
        pseudo_pred[cat_id] += 1
    for ann in real_ann:
        for pse_ann in pseudo_ann:
            pred_box = pse_ann['bbox']
            gt_box = ann['bbox']
            abs_err = sum([abs(pred_box[i] - gt_box[i]) for i in range(4)])
            if abs_err < 100 and pse_ann['category_id'] == ann['category_id']:
                cat_id = pse_ann['category_id']
                good_pred[cat_id] += 1
        true_cat_id = ann['category_id']
        total_pred[true_cat_id] += 1

    pbar.update(1)

pbar.close()

for i in range(1, 200):
    if pseudo_pred[i] == 0 and total_pred[i] == 0:
        continue
    else:
        print('cat: {} , good pred: {} , total pred: {} , total gt: {} , ratio_1: {} , ratio_2: {} .'.format(
            str(i), good_pred[i], pseudo_pred[i], total_pred[i],
            good_pred[i] / (pseudo_pred[i] + 0.01), good_pred[i] / (total_pred[i] + 0.01)))
