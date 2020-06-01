import os
import argparse
import cv2

import os.path as osp
import numpy as np

from tqdm import tqdm
from pycocotools.coco import COCO


def scoring(item):
    return item['score']


parser = argparse.ArgumentParser(description='Failure Cases Analysis')
parser.add_argument('-i', '--img-dir', action='store', dest='img_dir', help='image directory')
parser.add_argument('-b', '--bbox-dir', action='store', dest='bbox_dir', help='bounding box directory')
parser.add_argument('-s', '--save-dir', action='store', dest='save_dir', help='save directory')
parser.add_argument('-n', '--num', action='store', dest='num_of_skus', help='number of SKUs in an image')

args = parser.parse_args()

img_dir = args.img_dir
bbox_dir = args.bbox_dir
save_dir = args.save_dir
num_of_skus = int(args.num_of_skus)

# For example:
# img_dir = '../testing_images'
# bbox_dir = '../sku-box-test-101-200-v1'
# save_dir = '../1_Products'

if not osp.exists(save_dir):
    os.mkdir(save_dir)
if not osp.exists(osp.join(save_dir, 'None_Detections')):
    os.mkdir(osp.join(save_dir, 'None_Detections'))
if not osp.exists(osp.join(save_dir, '{}_Product_fix'.format(str(num_of_skus)))):
    os.mkdir(osp.join(save_dir, '{}_Product_fix'.format(str(num_of_skus))))

# load annotations
anno_path = osp.join(img_dir, 'annotations_train.json')
# DO NOT USE
bbox_path = osp.join(bbox_dir, 'bbox_fix_done.json')

coco = COCO(anno_path)
cocoDt = coco.loadRes(bbox_path)

print ('--- --- End Loading Annotations and Predictions --- ---')

# read img
imgs = coco.imgs
cats = coco.cats
anns = coco.imgToAnns
dets = cocoDt.imgToAnns

gt_color = (0, 255, 255)
det_color = (0, 0, 255)
thick = 2

inconsistency_flag = False
pbar = tqdm(total=len(imgs))

for key, val in imgs.items():
    key = int(key)
    # print (key, val)
    file_path = osp.join(img_dir, val['file_name'])
    gt_im = cv2.imread(file_path)
    det_im = cv2.imread(file_path)

    h = val['height']
    num_bbox = len(anns[key])
    print(num_bbox)

    if num_bbox != num_of_skus:
        continue

    gt_cls_all = []
    det_cls_all = []
    inconsistency_flag = False

    for idx in range(0, num_bbox):

        gt_bbox = anns[key][idx]['bbox']
        gt_bbox = [int(p) for p in gt_bbox]
        gt_catId = anns[key][idx]['category_id']
        gt_cls = str(cats[gt_catId]['name'])
        gt_cls_all.append(gt_cls)
        gt_score = 1
        gt_title = 'GT: %s:%.2f' % (gt_cls, gt_score)

        print("--- gt_box: ", gt_bbox)
        cv2.rectangle(gt_im,
                      (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]),
                      gt_color, thick)
        cv2.putText(gt_im, gt_title, (gt_bbox[0], gt_bbox[3] + 8),
                    0, 8e-4 * h, gt_color, thick // 3)

        # cv2.imwrite(osp.join(save_dir, '{}_Product_fix'.format(str(num_of_skus)), val['file_name']), gt_im)

    if len(dets[key]) == 0:
        print('{} None detections'.format(key))
        im = np.hstack((det_im, gt_im))
        cv2.imwrite(osp.join(save_dir, 'None_Detections', val['file_name']), im)
        continue

    dets_sorted = sorted(dets[key], key=scoring, reverse=True)
    num_det_bbox = len(dets_sorted)
    # not drawing all the detected bounding boxes
    if num_det_bbox > num_bbox:
        num_det_bbox = num_bbox

    for idx in range(0, num_det_bbox):

        det_bbox = dets_sorted[idx]['bbox']
        det_catId = dets_sorted[idx]['category_id']
        det_cls = str(cats[det_catId]['name'])
        det_cls_all.append(det_cls)
        det_socre = dets_sorted[idx]['score']
        det_title = 'DET: %s:%.2f' % (det_cls, det_socre)

        # print ("--- det_box: ", det_bbox)
        cv2.rectangle(det_im,
                      (int(det_bbox[0]), int(det_bbox[1])), (int(det_bbox[2]), int(det_bbox[3])),
                      det_color, thick)
        cv2.putText(det_im, det_title, (int(det_bbox[0]), int(det_bbox[1]) - 8),
                    0, 8e-4 * h, det_color, thick // 3)

    im = np.hstack((det_im, gt_im))

    for idx in range(0, len(gt_cls_all)):
        if gt_cls_all[idx] not in det_cls_all:
            inconsistency_flag = True
    if inconsistency_flag:
        print(key)
        cv2.imwrite(osp.join(save_dir, '{}_Product_fix'.format(str(num_of_skus)), val['file_name']), im)

    pbar.update(1)

pbar.close()
